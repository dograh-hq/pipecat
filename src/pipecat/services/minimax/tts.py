#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax text-to-speech service implementation.

This module provides integration with MiniMax's T2A (Text-to-Audio) API
for streaming text-to-speech synthesis, over HTTP (``MiniMaxHttpTTSService``)
or over the bidirectional WebSocket API (``MiniMaxTTSService``).
"""

import asyncio
import json
import math
import random
import re
from collections import deque
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Self

import aiohttp
from loguru import logger
from pydantic import BaseModel
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.errors import (
    ErrorCategory,
    classify_http_exception,
    classify_http_status_code,
)
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 0.2
_RETRY_JITTER_S = 0.15
_RETRYABLE_CATEGORIES = frozenset(
    {ErrorCategory.RATE_LIMIT, ErrorCategory.CONNECTIVITY, ErrorCategory.SERVER}
)


def language_to_minimax_language(language: Language) -> str:
    """Convert a Language enum to MiniMax language format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding MiniMax language name. If ``language`` is not in
        the verified mapping, falls back to its base language's name (``pt-BR``
        becomes ``Portuguese``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).
    """
    LANGUAGE_MAP = {
        Language.AF: "Afrikaans",
        Language.AR: "Arabic",
        Language.BG: "Bulgarian",
        Language.CA: "Catalan",
        Language.CS: "Czech",
        Language.DA: "Danish",
        Language.DE: "German",
        Language.EL: "Greek",
        Language.EN: "English",
        Language.ES: "Spanish",
        Language.FA: "Persian",  # ⚠️ Only supported by speech-2.6-* models
        Language.FI: "Finnish",
        Language.FIL: "Filipino",  # ⚠️ Only supported by speech-2.6-* models
        Language.FR: "French",
        Language.HE: "Hebrew",
        Language.HI: "Hindi",
        Language.HR: "Croatian",
        Language.HU: "Hungarian",
        Language.ID: "Indonesian",
        Language.IT: "Italian",
        Language.JA: "Japanese",
        Language.KO: "Korean",
        Language.MS: "Malay",
        Language.NB: "Norwegian",
        Language.NN: "Nynorsk",
        Language.NL: "Dutch",
        Language.PL: "Polish",
        Language.PT: "Portuguese",
        Language.RO: "Romanian",
        Language.RU: "Russian",
        Language.SK: "Slovak",
        Language.SL: "Slovenian",
        Language.SV: "Swedish",
        Language.TA: "Tamil",  # ⚠️ Only supported by speech-2.6-* models
        Language.TH: "Thai",
        Language.TR: "Turkish",
        Language.UK: "Ukrainian",
        Language.VI: "Vietnamese",
        Language.YUE: "Chinese,Yue",
        Language.ZH: "Chinese",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class MiniMaxTTSSettings(TTSSettings):
    """Settings for MiniMaxHttpTTSService.

    Parameters:
        speed: Speech speed (range: 0.5 to 2.0).
        volume: Speech volume (range: 0 to 10).
        pitch: Pitch adjustment (range: -12 to 12).
        emotion: Emotional tone (options: "happy", "sad", "angry", "fearful",
            "disgusted", "surprised", "calm", "fluent").
        text_normalization: Enable text normalization (Chinese/English).
        latex_read: Enable LaTeX formula reading.
        language_boost: Language boost string for multilingual support.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pitch: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emotion: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    latex_read: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_boost: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> Self:
        """Construct settings from a plain dict, destructuring legacy nested dicts.

        Handles ``voice_setting`` (with ``vol`` → ``volume`` rename) and
        ``audio_setting`` (with prefixed field mapping).
        """
        flat = dict(settings)

        voice = flat.pop("voice_setting", None)
        if isinstance(voice, dict):
            flat.setdefault("speed", voice.get("speed"))
            flat.setdefault("volume", voice.get("vol"))
            flat.setdefault("pitch", voice.get("pitch"))
            flat.setdefault("emotion", voice.get("emotion"))
            flat.setdefault("text_normalization", voice.get("text_normalization"))
            flat.setdefault("latex_read", voice.get("latex_read"))

        return super().from_mapping(flat)


# What a MiniMax status code says about whether the request could ever
# succeed. A permanent category - authentication, authorization, invalid
# request - costs the service its usability the moment it is reported, so a
# rejected key, an unknown voice or a malformed setting stops the call on the
# first turn rather than leaving the line silent until the consecutive-silence
# watchdog gives up. An exhausted balance (1008) also makes the service
# unusable; a usage-window quota (2056) can recover on a later turn.
#
# Codes that reject one piece of text rather than the configuration are
# deliberately absent, and so report as UNKNOWN: the content-safety codes
# (1026, 1027) and the invisible-character limit (1042) say nothing about
# whether the turn after them can be spoken. Voice-cloning codes are absent
# because T2A never returns them.
#
# https://platform.minimax.io/docs/api-reference/errorcode
_STATUS_CODE_CATEGORIES: dict[int, ErrorCategory] = {
    1001: ErrorCategory.CONNECTIVITY,  # request timeout
    1002: ErrorCategory.RATE_LIMIT,
    1004: ErrorCategory.AUTHENTICATION,  # not authorized / token not match group
    1008: ErrorCategory.QUOTA,  # insufficient balance
    1024: ErrorCategory.SERVER,  # internal error
    1033: ErrorCategory.SERVER,  # system error
    1039: ErrorCategory.RATE_LIMIT,  # token limit
    1041: ErrorCategory.RATE_LIMIT,  # connection limit
    2013: ErrorCategory.INVALID_REQUEST,  # invalid params
    2042: ErrorCategory.AUTHORIZATION,  # no access to this voice_id
    2045: ErrorCategory.RATE_LIMIT,  # rate growth limit
    2049: ErrorCategory.AUTHENTICATION,  # invalid API key
    2056: ErrorCategory.QUOTA,  # usage limit exceeded for this window
    20132: ErrorCategory.INVALID_REQUEST,  # invalid samples or voice_id
}


def _base_resp_error(payload: dict) -> ErrorFrame | None:
    """Return an error frame if a MiniMax payload reports a failure, else None.

    MiniMax answers HTTP 200 even when it rejects a request, and puts the real
    outcome in ``base_resp``, which rides on a non-streamed body and on every
    streaming chunk alike. ``status_code`` 0 is success; anything else means no
    audio is coming. Checking only the HTTP status therefore produces a call
    that is answered, billed and completely silent, recorded as a healthy
    synthesis - the failure reaches the person on the line and nobody else.

    The frame carries the category its status code maps to, which is what
    decides whether the service is left able to attempt the next turn.

    https://platform.minimax.io/docs/api-reference/speech-t2a-http
    """
    base_resp = payload.get("base_resp")
    if not isinstance(base_resp, dict):
        return None

    status_code = base_resp.get("status_code")
    if not status_code:
        # 0, None and absent all mean there is nothing wrong to report.
        return None

    status_msg = base_resp.get("status_msg") or "no message"
    # trace_id is the first thing MiniMax support asks for.
    trace_id = payload.get("trace_id")
    suffix = f" (trace_id={trace_id})" if trace_id else ""
    return ErrorFrame(
        error=f"MiniMax TTS error: {status_code} {status_msg}{suffix}",
        category=_STATUS_CODE_CATEGORIES.get(status_code, ErrorCategory.UNKNOWN),
    )


@dataclass
class MiniMaxSynthesisOutcome:
    """Request-local completion evidence and retry timing from the current attempt."""

    completed: bool = False
    retry_after_secs: float = 0


def _retry_after_seconds(value: str | None) -> float:
    """Read an HTTP Retry-After delay or date; ignore malformed provider hints."""
    if value is None:
        return 0
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            seconds = (when - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0
    return max(0, seconds) if math.isfinite(seconds) else 0


async def _response_payloads(response: aiohttp.ClientResponse) -> AsyncGenerator[dict, None]:
    """Decode SSE events or a plain JSON response, including an unterminated final event."""
    pending = bytearray()
    event = bytearray()
    is_json = None
    max_event_bytes = 16 * 1024 * 1024

    def decode(raw: bytes | bytearray) -> dict:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("MiniMax response must be a JSON object")
        return payload

    def line_received(line: bytes) -> dict | None:
        line = line.rstrip(b"\r")
        if not line:
            if event:
                payload = decode(event)
                event.clear()
                return payload
        elif line.startswith(b"data:"):
            event.extend(line[5:].lstrip(b" "))
            event.extend(b"\n")
            if len(event) > max_event_bytes:
                raise ValueError("MiniMax event exceeds the size limit")
        elif not line.startswith((b":", b"event:", b"id:", b"retry:")):
            raise ValueError("Invalid MiniMax SSE event")
        return None

    async for chunk in response.content.iter_any():
        pending.extend(chunk)
        if is_json is None and pending.strip():
            is_json = pending.lstrip().startswith((b"{", b"["))
        if not is_json:
            while b"\n" in pending:
                line, _, remainder = pending.partition(b"\n")
                pending = bytearray(remainder)
                payload = line_received(line)
                if payload is not None:
                    yield payload
        if len(pending) > max_event_bytes:
            raise ValueError("MiniMax response exceeds the size limit")

    if is_json:
        yield decode(pending)
    else:
        if pending:
            payload = line_received(bytes(pending))
            if payload is not None:
                yield payload
        if event:
            yield decode(event)


class MiniMaxHttpTTSService(TTSService):
    """Text-to-speech service using MiniMax's T2A (Text-to-Audio) API.

    Provides streaming text-to-speech synthesis using MiniMax's HTTP API
    with support for various voice settings, emotions, and audio configurations.
    Supports real-time audio streaming with configurable voice parameters.

    Platform documentation:
    https://platform.minimax.io/docs/api-reference/speech-t2a-http
    """

    Settings = MiniMaxTTSSettings
    _settings: Settings

    @deprecated(
        "`MiniMaxHttpTTSService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `MiniMaxHttpTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for MiniMax TTS.

        .. deprecated:: 0.0.105
            Use ``MiniMaxHttpTTSService.Settings`` directly via the ``settings`` parameter instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language for TTS generation. Supports 40 languages.
                Note: Filipino, Tamil, and Persian require speech-2.6-* models.
            speed: Speech speed (range: 0.5 to 2.0).
            volume: Speech volume (range: 0 to 10).
            pitch: Pitch adjustment (range: -12 to 12).
            emotion: Emotional tone (options: "happy", "sad", "angry", "fearful",
                "disgusted", "surprised", "calm", "fluent").
            text_normalization: Enable text normalization (Chinese/English).
            latex_read: Enable LaTeX formula reading.
            exclude_aggregated_audio: Whether to exclude aggregated audio in final chunk.
        """

        language: Language | None = Language.EN
        speed: float | None = 1.0
        volume: float | None = 1.0
        pitch: int | None = 0
        emotion: str | None = None
        text_normalization: bool | None = None
        latex_read: bool | None = None
        exclude_aggregated_audio: bool | None = None

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1/t2a_v2",
        group_id: str,
        model: str | None = None,
        voice_id: str | None = None,
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: int | None = None,
        stream: bool = True,
        params: InputParams | None = None,
        settings: Settings | None = None,
        retry_timeout_secs: float = 5.0,
        **kwargs,
    ):
        """Initialize the MiniMax TTS service.

        Args:
            api_key: MiniMax API key for authentication.
            base_url: API base URL, defaults to MiniMax's T2A endpoint.
                Global: https://api.minimax.io/v1/t2a_v2
                Mainland China: https://api.minimaxi.chat/v1/t2a_v2
                Western United States: https://api-uw.minimax.io/v1/t2a_v2
            group_id: MiniMax Group ID to identify project.
            model: TTS model name. Defaults to "speech-2.8-turbo".

                .. deprecated:: 0.0.105
                    Use ``settings=MiniMaxHttpTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            voice_id: Voice identifier. Defaults to "Calm_Woman".

                .. deprecated:: 0.0.105
                    Use ``settings=MiniMaxHttpTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            aiohttp_session: aiohttp.ClientSession for API communication.
            sample_rate: Output audio sample rate in Hz. If None, uses pipeline default.
            stream: Whether to use streaming mode. Defaults to True.
            params: Additional configuration parameters.

                .. deprecated:: 0.0.105
                    Use ``settings=MiniMaxHttpTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            retry_timeout_secs: Maximum time to first audio across the initial request
                and up to two transient-error retries, including backoff. Once audio
                starts, synthesis is not retried and the session's HTTP timeouts apply.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        if not math.isfinite(retry_timeout_secs) or retry_timeout_secs <= 0:
            raise ValueError("retry_timeout_secs must be positive and finite")
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="speech-2.8-turbo",
            voice="Calm_Woman",
            language=None,
            speed=1.0,
            volume=1.0,
            pitch=0,
            language_boost=None,
            emotion=None,
            text_normalization=None,
            latex_read=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.speed = params.speed
                default_settings.volume = params.volume
                default_settings.pitch = params.pitch
                default_settings.latex_read = params.latex_read

                # Resolve language boost
                if params.language:
                    service_lang = self.language_to_service_language(params.language)
                    if service_lang:
                        default_settings.language_boost = service_lang

                # Resolve emotion
                if params.emotion:
                    supported_emotions = [
                        "happy",
                        "sad",
                        "angry",
                        "fearful",
                        "disgusted",
                        "surprised",
                        "neutral",
                        "fluent",
                    ]
                    if params.emotion in supported_emotions:
                        default_settings.emotion = params.emotion
                    else:
                        logger.warning(
                            f"Unsupported emotion: {params.emotion}. Supported emotions: {supported_emotions}"
                        )

                # Resolve text_normalization
                if params.text_normalization is not None:
                    default_settings.text_normalization = params.text_normalization

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._group_id = group_id
        self._stream = stream
        self._base_url = f"{base_url}?GroupId={group_id}"
        self._session = aiohttp_session
        self._retry_timeout_secs = retry_timeout_secs

        # Init-only audio format config
        self._audio_bitrate = 128000
        self._audio_format = "pcm"
        self._audio_channel = 1
        self._audio_sample_rate = 0  # Set in start()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as MiniMax service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to MiniMax service language format.

        Args:
            language: The language to convert.

        Returns:
            The MiniMax-specific language name, or None if not supported.
        """
        return language_to_minimax_language(language)

    async def start(self, frame: StartFrame):
        """Start the MiniMax TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._audio_sample_rate = self.sample_rate
        logger.debug(f"MiniMax TTS initialized with sample_rate: {self.sample_rate}")

    def _build_request(self, text: str) -> dict:
        """Snapshot the effective synthesis payload for a single request."""
        # Build voice_setting dict for API
        voice_setting = {
            "voice_id": self._settings.voice,
            "speed": self._settings.speed,
            "vol": self._settings.volume,
            "pitch": self._settings.pitch,
        }
        if self._settings.emotion is not None:
            voice_setting["emotion"] = self._settings.emotion
        if self._settings.text_normalization is not None:
            voice_setting["text_normalization"] = self._settings.text_normalization
        if self._settings.latex_read is not None:
            voice_setting["latex_read"] = self._settings.latex_read

        # Build audio_setting dict for API
        audio_setting = {
            "bitrate": self._audio_bitrate,
            "format": self._audio_format,
            "channel": self._audio_channel,
            "sample_rate": self._audio_sample_rate,
        }

        payload = {
            "stream": self._stream,
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            "model": self._settings.model,
            "text": text,
        }
        if self._settings.language_boost is not None:
            payload["language_boost"] = self._settings.language_boost

        if self._stream:
            payload["stream_options"] = {"exclude_aggregated_audio": True}
        return payload

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize one request and yield its PCM frames or a provider error."""
        outcome = MiniMaxSynthesisOutcome()
        async with aclosing(
            self._run_tts_request(self._build_request(text), context_id, outcome)
        ) as frames:
            async for frame in frames:
                yield frame

    async def _run_tts_request(
        self, payload: dict, context_id: str, outcome: MiniMaxSynthesisOutcome
    ) -> AsyncGenerator[Frame, None]:
        """Retry transient refusals within one deadline, until the first audio is emitted."""
        outcome.completed = False
        audio_emitted = False
        error = None
        deadline_at = asyncio.get_running_loop().time() + self._retry_timeout_secs
        keepalive = self.create_task(self._keep_request_context_alive(context_id))
        try:
            try:
                async with asyncio.timeout_at(deadline_at) as deadline:
                    for attempt in range(1, _MAX_ATTEMPTS + 1):
                        error = None
                        async with aclosing(
                            self._run_tts_once(payload, context_id, outcome)
                        ) as frames:
                            async for frame in frames:
                                if isinstance(frame, ErrorFrame):
                                    error = frame
                                    break
                                if isinstance(frame, TTSAudioRawFrame) and frame.audio:
                                    if not audio_emitted:
                                        audio_emitted = True
                                        deadline.reschedule(None)
                                        if keepalive is not None:
                                            await self.cancel_task(keepalive)
                                            keepalive = None
                                        await self.start_tts_usage_metrics(payload["text"])
                                        await self.stop_ttfb_metrics()
                                yield frame

                        if error is None:
                            return
                        if (
                            audio_emitted
                            or error.category not in _RETRYABLE_CATEGORIES
                            or attempt == _MAX_ATTEMPTS
                        ):
                            break

                        delay = _RETRY_BASE_DELAY_S * 2 ** (attempt - 1) + random.uniform(
                            0, _RETRY_JITTER_S
                        )
                        delay = max(delay, outcome.retry_after_secs)
                        if asyncio.get_running_loop().time() + delay >= deadline_at:
                            break
                        logger.warning(
                            "{}: MiniMax TTS attempt {}/{} failed ({}); retrying in {:.2f}s",
                            self,
                            attempt,
                            _MAX_ATTEMPTS,
                            error.error,
                            delay,
                        )
                        await asyncio.sleep(delay)
            except TimeoutError as exc:
                error = ErrorFrame(
                    error="MiniMax TTS timed out before first audio",
                    exception=exc,
                    category=ErrorCategory.CONNECTIVITY,
                )
            if error is not None:
                yield error
        finally:
            if keepalive is not None:
                await self.cancel_task(keepalive)
            await self.stop_ttfb_metrics()

    async def _keep_request_context_alive(self, context_id: str) -> None:
        """Prevent audio-context expiry while bounded synthesis/retries await first audio."""
        while True:
            self._refresh_audio_context(context_id)
            await asyncio.sleep(min(1.0, self._stop_frame_timeout_s / 2))

    async def _run_tts_once(
        self, payload: dict, context_id: str, outcome: MiniMaxSynthesisOutcome
    ) -> AsyncGenerator[Frame, None]:
        """Execute a payload, confirming completion only after a clean, valid response."""
        outcome.completed = False
        outcome.retry_after_secs = 0
        headers = {
            "accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        audio_settings = payload["audio_setting"]
        sample_rate = audio_settings["sample_rate"]
        channels = audio_settings["channel"]
        alignment = 2 * channels
        pcm = bytearray()
        received_audio = False
        completed = False
        try:
            async with self._session.post(
                self._base_url, headers=headers, json=payload
            ) as response:
                outcome.retry_after_secs = _retry_after_seconds(response.headers.get("Retry-After"))
                if response.status != 200:
                    yield ErrorFrame(
                        error=f"MiniMax TTS error: HTTP {response.status}",
                        category=classify_http_status_code(response.status),
                    )
                    return

                async for data in _response_payloads(response):
                    error = _base_resp_error(data)
                    if error:
                        if data["base_resp"]["status_code"] == 1008:
                            # Credits must be replenished before any later turn can speak.
                            await self.set_usable(False)
                        yield error
                        return
                    chunk_data = data.get("data") or {}
                    status = chunk_data.get("status")
                    if completed:
                        raise ValueError("MiniMax sent data after synthesis completed")
                    audio_hex = chunk_data.get("audio") or ""
                    audio = bytes.fromhex(audio_hex)
                    # A streaming status-2 event may repeat the entire response.
                    # Its audio is never appended to the incremental stream.
                    if not payload["stream"] or status != 2:
                        pcm.extend(audio)
                    if status == 2:
                        completed = True
                    elif status not in (None, 1):
                        raise ValueError("Unknown MiniMax synthesis status")

                    size = max(alignment, self.chunk_size // alignment * alignment)
                    while len(pcm) >= alignment:
                        count = min(size, len(pcm) // alignment * alignment)
                        audio_chunk = bytes(pcm[:count])
                        del pcm[:count]
                        received_audio = True
                        yield TTSAudioRawFrame(
                            audio=audio_chunk,
                            sample_rate=sample_rate,
                            num_channels=channels,
                            context_id=context_id,
                        )

                if not completed:
                    raise ValueError("MiniMax response ended before synthesis completed")
                if pcm:
                    raise ValueError("MiniMax returned incomplete PCM samples")
                if not received_audio:
                    raise ValueError("MiniMax synthesis completed without audio")
            outcome.completed = True
        except Exception as e:
            yield ErrorFrame(
                error=f"MiniMax TTS response error: {e}",
                exception=e,
                category=(
                    ErrorCategory.CONNECTIVITY
                    if isinstance(e, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError))
                    else classify_http_exception(e)
                ),
            )


# Status codes only the WebSocket APIs return.
_WEBSOCKET_STATUS_CODE_CATEGORIES: dict[int, ErrorCategory] = {
    2201: ErrorCategory.CONNECTIVITY,  # closed after ~120 s with no events
}


def _status_error(payload: dict) -> ErrorFrame | None:
    """Error frame for a failed MiniMax WebSocket event, categorised like the HTTP API's codes."""
    error = _base_resp_error(payload)
    if error is not None and error.category == ErrorCategory.UNKNOWN:
        code = (payload.get("base_resp") or {}).get("status_code")
        error.category = _WEBSOCKET_STATUS_CODE_CATEGORIES.get(code, ErrorCategory.UNKNOWN)
    return error


def _subtitle_words(subtitle: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    """Rebuild whole words, with start and end seconds, from one sentence's subtitle.

    MiniMax's ``timestamped_words`` are sub-word pieces ("B", "u", "ong", "ior",
    "no", then a space entry, then "," ...) carrying character offsets into the
    subtitle text. A word is a whitespace-separated span of that text; it starts
    with its first piece and ends with its last.
    """
    text = subtitle.get("text") or ""
    pieces = [p for p in subtitle.get("timestamped_words") or [] if (p.get("word") or "").strip()]
    words = []
    for match in re.finditer(r"\S+", text):
        inside = [
            p for p in pieces if p["word_begin"] < match.end() and p["word_end"] > match.start()
        ]
        if inside:
            start = min(p["time_begin"] for p in inside) / 1000
            end = max(p["time_end"] for p in inside) / 1000
            words.append((match.group(), start, end))
    return words


@dataclass
class _Synthesis:
    """One audio context as MiniMax sees it: texts sent on the shared session, then one flush."""

    context_id: str
    audio_bytes: int = 0
    sentence_offset_secs: float = 0.0
    flushed: bool = False
    stale: bool = False
    pcm_carry: bytes = b""


class MiniMaxTTSService(WebsocketTTSService):
    """MiniMax text-to-speech over the bidirectional WebSocket API, with word timestamps.

    Uses ``/ws/v1/t2a_v2_bidi``, which, unlike ``/ws/v1/t2a_v2``, can interrupt
    synthesis (``task_cancel``) without reconnecting and can end a reply
    (``task_flush``) without closing the session. One session lives for the
    whole call; client pings keep it past MiniMax's 120 s idle limit.

    The protocol has no per-request id the client can choose, so audio is matched
    to contexts by order, and interruptions need care:

    * After ``task_cancel``, MiniMax keeps sending the cancelled sentence's
      current audio block (up to ~1.4 s of speech) until ``task_canceled``. A
      sentence whose synthesis had started but whose first chunk had not yet
      arrived can still deliver that block under a trace id never seen before.
      Audio is therefore dropped until ``task_canceled`` *and* a later
      ``sentence_start``, and any trace id seen while dropping stays dropped.
    * Each flush still pending at the cancel is acknowledged with a
      ``task_flushed`` after ``task_canceled``. Those acknowledgements are matched
      to their cancelled contexts so they cannot close the next reply early.

    Subtitles arrive per sentence with times starting at 0. Each sentence is
    offset by the audio already received for its context, and a word is reported
    once ``word_min_played_secs`` (or half of it, for short words) has played, so
    an interrupted reply's transcript holds only words the caller could hear.

    Platform documentation:
    https://platform.minimax.io/docs/api-reference/speech-t2a-websocket-bidi
    """

    Settings = MiniMaxTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.minimax.io/ws/v1/t2a_v2_bidi",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        word_min_played_secs: float = 0.08,
        **kwargs,
    ):
        """Initialize the MiniMax WebSocket TTS service.

        Args:
            api_key: MiniMax API key for authentication.
            url: Bidirectional WebSocket endpoint.
            sample_rate: Output audio sample rate in Hz. If None, uses pipeline default.
                MiniMax accepts 8000, 16000, 22050, 24000, 32000 and 44100.
            settings: Runtime-updatable settings. A change to any of them restarts the session.
            word_min_played_secs: How much of a word must play before it counts as spoken.
                Words shorter than twice this count once half of them has played.
            **kwargs: Additional arguments passed to the parent WebsocketTTSService.
        """
        default_settings = self.Settings(
            model="speech-2.8-turbo",
            voice="Calm_Woman",
            language=None,
            speed=1.0,
            volume=1.0,
            pitch=0,
            language_boost=None,
            emotion=None,
            text_normalization=None,
            latex_read=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_text_frames=False,
            push_stop_frames=False,
            push_start_frame=True,
            # Replies share one ordered stream, so hold the next one until this one has played.
            pause_frame_processing=kwargs.pop("pause_frame_processing", True),
            # MiniMax joins task_continue texts verbatim: without a separator,
            # "azienda." + "Vecchia" reaches it as "azienda.Vecchia" and its
            # sentence segmentation splits in the wrong place.
            append_trailing_space=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._word_min_played_secs = word_min_played_secs
        self._receive_task = None
        self._reset_stream_state()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as MiniMax service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to MiniMax service language format.

        Args:
            language: The language to convert.

        Returns:
            The MiniMax-specific language name, or None if not supported.
        """
        return language_to_minimax_language(language)

    def _reset_stream_state(self):
        # Contexts whose audio may still arrive, in the order their text was sent.
        self._syntheses: deque[_Synthesis] = deque()
        # One entry per task_flush awaiting its task_flushed, in send order.
        self._flushes: deque[_Synthesis] = deque()
        self._cancels_pending = 0
        self._awaiting_sentence_start = False
        self._stale_traces: set[str] = set()
        self._dropped_bytes = 0

    @property
    def _dropping_audio(self) -> bool:
        return self._cancels_pending > 0 or self._awaiting_sentence_start

    def _task_start_message(self) -> dict:
        voice_setting = {
            "voice_id": self._settings.voice,
            "speed": self._settings.speed,
            "vol": self._settings.volume,
            "pitch": self._settings.pitch,
        }
        if self._settings.emotion is not None:
            voice_setting["emotion"] = self._settings.emotion
        if self._settings.latex_read is not None:
            voice_setting["latex_read"] = self._settings.latex_read
        message = {
            "event": "task_start",
            "model": self._settings.model,
            "voice_setting": voice_setting,
            "audio_setting": {"sample_rate": self.sample_rate, "format": "pcm", "channel": 1},
            "subtitle_enable": True,
            "subtitle_type": "word",
        }
        if self._settings.text_normalization:
            message["text_normalization_mode"] = "basic"
        if self._settings.language_boost is not None:
            message["language_boost"] = self._settings.language_boost
        return message

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and open the session, so the first reply does not pay for it.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        # The receive loop ends when MiniMax closes the session (task_failed, idle
        # limit), so a finished task is replaced as well as a missing one.
        if self._websocket and (not self._receive_task or self._receive_task.done()):
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug(f"{self}: connecting to MiniMax")
            self._reset_stream_state()
            websocket = await self._websocket_connect(
                self._url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            )
            for expected, outgoing in (
                ("connected_success", None),
                ("task_started", self._task_start_message()),
            ):
                if outgoing is not None:
                    await websocket.send(json.dumps(outgoing))
                reply = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
                error = _status_error(reply)
                if error is not None or reply.get("event") != expected:
                    await websocket.close()
                    if (
                        error is not None
                        and (reply.get("base_resp") or {}).get("status_code") == 1008
                    ):
                        await self.set_usable(False)
                    await self.push_error_frame(
                        error
                        or ErrorFrame(
                            error=f"MiniMax TTS: expected {expected}, got {reply.get('event')}"
                        )
                    )
                    return
            self._websocket = websocket
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}", exception=e))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: disconnecting from MiniMax")
                await self._websocket.close()
        except Exception as e:
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}", exception=e))
        finally:
            self._reset_stream_state()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta; MiniMax fixes them at task_start, so the session restarts.

        Args:
            delta: A settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
            await self._disconnect()
            await self._connect()
        return changed

    async def _send_flush(self, synthesis: _Synthesis):
        await self._get_websocket().send(json.dumps({"event": "task_flush"}))
        synthesis.flushed = True
        self._flushes.append(synthesis)

    async def flush_audio(self, context_id: str | None = None):
        """End a reply: MiniMax synthesizes its unpunctuated tail and acknowledges with task_flushed.

        Args:
            context_id: The context to flush. If None, falls back to the active context.
        """
        context_id = context_id or self.get_active_audio_context_id()
        for synthesis in self._syntheses:
            if synthesis.context_id == context_id and not synthesis.flushed and self._websocket:
                try:
                    await self._send_flush(synthesis)
                except Exception as e:
                    logger.error(f"{self} error sending task_flush: {e}")
                return

    async def _cancel_outstanding(self):
        """Cancel whatever MiniMax is still producing, and start dropping it.

        Sent once per interruption, and only when MiniMax still owes audio or a
        flush acknowledgement: everything it sends before task_canceled belongs
        to the cancelled replies.
        """
        if not (self._syntheses or self._flushes):
            return
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._websocket.send(json.dumps({"event": "task_cancel"}))
        except Exception as e:
            logger.error(f"{self} error sending task_cancel: {e}")
            return
        self._cancels_pending += 1
        self._awaiting_sentence_start = True
        for synthesis in self._syntheses:
            synthesis.stale = True
        self._syntheses.clear()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await self._cancel_outstanding()
        await super()._handle_interruption(frame, direction)

    async def _finish(self, synthesis: _Synthesis):
        if self.audio_context_available(synthesis.context_id):
            await self.append_to_audio_context(
                synthesis.context_id, TTSStoppedFrame(context_id=synthesis.context_id)
            )
            await self.remove_audio_context(synthesis.context_id)

    async def _on_audio(self, message: dict, data: dict):
        trace = message.get("trace_id")
        if self._dropping_audio or trace in self._stale_traces:
            if trace:
                self._stale_traces.add(trace)
            self._dropped_bytes += len(data["audio"]) // 2
            return
        if not self._syntheses:
            logger.warning(f"{self}: MiniMax audio arrived with no reply waiting for it; dropped")
            return
        synthesis = self._syntheses[0]
        pcm = synthesis.pcm_carry + bytes.fromhex(data["audio"])
        usable = len(pcm) - len(pcm) % 2
        synthesis.pcm_carry = pcm[usable:]
        if usable:
            await self.append_to_audio_context(
                synthesis.context_id,
                TTSAudioRawFrame(
                    audio=pcm[:usable],
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=synthesis.context_id,
                ),
            )
        subtitle = data.get("subtitle")
        if subtitle:
            # Times restart at 0 for every sentence; place them within the reply.
            offset = synthesis.sentence_offset_secs
            word_times = [
                (word, offset + start + min(self._word_min_played_secs, (end - start) / 2))
                for word, start, end in _subtitle_words(subtitle)
            ]
            if word_times:
                await self.add_word_timestamps(word_times, synthesis.context_id)
        synthesis.audio_bytes += usable

    async def _on_message(self, message: dict):
        event = message.get("event")
        data = message.get("data") or {}

        if event == "task_failed":
            error = _status_error(message) or ErrorFrame(
                error=f"MiniMax TTS task failed: {message}"
            )
            if (message.get("base_resp") or {}).get("status_code") == 1008:
                await self.set_usable(False)
            # MiniMax closes the session after this; end every reply it was serving,
            # once each (a flushed reply sits in both queues).
            pending = {id(s): s for s in [*self._syntheses, *self._flushes] if not s.stale}
            for synthesis in pending.values():
                await self._finish(synthesis)
            self._reset_stream_state()
            await self.stop_all_metrics()
            await self.push_error_frame(error)
            return

        error = _status_error(message)
        if error is not None:
            # A rejected piece of text (2204 too long, 2205 queue full) leaves the session open.
            await self.push_error_frame(error)
            return

        if event == "task_canceled":
            self._cancels_pending = max(0, self._cancels_pending - 1)
            if not self._cancels_pending and self._dropped_bytes:
                logger.debug(
                    f"{self}: dropped {self._dropped_bytes} bytes of cancelled MiniMax audio"
                )
                self._dropped_bytes = 0
        elif event == "sentence_start":
            if not self._cancels_pending:
                self._awaiting_sentence_start = False
                if self._syntheses:
                    synthesis = self._syntheses[0]
                    synthesis.sentence_offset_secs = synthesis.audio_bytes / 2 / self.sample_rate
        elif event == "task_flushed":
            synthesis = self._flushes.popleft() if self._flushes else None
            if synthesis is not None and not synthesis.stale:
                if synthesis in self._syntheses:
                    self._syntheses.remove(synthesis)
                await self._finish(synthesis)
        elif event == "task_continued" and data.get("audio"):
            await self._on_audio(message, data)

    async def _receive_messages(self):
        async for raw in self._get_websocket():
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.error(f"{self}: invalid JSON from MiniMax: {raw!r:.200}")
                continue
            await self._on_message(message)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text for synthesis; audio arrives on the receive task.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: None, or an ErrorFrame if the text could not be sent.
        """
        try:
            if not self._websocket or self._websocket.state is not State.OPEN:
                await self._connect()
            if not self._syntheses or self._syntheses[-1].context_id != context_id:
                # A reply that never flushed would run into this one on MiniMax's side.
                if self._syntheses and not self._syntheses[-1].flushed:
                    await self._send_flush(self._syntheses[-1])
                self._syntheses.append(_Synthesis(context_id=context_id))
            await self._get_websocket().send(json.dumps({"event": "task_continue", "text": text}))
            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"MiniMax TTS error: {e}", exception=e)
