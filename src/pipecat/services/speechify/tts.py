#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechify text-to-speech service implementation."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_speechify_language(language: Language) -> str:
    """Convert a Language enum to a Speechify language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding service language code. If ``language`` is not in
        the verified mapping, falls back to the base language code (e.g.,
        ``en`` from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).
    """
    LANGUAGE_MAP = {
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.IT: "it",
        Language.PT: "pt",
        Language.PT_BR: "pt-BR",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def output_format_from_sample_rate(sample_rate: int) -> str:
    """Get the Speechify PCM output format string for a given sample rate.

    Args:
        sample_rate: The audio sample rate in Hz.

    Returns:
        The Speechify output format string.

    Raises:
        ValueError: If the sample rate has no matching Speechify PCM format.
            Speechify would otherwise return audio at a different rate than
            the frames are labeled with, playing back at the wrong speed.
    """
    match sample_rate:
        case 8000 | 16000 | 22050 | 24000 | 44100 | 48000:
            return f"pcm_{sample_rate}"
    raise ValueError(
        f"SpeechifyTTSService: unsupported sample rate {sample_rate}; "
        "supported PCM rates are 8000, 16000, 22050, 24000, 44100, 48000"
    )


@dataclass
class SpeechifyTTSSettings(TTSSettings):
    """Settings for SpeechifyTTSService."""

    pass


class SpeechifyTTSService(TTSService):
    """Speechify HTTP streaming text-to-speech service.

    Streams raw PCM audio from Speechify's ``/v1/audio/stream`` endpoint.
    Supports the streaming-native Simba models (``simba-3.2``, ``simba-3.0``)
    as well as the legacy ``simba-english`` and ``simba-multilingual`` models.
    """

    Settings = SpeechifyTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.speechify.ai",
        aiohttp_session: aiohttp.ClientSession | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Speechify TTS service.

        Args:
            api_key: Speechify API key for authentication.
            base_url: Base URL for the Speechify API.
            aiohttp_session: Optional aiohttp ClientSession for HTTP requests.
                If not provided, a session will be created and managed internally.
            sample_rate: Audio sample rate. Must be one of Speechify's supported
                PCM rates (8000, 16000, 22050, 24000, 44100, 48000). If None,
                uses default.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        default_settings = self.Settings(
            model="simba-3.2",
            voice="beatrice_32",
            language=None,
        )

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
        self._base_url = base_url.rstrip("/")

        # Audio output format — set in start() from self.sample_rate
        self._output_format = "pcm_24000"

        self._session: aiohttp.ClientSession | None = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Speechify service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Speechify language format.

        Args:
            language: The language to convert.

        Returns:
            The Speechify-specific language code, or None if not supported.
        """
        return language_to_speechify_language(language)

    async def start(self, frame: StartFrame):
        """Start the Speechify TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        if self._owns_session:
            self._session = aiohttp.ClientSession()

    async def _close_session(self):
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None

    async def cleanup(self):
        """Close the owned HTTP session at teardown."""
        await super().cleanup()
        await self._close_session()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Speechify's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            if self._session is None:
                raise RuntimeError("HTTP session is not initialized; call start() before run_tts()")

            payload = {
                "input": text,
                "voice_id": self._settings.voice,
                "model": self._settings.model,
                "output_format": self._output_format,
            }

            if self._settings.language:
                payload["language"] = self._settings.language

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            url = f"{self._base_url}/v1/audio/stream"

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(
                        error=f"Speechify API error (status: {response.status}): {error_text}"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                # PCM samples are 16-bit; carry any odd trailing byte so frames
                # never split a sample.
                carry = b""
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    audio = carry + chunk
                    if len(audio) % 2:
                        audio, carry = audio[:-1], audio[-1:]
                    else:
                        carry = b""
                    if audio:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            audio=audio,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
