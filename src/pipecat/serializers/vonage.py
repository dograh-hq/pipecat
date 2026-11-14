#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vonage Audio Connector WebSocket serializer for Pipecat."""

import json
from typing import Any

from loguru import logger
from pydantic import model_validator

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.deprecation import warn_deprecated


class VonageFrameSerializer(FrameSerializer):
    """Serializer for Vonage Video API Audio Connector WebSocket protocol.

    This serializer converts between Pipecat frames and the Vonage Audio Connector
    WebSocket streaming protocol.

    Note:
        Ref docs: https://developer.vonage.com/en/video/guides/audio-connector
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for VonageFrameSerializer.

        Parameters:
            audio_in_sample_rate: Sample rate of the audio Vonage sends, defaults to
                16000 Hz. Common values: 8000, 16000, 24000 Hz.
            audio_out_sample_rate: Sample rate of the audio sent to Vonage, defaults to
                16000 Hz.
            vonage_sample_rate: Sample rate used by Vonage.

                .. deprecated:: 1.16.0
                    Use ``audio_in_sample_rate`` and ``audio_out_sample_rate`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Has no effect: incoming audio is converted to the pipeline's input rate.

                .. deprecated:: 1.16.0
                    Use :attr:`~pipecat.pipeline.worker.PipelineParams.audio_in_sample_rate`
                    instead. Will be removed in 2.0.0.

            auto_hang_up: Whether to automatically terminate call on EndFrame.
            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        audio_in_sample_rate: int = 16000
        audio_out_sample_rate: int = 16000
        vonage_sample_rate: int | None = None
        sample_rate: int | None = None
        auto_hang_up: bool = True

        @model_validator(mode="before")
        @classmethod
        def _carry_deprecated_rates(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if data.get("vonage_sample_rate") is not None:
                warn_deprecated(
                    "`VonageFrameSerializer.InputParams.vonage_sample_rate` is deprecated since "
                    "1.16.0 and will be removed in 2.0.0. Use `audio_in_sample_rate` and "
                    "`audio_out_sample_rate` instead.",
                    stacklevel=4,
                )
                data.setdefault("audio_in_sample_rate", data["vonage_sample_rate"])
                data.setdefault("audio_out_sample_rate", data["vonage_sample_rate"])
            if data.get("sample_rate") is not None:
                warn_deprecated(
                    "`VonageFrameSerializer.InputParams.sample_rate` is deprecated since 1.16.0 "
                    "and will be removed in 2.0.0. Use `PipelineParams.audio_in_sample_rate` "
                    "instead.",
                    stacklevel=4,
                )
            return data

    def __init__(
        self,
        call_uuid: str,
        application_id: str | None = None,
        private_key: str | None = None,
        params: InputParams | None = None,
    ):
        """Initialize the VonageFrameSerializer.

        Args:
            call_uuid: The Vonage Call UUID.
            application_id: Vonage application ID (required for call control).
            private_key: Private key for JWT generation (required for call control).
            params: Configuration parameters.
        """
        params = params or VonageFrameSerializer.InputParams()
        super().__init__(params)
        self._params: VonageFrameSerializer.InputParams = params

        self._call_uuid = call_uuid
        self._application_id = application_id
        self._private_key = private_key
        self._hangup_attempted = False

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Vonage WebSocket format.

        Handles conversion of various frame types to Vonage WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string (JSON commands) or bytes (audio), or None if the frame isn't handled.
        """
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._hangup_attempted = True
            await self._hang_up_call()
            return None
        elif isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            # Clear the audio buffer to stop playback immediately
            answer = {"action": "clear"}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = await self._resample_audio_out(frame)
            if not data:
                return None

            # Vonage expects raw binary PCM data (not base64 encoded)
            return data
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            # Allow sending custom JSON commands (e.g., notify)
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Vonage WebSocket data to Pipecat frames.

        Handles conversion of Vonage events to appropriate Pipecat frames.
        - Binary messages contain audio data (16-bit linear PCM)
        - Text messages contain JSON events (websocket:connected, websocket:cleared, dtmf, etc.)

        Args:
            data: The raw WebSocket data from Vonage.

        Returns:
            A Pipecat frame corresponding to the Vonage event, or None if unhandled.
        """
        # Check if this is binary audio data
        if isinstance(data, bytes):
            # Binary message = audio data (16-bit linear PCM)
            audio = await self._resample_audio_in(data)
            if not audio:
                return None

            return InputAudioRawFrame(
                audio=audio,
                num_channels=1,  # Vonage uses mono audio
                sample_rate=self._sample_rate,
            )
        else:
            # Text message = JSON event
            try:
                message = json.loads(data)
                event = message.get("event")

                # Handle different event types
                if event == "websocket:connected":
                    logger.debug(
                        f"Vonage WebSocket connected: content-type={message.get('content-type')}"
                    )
                    return None
                elif event == "websocket:cleared":
                    logger.debug("Vonage audio buffer cleared")
                    return None
                elif event == "websocket:notify":
                    logger.debug(f"Vonage notify event: {message.get('payload')}")
                    return None
                elif event == "websocket:dtmf":
                    # Handle DTMF input
                    # Vonage may send digit in different formats, try both
                    digit = message.get("digit") or message.get("dtmf", {}).get("digit")
                    if digit is None:
                        logger.warning(f"DTMF event received but no digit found: {message}")
                        return None

                    digit = str(digit)
                    logger.debug(f"Received DTMF digit: {digit}")
                    try:
                        return InputDTMFFrame(KeypadEntry(digit))
                    except ValueError:
                        logger.warning(f"Invalid DTMF digit received: {digit}")
                        return None
                else:
                    logger.debug(f"Vonage event: {event}")
                    return None

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON message from Vonage: {data}")
                return None

    async def _hang_up_call(self):
        """Hang up the Vonage call using Vonage's REST API."""
        try:
            import time

            import aiohttp
            import jwt

            if not self._call_uuid or not self._application_id or not self._private_key:
                missing = []
                if not self._call_uuid:
                    missing.append("call_uuid")
                if not self._application_id:
                    missing.append("application_id")
                if not self._private_key:
                    missing.append("private_key")

                logger.warning(
                    f"Cannot hang up Vonage call: missing required parameters: {', '.join(missing)}"
                )
                return

            claims = {
                "application_id": self._application_id,
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
                "jti": str(time.time()),
            }
            token = jwt.encode(claims, self._private_key, algorithm="RS256")

            endpoint = f"https://api.nexmo.com/v1/calls/{self._call_uuid}"

            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

            data = {"action": "hangup"}

            async with aiohttp.ClientSession() as session:
                async with session.put(endpoint, headers=headers, json=data) as response:
                    if response.status in (200, 204):
                        logger.info(f"Successfully terminated Vonage call {self._call_uuid}")
                    elif response.status == 404:
                        logger.debug(f"Vonage call {self._call_uuid} was already terminated")
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Vonage call {self._call_uuid}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.exception(f"Failed to hang up Vonage call: {e}")
