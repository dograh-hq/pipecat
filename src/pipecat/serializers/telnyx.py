#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Telnyx WebSocket frame serializer for Pipecat."""

import base64
import json
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from pydantic import model_validator

from pipecat.audio import g711
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
from pipecat.utils.enums import EndTaskReason

if TYPE_CHECKING:
    from pipecat.serializers.call_strategies import HangupStrategy, TransferStrategy


class TelnyxFrameSerializer(FrameSerializer):
    """Serializer for Telnyx WebSocket protocol.

    This serializer handles converting between Pipecat frames and Telnyx's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Telnyx call when an EndFrame or CancelFrame is processed, but requires Telnyx
    credentials to be provided.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for TelnyxFrameSerializer.

        Parameters:
            audio_in_sample_rate: Sample rate of the audio Telnyx sends, defaults to 8000 Hz.
            audio_out_sample_rate: Sample rate of the audio sent to Telnyx, defaults to 8000 Hz.
            telnyx_sample_rate: Sample rate used by Telnyx.

                .. deprecated:: 1.16.0
                    Use ``audio_in_sample_rate`` and ``audio_out_sample_rate`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Has no effect: incoming audio is converted to the pipeline's input rate.

                .. deprecated:: 1.16.0
                    Use :attr:`~pipecat.pipeline.worker.PipelineParams.audio_in_sample_rate`
                    instead. Will be removed in 2.0.0.

            inbound_encoding: Audio encoding for data sent to Telnyx (e.g., "PCMU").
            outbound_encoding: Audio encoding for data received from Telnyx (e.g., "PCMU").
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        audio_in_sample_rate: int = 8000
        audio_out_sample_rate: int = 8000
        telnyx_sample_rate: int | None = None
        sample_rate: int | None = None
        inbound_encoding: str = "PCMU"
        outbound_encoding: str = "PCMU"
        auto_hang_up: bool = True

        @model_validator(mode="before")
        @classmethod
        def _carry_deprecated_rates(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if data.get("telnyx_sample_rate") is not None:
                warn_deprecated(
                    "`TelnyxFrameSerializer.InputParams.telnyx_sample_rate` is deprecated since "
                    "1.16.0 and will be removed in 2.0.0. Use `audio_in_sample_rate` and "
                    "`audio_out_sample_rate` instead.",
                    stacklevel=4,
                )
                data.setdefault("audio_in_sample_rate", data["telnyx_sample_rate"])
                data.setdefault("audio_out_sample_rate", data["telnyx_sample_rate"])
            if data.get("sample_rate") is not None:
                warn_deprecated(
                    "`TelnyxFrameSerializer.InputParams.sample_rate` is deprecated since 1.16.0 "
                    "and will be removed in 2.0.0. Use `PipelineParams.audio_in_sample_rate` "
                    "instead.",
                    stacklevel=4,
                )
            return data

    def __init__(
        self,
        stream_id: str,
        outbound_encoding: str,
        inbound_encoding: str,
        call_control_id: str | None = None,
        api_key: str | None = None,
        transfer_strategy: "TransferStrategy | None" = None,
        hangup_strategy: "HangupStrategy | None" = None,
        params: InputParams | None = None,
    ):
        """Initialize the TelnyxFrameSerializer.

        Args:
            stream_id: The Stream ID for Telnyx.
            outbound_encoding: The encoding type for outbound audio (e.g., "PCMU").
            inbound_encoding: The encoding type for inbound audio (e.g., "PCMU").
            call_control_id: The Call Control ID for the Telnyx call (optional, but required for auto hang-up).
            api_key: Your Telnyx API key (required for auto hang-up).
            transfer_strategy: Strategy for handling call transfers. Invoked on
                EndFrame/CancelFrame whose reason is EndTaskReason.TRANSFER_CALL.
            hangup_strategy: Strategy for handling call hangups. Required when
                auto_hang_up is True to actually terminate the call.
            params: Configuration parameters.
        """
        params = params or TelnyxFrameSerializer.InputParams()
        super().__init__(params)
        self._params: TelnyxFrameSerializer.InputParams = params

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            missing_credentials = []
            if not call_control_id:
                missing_credentials.append("call_control_id")
            if not api_key:
                missing_credentials.append("api_key")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )

        self._stream_id = stream_id
        self._call_control_id = call_control_id
        self._api_key = api_key
        self._transfer_strategy = transfer_strategy
        self._hangup_strategy = hangup_strategy
        self._params.outbound_encoding = outbound_encoding
        self._params.inbound_encoding = inbound_encoding

        self._hangup_attempted = False
        self._transfer_attempted = False

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Telnyx WebSocket format.

        Handles conversion of various frame types to Telnyx WebSocket messages.
        For EndFrames and CancelFrames, initiates call termination if auto_hang_up is enabled.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.

        Raises:
            ValueError: If an unsupported encoding is specified.
        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            frame_reason = getattr(frame, "reason", None)
            if frame_reason == EndTaskReason.TRANSFER_CALL.value:
                if self._transfer_attempted:
                    return None
                self._transfer_attempted = True
                if self._transfer_strategy:
                    context = {
                        "call_control_id": self._call_control_id,
                        "api_key": self._api_key,
                    }
                    success = await self._transfer_strategy.execute_transfer(context)
                    if not success:
                        logger.error(f"Transfer strategy failed for call {self._call_control_id}")
                else:
                    logger.warning(
                        f"No transfer strategy configured for call {self._call_control_id}"
                    )
                return None
            elif self._params.auto_hang_up and not self._hangup_attempted:
                self._hangup_attempted = True
                if self._hangup_strategy:
                    context = {
                        "call_control_id": self._call_control_id,
                        "api_key": self._api_key,
                    }
                    success = await self._hangup_strategy.execute_hangup(context)
                    if not success:
                        logger.error(f"Hangup strategy failed for call {self._call_control_id}")
                else:
                    logger.warning(
                        f"No hangup strategy configured for call {self._call_control_id}"
                    )
                return None
        elif isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            answer = {"event": "clear"}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = await self._resample_audio_out(frame)
            if not data:
                return None

            if self._params.inbound_encoding == "PCMU":
                serialized_data = g711.pcm_to_ulaw(data)
            elif self._params.inbound_encoding == "PCMA":
                serialized_data = g711.pcm_to_alaw(data)
            else:
                raise ValueError(f"Unsupported encoding: {self._params.inbound_encoding}")

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Telnyx WebSocket data to Pipecat frames.

        Handles conversion of Telnyx media events to appropriate Pipecat frames,
        including audio data and DTMF keypresses.

        Args:
            data: The raw WebSocket data from Telnyx.

        Returns:
            A Pipecat frame corresponding to the Telnyx event, or None if unhandled.

        Raises:
            ValueError: If an unsupported encoding is specified.
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON message: {data}")
            return None

        if message.get("event") == "media":
            payload_base64 = message.get("media", {}).get("payload")
            if not payload_base64:
                return None
            payload = base64.b64decode(payload_base64)

            if self._params.outbound_encoding == "PCMU":
                deserialized_data = g711.ulaw_to_pcm(payload)
            elif self._params.outbound_encoding == "PCMA":
                deserialized_data = g711.alaw_to_pcm(payload)
            else:
                raise ValueError(f"Unsupported encoding: {self._params.outbound_encoding}")

            audio = await self._resample_audio_in(deserialized_data)
            if not audio:
                return None

            return InputAudioRawFrame(audio=audio, num_channels=1, sample_rate=self._sample_rate)
        elif message.get("event") == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                return None
        else:
            return None
