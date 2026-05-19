#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ConVox (Deepija Telecom) Media Streams WebSocket protocol serializer for Pipecat."""

import base64
import json
import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler
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
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class ConVoxFrameSerializer(FrameSerializer):
    """Serializer for ConVox (Deepija Telecom) Media Streams WebSocket protocol.

    ConVox uses a JSON-event WebSocket protocol with base64-encoded Linear PCM
    audio (audio/x-l16, 16-bit little-endian, mono). Unlike most telephony
    serializers, ConVox sends raw 16-bit PCM, not μ-law.

    Graceful call termination is performed in-band via an ``endOfInteraction``
    event sent over the same WebSocket. ConVox has no REST API, so no external
    credentials are required.
    """

    class InputParams(BaseModel):
        """Configuration parameters for ConVoxFrameSerializer.

        Parameters:
            convox_sample_rate: Sample rate used by ConVox, defaults to 8000 Hz.
                Set dynamically from the inbound ``start`` event by calling code.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically send ``endOfInteraction`` on
                EndFrame or CancelFrame.
        """

        convox_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_sid: str,
        call_sid: Optional[str] = None,
        account_sid: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the ConVoxFrameSerializer.

        Args:
            stream_sid: The ConVox stream identifier from the ``start`` event.
            call_sid: The associated ConVox call identifier (optional).
            account_sid: The ConVox account/client identifier (optional).
            params: Configuration parameters.
        """
        # No external credentials are required: endOfInteraction is in-band,
        # not a REST call, so auto_hang_up does not need auth tokens.
        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._account_sid = account_sid
        self._params = params or ConVoxFrameSerializer.InputParams()

        self._convox_sample_rate = self._params.convox_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        # sequence_number increments on every outbound event; chunk increments
        # only on outbound media events.
        self._sequence_number = 0
        self._chunk = 0
        self._hangup_attempted = False

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

        # Debug counters — used to diagnose "client hears nothing but recording is fine".
        self._out_audio_serialized = 0
        self._out_audio_dropped = 0
        self._in_audio_received = 0
        self._in_audio_dropped = 0
        # Per-frame detailed logging window: log every frame for the first
        # N frames in each direction so timestamps/sequences can be
        # correlated against ConVox-side logs.
        self._detailed_log_first_n = 50
        # 5-second heartbeat: when active in either direction, emit a
        # rolling-totals line so silent gaps in the WS are visible.
        self._heartbeat_interval_s = 5.0
        self._last_out_heartbeat_ts = 0.0
        self._last_in_heartbeat_ts = 0.0
        # Idempotent flag — log session totals exactly once when the
        # pipeline terminates (any End/Cancel path), independent of
        # auto_hang_up. Without this, hangup-by-caller-first paths skip
        # the totals entirely.
        self._final_totals_logged = False

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | list[str | bytes] | None:
        """Serializes a Pipecat frame to ConVox WebSocket format.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            JSON-encoded string for ConVox events, a list of JSON strings when a
            single frame produces multiple events (the hangup path emits
            ``endOfInteraction`` followed by ``stop``), or None if the frame
            isn't handled.
        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            # Always emit session totals exactly once on any termination
            # path (graceful EndFrame, CancelFrame from user-hangup, etc.),
            # independent of auto_hang_up. This is the share-with-vendor
            # summary line.
            if not self._final_totals_logged:
                self._final_totals_logged = True
                logger.info(
                    f"ConVox serializer: session totals on {type(frame).__name__} — "
                    f"stream_sid={self._stream_sid}, "
                    f"outbound_audio_serialized={self._out_audio_serialized} "
                    f"(dropped={self._out_audio_dropped}), "
                    f"inbound_audio_received={self._in_audio_received} "
                    f"(dropped={self._in_audio_dropped}), "
                    f"final_seq={self._sequence_number}, final_chunk={self._chunk}, "
                    f"hangup_attempted={self._hangup_attempted}"
                )

            if self._params.auto_hang_up and not self._hangup_attempted:
                self._hangup_attempted = True
                self._sequence_number += 1
                # NOTE: endOfInteraction uses camelCase "streamSid" (vendor-confirmed,
                # intentional). All other ConVox events use snake_case "stream_sid".
                answer = {
                    "event": "endOfInteraction",
                    "streamSid": self._stream_sid,
                    "reason": "hangup",
                    "context": {},
                }
                # ConVox additionally expects an explicit "stop" event after
                # endOfInteraction so the carrier tears the stream down rather
                # than waiting for the WS to drop. Uses snake_case "stream_sid"
                # like every event except endOfInteraction.
                self._sequence_number += 1
                stop_event = {
                    "event": "stop",
                    "sequence_number": self._sequence_number,
                    "stream_sid": self._stream_sid,
                    "stop": {
                        "call_sid": self._call_sid,
                        "account_sid": self._account_sid,
                        "reason": "stopped",
                    },
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    ),
                }
                logger.info(
                    f"Sending ConVox endOfInteraction + stop for stream "
                    f"{self._stream_sid}, final_seq={self._sequence_number}"
                )
                return [json.dumps(answer), json.dumps(stop_event)]
            return None
        elif isinstance(frame, InterruptionFrame):
            # ConVox uses the simple "clear" format (spec page 8): only event
            # and stream_sid, no other fields.
            self._sequence_number += 1
            answer = {"event": "clear", "stream_sid": self._stream_sid}
            logger.info(
                f"ConVox serializer: sending 'clear' (interruption) for stream "
                f"{self._stream_sid}, seq={self._sequence_number}, "
                f"audio_frames_serialized_so_far={self._out_audio_serialized}"
            )
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: resample PCM at the frame's rate to ConVox's sample rate.
            # ConVox expects raw 16-bit little-endian PCM, base64-encoded.
            serialized_data = await self._output_resampler.resample(
                data, frame.sample_rate, self._convox_sample_rate
            )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                self._out_audio_dropped += 1
                if self._out_audio_dropped <= 3 or self._out_audio_dropped % 100 == 0:
                    logger.warning(
                        f"ConVox serializer: dropped outbound audio frame "
                        f"#{self._out_audio_dropped} (resampler returned empty) — "
                        f"in_rate={frame.sample_rate}, out_rate={self._convox_sample_rate}, "
                        f"raw_bytes={len(data)}, stream_sid={self._stream_sid}"
                    )
                return None

            # Spec note: payload is "typically 320 bytes" (160 samples * 2
            # bytes = 20ms at 8kHz/16-bit). Don't enforce — emit whatever the
            # resampler returns.
            payload = base64.b64encode(serialized_data).decode("utf-8")

            self._sequence_number += 1
            self._chunk += 1
            self._out_audio_serialized += 1
            answer = {
                "event": "media",
                "sequence_number": self._sequence_number,
                "stream_sid": self._stream_sid,
                "media": {
                    "chunk": self._chunk,
                    "timestamp": str(int(time.time() * 1000)),
                    "payload": payload,
                },
            }
            serialized = json.dumps(answer)

            # Per-frame detailed log for the first N outbound frames so
            # timestamps and sequence numbers can be matched against
            # vendor-side logs. After N, fall back to periodic + heartbeat.
            if self._out_audio_serialized <= self._detailed_log_first_n:
                logger.info(
                    f"ConVox serializer: outbound frame #{self._out_audio_serialized} — "
                    f"stream_sid={self._stream_sid}, "
                    f"in_rate={frame.sample_rate}Hz, out_rate={self._convox_sample_rate}Hz, "
                    f"raw_bytes={len(data)}, resampled_bytes={len(serialized_data)}, "
                    f"payload_b64_chars={len(payload)}, json_bytes={len(serialized)}, "
                    f"seq={self._sequence_number}, chunk={self._chunk}, "
                    f"timestamp_ms={int(time.time() * 1000)}"
                )
            elif self._out_audio_serialized % 100 == 0:
                logger.debug(
                    f"ConVox serializer: serialized {self._out_audio_serialized} outbound "
                    f"audio frames (dropped={self._out_audio_dropped}) — seq={self._sequence_number}, "
                    f"chunk={self._chunk}, last_payload_b64_chars={len(payload)}, "
                    f"stream_sid={self._stream_sid}"
                )

            # 5-second outbound heartbeat — emits at most every interval.
            now = time.monotonic()
            if now - self._last_out_heartbeat_ts >= self._heartbeat_interval_s:
                self._last_out_heartbeat_ts = now
                logger.info(
                    f"ConVox serializer: outbound heartbeat — "
                    f"stream_sid={self._stream_sid}, "
                    f"out_serialized={self._out_audio_serialized} "
                    f"(dropped={self._out_audio_dropped}), "
                    f"in_received={self._in_audio_received} "
                    f"(dropped={self._in_audio_dropped}), "
                    f"seq={self._sequence_number}, chunk={self._chunk}"
                )
            return serialized
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            # Allows application code to emit endOfInteraction with custom
            # reason/context (e.g. transfer with a target_number).
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes ConVox WebSocket data to Pipecat frames.

        Args:
            data: The raw WebSocket data from ConVox (JSON text).

        Returns:
            A Pipecat frame for the ConVox event, or None if unhandled.
        """
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse JSON message from ConVox: {data!r}")
            return None

        event = message.get("event")

        if event == "media":
            payload_base64 = message.get("media", {}).get("payload", "")
            payload = base64.b64decode(payload_base64)

            # Input: ConVox sends raw 16-bit little-endian PCM. Resample from
            # ConVox's sample rate to the pipeline input rate. Do NOT use μ-law
            # decoding — this is linear PCM, like Vonage.
            deserialized_data = await self._input_resampler.resample(
                payload, self._convox_sample_rate, self._sample_rate
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                self._in_audio_dropped += 1
                if self._in_audio_dropped <= 3 or self._in_audio_dropped % 100 == 0:
                    logger.warning(
                        f"ConVox serializer: dropped inbound audio frame "
                        f"#{self._in_audio_dropped} (resampler returned empty) — "
                        f"in_rate={self._convox_sample_rate}, out_rate={self._sample_rate}, "
                        f"raw_b64_chars={len(payload_base64)}, raw_bytes={len(payload)}"
                    )
                return None

            self._in_audio_received += 1
            if self._in_audio_received <= self._detailed_log_first_n:
                logger.info(
                    f"ConVox serializer: inbound frame #{self._in_audio_received} — "
                    f"stream_sid={self._stream_sid}, "
                    f"convox_rate={self._convox_sample_rate}Hz, pipeline_rate={self._sample_rate}Hz, "
                    f"raw_b64_chars={len(payload_base64)}, raw_bytes={len(payload)}, "
                    f"resampled_bytes={len(deserialized_data)}, "
                    f"timestamp_ms={int(time.time() * 1000)}"
                )
            elif self._in_audio_received % 500 == 0:
                logger.debug(
                    f"ConVox serializer: received {self._in_audio_received} inbound audio frames "
                    f"(dropped={self._in_audio_dropped}) | outbound serialized={self._out_audio_serialized} "
                    f"(dropped={self._out_audio_dropped})"
                )

            # 5-second inbound heartbeat. Pairing this with the outbound
            # heartbeat makes silent gaps in either direction obvious.
            now = time.monotonic()
            if now - self._last_in_heartbeat_ts >= self._heartbeat_interval_s:
                self._last_in_heartbeat_ts = now
                logger.info(
                    f"ConVox serializer: inbound heartbeat — "
                    f"stream_sid={self._stream_sid}, "
                    f"in_received={self._in_audio_received} "
                    f"(dropped={self._in_audio_dropped}), "
                    f"out_serialized={self._out_audio_serialized} "
                    f"(dropped={self._out_audio_dropped})"
                )

            return InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=1,
                sample_rate=self._sample_rate,
            )
        elif event == "dtmf":
            digit = message.get("dtmf", {}).get("digit")
            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                logger.warning(f"Invalid DTMF digit received from ConVox: {digit!r}")
                return None
        elif event == "connected":
            logger.debug("ConVox WebSocket connected")
            return None
        elif event == "start":
            # Normally consumed by calling code before serializer construction.
            # If it reaches deserialize(), just log and ignore.
            logger.debug(f"ConVox start event reached serializer: {message.get('stream_sid')}")
            return None
        elif event == "mark":
            mark_name = message.get("mark", {}).get("name")
            logger.debug(f"ConVox mark received: {mark_name}")
            return None
        elif event == "stop":
            reason = message.get("stop", {}).get("reason")
            logger.info(f"ConVox stop event received (reason={reason}); awaiting WS close")
            return None
        else:
            logger.debug(f"Unhandled ConVox event: {event}")
            return None
