#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ConVox (Deepija Telecom) Media Streams serializer."""

import base64
import json

import pytest

from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    StartFrame,
)
from pipecat.serializers.convox import ConVoxFrameSerializer


STREAM_SID = "stream-abc-123"
CALL_SID = "call-uuid-456"
ACCOUNT_SID = "client-789"
CONVOX_SAMPLE_RATE = 8000
PIPELINE_SAMPLE_RATE = 16000


def _make_serializer(auto_hang_up: bool = True) -> ConVoxFrameSerializer:
    return ConVoxFrameSerializer(
        stream_sid=STREAM_SID,
        call_sid=CALL_SID,
        account_sid=ACCOUNT_SID,
        params=ConVoxFrameSerializer.InputParams(
            convox_sample_rate=CONVOX_SAMPLE_RATE,
            sample_rate=PIPELINE_SAMPLE_RATE,
            auto_hang_up=auto_hang_up,
        ),
    )


async def _setup(serializer: ConVoxFrameSerializer) -> None:
    start = StartFrame(
        audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
        audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
    )
    await serializer.setup(start)


def _pcm_payload(num_samples: int = 1600) -> str:
    # 16-bit little-endian PCM, mono. 1600 samples = 200ms at 8kHz/16-bit.
    # The streaming resampler buffers to produce whole output samples, so
    # tiny chunks (e.g. 20ms = 160 samples) can yield empty output. 200ms
    # is comfortably above that threshold.
    pcm = bytes(num_samples * 2)
    return base64.b64encode(pcm).decode("utf-8")


@pytest.mark.asyncio
async def test_deserialize_connected_returns_none():
    serializer = _make_serializer()
    await _setup(serializer)

    msg = json.dumps({"event": "connected"})
    assert await serializer.deserialize(msg) is None


@pytest.mark.asyncio
async def test_deserialize_start_returns_none():
    serializer = _make_serializer()
    await _setup(serializer)

    msg = json.dumps(
        {
            "event": "start",
            "sequence_number": 1,
            "stream_sid": STREAM_SID,
            "start": {
                "stream_sid": STREAM_SID,
                "call_sid": CALL_SID,
                "account_sid": ACCOUNT_SID,
                "from": "+15551234",
                "to": "+15555678",
                "custom_parameters": {},
                "media_format": {"encoding": "audio/x-l16", "sample_rate": 8000},
            },
        }
    )
    assert await serializer.deserialize(msg) is None


@pytest.mark.asyncio
async def test_deserialize_media_returns_input_audio_raw_frame():
    serializer = _make_serializer()
    await _setup(serializer)

    msg = json.dumps(
        {
            "event": "media",
            "sequence_number": 5,
            "stream_sid": STREAM_SID,
            "media": {
                "chunk": 5,
                "timestamp": "1700000000000",
                "payload": _pcm_payload(),
            },
        }
    )
    frame = await serializer.deserialize(msg)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.sample_rate == PIPELINE_SAMPLE_RATE
    assert frame.num_channels == 1
    assert isinstance(frame.audio, (bytes, bytearray))
    assert len(frame.audio) > 0


@pytest.mark.asyncio
async def test_deserialize_dtmf_returns_input_dtmf_frame():
    serializer = _make_serializer()
    await _setup(serializer)

    msg = json.dumps(
        {
            "event": "dtmf",
            "sequence_number": 3,
            "stream_sid": STREAM_SID,
            "dtmf": {"digit": "1", "duration": "100"},
        }
    )
    frame = await serializer.deserialize(msg)
    assert isinstance(frame, InputDTMFFrame)
    assert frame.button.value == "1"


@pytest.mark.asyncio
async def test_deserialize_dtmf_invalid_digit_returns_none():
    serializer = _make_serializer()
    await _setup(serializer)

    msg = json.dumps(
        {
            "event": "dtmf",
            "sequence_number": 3,
            "stream_sid": STREAM_SID,
            "dtmf": {"digit": "Z", "duration": "100"},
        }
    )
    # Should not raise, just return None.
    assert await serializer.deserialize(msg) is None


@pytest.mark.asyncio
async def test_deserialize_mark_returns_none():
    serializer = _make_serializer()
    await _setup(serializer)

    msg = json.dumps(
        {
            "event": "mark",
            "sequence_number": 7,
            "stream_sid": STREAM_SID,
            "mark": {"name": "audio_played_1"},
        }
    )
    assert await serializer.deserialize(msg) is None


@pytest.mark.asyncio
async def test_deserialize_stop_returns_none():
    serializer = _make_serializer()
    await _setup(serializer)

    msg = json.dumps(
        {
            "event": "stop",
            "sequence_number": 9,
            "stream_sid": STREAM_SID,
            "stop": {
                "call_sid": CALL_SID,
                "account_sid": ACCOUNT_SID,
                "reason": "stopped",
            },
            "timestamp": "2026-04-30T00:00:00Z",
        }
    )
    assert await serializer.deserialize(msg) is None


@pytest.mark.asyncio
async def test_serialize_audio_raw_frame_first_message():
    serializer = _make_serializer()
    await _setup(serializer)

    # 16 kHz pipeline audio, will be resampled down to 8 kHz for ConVox.
    # 6400 bytes = 3200 samples = 200ms at 16kHz/16-bit (large enough to
    # survive the streaming resampler's buffering).
    pcm = bytes(6400)
    audio = AudioRawFrame(audio=pcm, sample_rate=PIPELINE_SAMPLE_RATE, num_channels=1)

    out = await serializer.serialize(audio)
    assert isinstance(out, str)

    parsed = json.loads(out)
    assert parsed["event"] == "media"
    assert parsed["stream_sid"] == STREAM_SID
    assert parsed["sequence_number"] == 1
    assert parsed["media"]["chunk"] == 1
    assert "timestamp" in parsed["media"]
    decoded = base64.b64decode(parsed["media"]["payload"])
    assert len(decoded) > 0


@pytest.mark.asyncio
async def test_serialize_audio_raw_frame_increments_counters():
    serializer = _make_serializer()
    await _setup(serializer)

    pcm = bytes(6400)
    audio = AudioRawFrame(audio=pcm, sample_rate=PIPELINE_SAMPLE_RATE, num_channels=1)

    first = json.loads(await serializer.serialize(audio))
    second = json.loads(await serializer.serialize(audio))

    assert first["sequence_number"] == 1
    assert first["media"]["chunk"] == 1
    assert second["sequence_number"] == 2
    assert second["media"]["chunk"] == 2


@pytest.mark.asyncio
async def test_serialize_interruption_frame_emits_clear():
    serializer = _make_serializer()
    await _setup(serializer)

    out = await serializer.serialize(InterruptionFrame())
    assert isinstance(out, str)

    parsed = json.loads(out)
    assert parsed == {"event": "clear", "stream_sid": STREAM_SID}


@pytest.mark.asyncio
async def test_serialize_end_frame_emits_end_of_interaction():
    serializer = _make_serializer(auto_hang_up=True)
    await _setup(serializer)

    out = await serializer.serialize(EndFrame())
    assert isinstance(out, str)

    parsed = json.loads(out)
    # camelCase "streamSid" is intentional and vendor-confirmed for this event.
    assert parsed["event"] == "endOfInteraction"
    assert parsed["streamSid"] == STREAM_SID
    assert parsed["reason"] == "hangup"
    assert parsed["context"] == {}


@pytest.mark.asyncio
async def test_second_end_frame_returns_none():
    serializer = _make_serializer(auto_hang_up=True)
    await _setup(serializer)

    first = await serializer.serialize(EndFrame())
    assert first is not None

    # Hangup guard: second EndFrame must not emit again.
    second = await serializer.serialize(EndFrame())
    assert second is None


@pytest.mark.asyncio
async def test_serialize_output_transport_message_round_trips_camelcase():
    serializer = _make_serializer()
    await _setup(serializer)

    payload = {
        "event": "endOfInteraction",
        "streamSid": STREAM_SID,
        "reason": "transfer",
        "context": {"target_number": "+919876543210"},
    }
    frame = OutputTransportMessageFrame(message=payload)

    out = await serializer.serialize(frame)
    assert isinstance(out, str)

    parsed = json.loads(out)
    assert parsed == payload
    assert parsed["streamSid"] == STREAM_SID  # camelCase preserved
