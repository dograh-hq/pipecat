"""Flux partials reach turn strategies in provider event order."""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.dograh.flux.stt import DograhFluxSTTService


@pytest.mark.asyncio
@pytest.mark.parametrize("service_type", [DeepgramFluxSTTService, DograhFluxSTTService])
async def test_start_and_updates_emit_partials_before_the_single_final(service_type):
    service = service_type(api_key="test-key")
    frames = []

    async def push(frame, *args, **kwargs):
        frames.append(frame)

    async def broadcast(frame_type, **kwargs):
        frames.append(frame_type(**kwargs))

    service.push_frame = push
    service.broadcast_frame = broadcast
    service._call_event_handler = AsyncMock()
    service._handle_transcription = AsyncMock()
    service.emit_stt_usage_metrics = AsyncMock()

    await service._handle_start_of_turn("Hello")
    await service._handle_update("Hello there")
    await service._handle_end_of_turn("Hello there.", {"words": []})

    assert [type(frame) for frame in frames] == [
        ProposedUserStartedSpeakingFrame,
        InterimTranscriptionFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    assert [frame.text for frame in frames if hasattr(frame, "text")] == [
        "Hello",
        "Hello there",
        "Hello there.",
    ]
    assert frames[-2].finalized
    service._call_event_handler.assert_any_await("on_update", "Hello there")


@pytest.mark.asyncio
async def test_empty_updates_do_not_create_transcription_frames():
    service = DeepgramFluxSTTService(api_key="test-key")
    service.push_frame = AsyncMock()
    service.broadcast_frame = AsyncMock()
    service._call_event_handler = AsyncMock()
    await service._handle_start_of_turn("")
    await service._handle_update("")
    service.push_frame.assert_not_awaited()
