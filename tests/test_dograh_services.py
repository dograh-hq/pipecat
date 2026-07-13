#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Dograh-managed AI services."""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import CancelFrame, EndFrame, TTSStoppedFrame
from pipecat.services.dograh.stt import DograhSTTService
from pipecat.services.dograh.tts import DograhTTSService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


def test_stt_metadata_recommends_external_turn_strategies_with_vad_events():
    service = DograhSTTService(api_key="test-key", vad_events=True)

    frame = service.service_metadata_frame()

    assert isinstance(frame.user_turn_strategies, ExternalUserTurnStrategies)


def test_stt_metadata_leaves_turn_strategies_unset_without_vad_events():
    service = DograhSTTService(api_key="test-key", vad_events=False)

    frame = service.service_metadata_frame()

    assert frame.user_turn_strategies is None


@pytest.mark.asyncio
async def test_stt_cleanup_disconnects_without_a_shutdown_frame():
    service = DograhSTTService(api_key="test-key")
    service._disconnect = AsyncMock()
    service._session_start_time = 1.0

    await service.cleanup()

    service._disconnect.assert_awaited_once_with()
    assert service._session_start_time is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "frame"),
    [("stop", EndFrame()), ("cancel", CancelFrame())],
)
async def test_tts_shutdown_disconnects_once(method_name, frame):
    service = DograhTTSService(api_key="test-key")
    service._disconnect = AsyncMock()

    await getattr(service, method_name)(frame)

    service._disconnect.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_tts_stopped_frame_does_not_add_a_reset_word_timestamp():
    service = DograhTTSService(api_key="test-key")
    service._check_started = lambda frame: True
    service.add_word_timestamps = AsyncMock()

    await service.push_frame(TTSStoppedFrame())

    service.add_word_timestamps.assert_not_awaited()
