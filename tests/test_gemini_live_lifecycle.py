from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.frames.frames import EndFrame
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


class _TestGeminiLiveLLMService(GeminiLiveLLMService):
    def create_client(self):
        self._client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=None)))


class _FakeSession:
    def __init__(self):
        self.close = AsyncMock()


def _make_service() -> _TestGeminiLiveLLMService:
    service = _TestGeminiLiveLLMService(api_key="test-key")
    service.stop_all_metrics = AsyncMock()
    service.cancel_task = AsyncMock()
    service.push_error = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_reconnect_without_deferred_end_frame_opens_new_session():
    service = _make_service()
    service._session = _FakeSession()
    service._session_resumption_handle = "resume-handle"
    service._connect = AsyncMock()

    await service._reconnect()

    service._connect.assert_awaited_once_with(session_resumption_handle="resume-handle")


@pytest.mark.asyncio
async def test_reconnect_releases_deferred_end_frame_instead_of_opening_new_session():
    service = _make_service()
    service._session = _FakeSession()
    service._connect = AsyncMock()
    service.queue_frame = AsyncMock()
    service._bot_is_responding = True

    end_frame = EndFrame(reason="user_idle_max_duration_exceeded")
    timeout_task = MagicMock()
    timeout_task.done.return_value = False
    service._end_frame_pending_bot_turn_finished = end_frame
    service._end_frame_deferral_timeout_task = timeout_task

    await service._reconnect()

    service.queue_frame.assert_awaited_once_with(end_frame)
    service._connect.assert_not_awaited()
    timeout_task.cancel.assert_called_once_with()
    assert service._end_frame_pending_bot_turn_finished is None
    assert service._end_frame_deferral_timeout_task is None
    assert service._bot_is_responding is False


@pytest.mark.asyncio
async def test_terminal_disconnect_still_discards_deferred_end_frame():
    service = _make_service()
    service._session = _FakeSession()
    service.queue_frame = AsyncMock()

    end_frame = EndFrame(reason="user_idle_max_duration_exceeded")
    timeout_task = MagicMock()
    timeout_task.done.return_value = False
    service._end_frame_pending_bot_turn_finished = end_frame
    service._end_frame_deferral_timeout_task = timeout_task

    await service._disconnect()

    service.queue_frame.assert_not_awaited()
    timeout_task.cancel.assert_called_once_with()
    assert service._end_frame_pending_bot_turn_finished is None
    assert service._end_frame_deferral_timeout_task is None
