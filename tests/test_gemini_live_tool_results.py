#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


class _TestGeminiLiveLLMService(GeminiLiveLLMService):
    def create_client(self):
        self._client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=None)))


class _FakeSession:
    def __init__(self):
        self.send_tool_response = AsyncMock()
        self.send_realtime_input = AsyncMock()


@pytest.mark.asyncio
async def test_gemini_3_tool_result_sends_only_tool_response():
    service = _TestGeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(model="gemini-3.1-flash-live-preview"),
    )
    session = _FakeSession()
    service._session = session

    delivered = await service._tool_result(
        "call-123",
        "get_account_balances",
        {"status": "success", "balances": []},
    )

    assert delivered is True
    session.send_tool_response.assert_awaited_once()
    response = session.send_tool_response.await_args.kwargs["function_responses"]
    assert response.id == "call-123"
    assert response.name == "get_account_balances"
    assert response.response == {"status": "success", "balances": []}
    session.send_realtime_input.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_final", [False, True])
@pytest.mark.parametrize("spoken", [False, True])
async def test_reconnect_preserves_async_tool_result_delivery_flags(is_final, spoken):
    service = _TestGeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(model="gemini-3.8-live-extended-thinking"),
    )
    service.register_function("get_balance", AsyncMock(), cancel_on_interruption=False)

    delivered = await service._tool_result(
        "call-123", "get_balance", {"balance": 42}, is_final=is_final, spoken=spoken
    )
    assert delivered is False

    session = _FakeSession()
    service._session = session
    await service._drain_pending_tool_results()

    session.send_tool_response.assert_awaited_once()
    response = session.send_tool_response.await_args.kwargs["function_responses"]
    assert response.will_continue is not is_final
    assert response.scheduling.value == ("WHEN_IDLE" if spoken else "SILENT")
    assert response.response["balance"] == 42
    assert ("call-123" in service._completed_tool_calls) is is_final
    assert not service._pending_tool_results
