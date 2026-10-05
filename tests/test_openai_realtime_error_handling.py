#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI Realtime fatal server-error handling.

A server-sent ``error`` event is fatal to the connection, but the service used
to only push an ``ErrorFrame`` without actually tearing the websocket down.
That left ``self._websocket`` looking alive, so every later attempt to send a
client event (a queued tool result, a session update, ...) hit the same dead
socket and logged its own "fatal" error — one per queued send.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


def _error_evt(code: str = "internal_error") -> events.ErrorEvent:
    return events.ErrorEvent.model_validate(
        {
            "event_id": "ev_err",
            "type": "error",
            "error": {
                "type": "server_error",
                "code": code,
                "message": "Internal error. (request id: abc123)",
            },
        }
    )


def _service_for_error_handling() -> OpenAIRealtimeLLMService:
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(model="gpt-realtime"),
    )
    service.push_error = AsyncMock()
    service.stop_all_metrics = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_handle_evt_error_disconnects_the_websocket():
    service = _service_for_error_handling()
    fake_websocket = AsyncMock()
    service._websocket = fake_websocket

    await service._handle_evt_error(_error_evt())

    service.push_error.assert_awaited_once()
    fake_websocket.close.assert_awaited_once()
    assert service._websocket is None


@pytest.mark.asyncio
async def test_later_send_after_fatal_error_is_a_silent_noop():
    service = _service_for_error_handling()
    service._websocket = AsyncMock()

    await service._handle_evt_error(_error_evt())
    service.push_error.reset_mock()

    # Simulates a queued outgoing message (e.g. a tool result) reaching
    # _ws_send after the connection already died: it should not raise, and it
    # should not report yet another "fatal" error for the same disconnect.
    await service._ws_send({"type": "response.create"})

    service.push_error.assert_not_called()
