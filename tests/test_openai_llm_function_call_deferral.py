#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for node-transition function-call deferral in OpenAI LLM services."""

from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.llm import OpenAILLMService


def _make_service() -> OpenAILLMService:
    with patch.object(OpenAILLMService, "create_client"):
        return OpenAILLMService(api_key="test-key")


def _make_function_call(name: str, tool_call_id: str) -> FunctionCallFromLLM:
    return FunctionCallFromLLM(
        context=LLMContext(),
        tool_call_id=tool_call_id,
        function_name=name,
        arguments={},
    )


@pytest.mark.asyncio
async def test_non_transition_call_is_not_deferred_after_generated_text():
    service = _make_service()
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call("look_up_account", "call-ordinary")

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=True,
    )

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_node_transition_call_is_deferred_after_generated_text():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=True,
    )

    service.run_function_calls.assert_not_awaited()
    assert service._pending_node_transition_function_calls == [function_call]


@pytest.mark.asyncio
async def test_node_transition_call_runs_immediately_without_generated_text():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=False,
    )

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_mixed_batch_with_node_transition_is_deferred_together():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_calls = [
        _make_function_call("look_up_account", "call-ordinary"),
        _make_function_call("transition_to_next_node", "call-transition"),
    ]

    await service._run_or_defer_function_calls(
        function_calls,
        text_generated=True,
    )

    service.run_function_calls.assert_not_awaited()
    assert service._pending_node_transition_function_calls == function_calls


@pytest.mark.asyncio
async def test_pending_node_transition_batch_runs_after_tts():
    service = _make_service()
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )
    service._pending_node_transition_function_calls = [function_call]

    await service._run_pending_node_transition_function_calls()

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []
