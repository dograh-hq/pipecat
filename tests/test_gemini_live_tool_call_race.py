#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test for the Gemini Live tool-call session-swap race.

When `_update_settings(system_instruction=...)` is invoked between a
`tool_call` arriving and its `tool_response` being delivered, the synchronous
reconnect closes the originating session before the tool-result is sent. The
subsequent `send_tool_response` then lands on a session that never emitted
the matching `tool_call` — under Gemini 3.x's sync-only function-call
contract the model is stranded.

This test asserts the wire-protocol invariant directly:
`send_tool_response(id=X)` reaches the `AsyncSession` instance that emitted
`tool_call(id=X)`. The harness substitutes a `MockGeminiSession` for the real
websocket; everything else (settings dispatch, function-call runner, deferred
tool-call handling) runs unmodified.
"""

import asyncio
import os
from types import SimpleNamespace

import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.settings import LLMSettings
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class MockGeminiSession:
    """Records every send and is uniquely labelled per instance."""

    _next_id = 0

    def __init__(self):
        type(self)._next_id += 1
        self.label = f"session-{type(self)._next_id}"
        self.sent_tool_responses: list = []
        self.sent_realtime_inputs: list = []
        self.sent_client_contents: list = []
        self.closed = False

    async def send_tool_response(self, function_responses):
        self.sent_tool_responses.append(function_responses)

    async def send_realtime_input(self, **kwargs):
        self.sent_realtime_inputs.append(kwargs)

    async def send_client_content(self, **kwargs):
        self.sent_client_contents.append(kwargs)

    async def close(self):
        self.closed = True

    def receive(self):
        async def _empty():
            if False:
                yield None

        return _empty()


class HarnessedGeminiLiveLLMService(GeminiLiveLLMService):
    """Replaces the real websocket connection with a `MockGeminiSession`.

    Skips `_connection_task_handler` / receive loop so the test owns
    message-arrival timing. Every other code path runs unmodified.
    """

    def __init__(self, *, sessions_log: list[MockGeminiSession], **kwargs):
        super().__init__(**kwargs)
        self._sessions_log = sessions_log

    async def _connect(self, session_resumption_handle: str | None = None):
        if self._session:
            return
        session = MockGeminiSession()
        self._sessions_log.append(session)
        await self._handle_session_ready(session)


def _make_tool_call_message(call_id: str, name: str, args: dict | None = None):
    """Duck-type a `LiveServerMessage` carrying a single tool_call."""
    return SimpleNamespace(
        tool_call=SimpleNamespace(
            function_calls=[SimpleNamespace(id=call_id, name=name, args=args or {})]
        )
    )


class TestGeminiLiveToolCallRace:
    @pytest.mark.asyncio
    async def test_send_tool_response_reaches_originating_session(self):
        """tool_response(id=X) must reach the session that emitted tool_call(id=X).

        Drives the production dispatch path: a tool_call arrives while the
        bot is mid-utterance and is deferred; when the bot's utterance ends,
        `_set_bot_is_responding(False)` triggers the function-call runner,
        which invokes a registered handler that calls
        `_update_settings(system_instruction=...)` then `result_callback(...)`.
        The aggregator path that would normally translate the broadcast result
        into `_tool_result(...)` is invoked manually at the end.

        Without the `_pending_tool_responses` gating, `_update_settings`
        reconnects synchronously, swapping `self._session` before
        `_tool_result` runs.
        """
        sessions: list[MockGeminiSession] = []
        service = HarnessedGeminiLiveLLMService(
            sessions_log=sessions,
            api_key=os.getenv("GOOGLE_API_KEY", "stub-value-mock-connect-bypasses-auth"),
            system_instruction="initial node prompt",
        )
        # Force a Gemini 3.x model: the bug exists on 2.5 too but is only a
        # hard hang on 3.x (sync-only function-call contract).
        service._settings.model = "models/gemini-3.1-flash-live-preview"

        service._context = LLMContext()

        # `run_function_calls` schedules handlers via `self.task_manager`,
        # which is populated by `FrameProcessor.setup`. Replicate the minimal
        # init that production performs on StartFrame.
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await service.setup(FrameProcessorSetup(clock=SystemClock(), task_manager=task_manager))

        handler_done = asyncio.Event()
        handler_trace: list[str] = []

        async def transition_func(params: FunctionCallParams):
            handler_trace.append("entered")
            await service._update_settings(LLMSettings(system_instruction="new node prompt"))
            handler_trace.append("update_settings_returned")
            await params.result_callback({"status": "ok"})
            handler_trace.append("result_callback_returned")
            handler_done.set()

        service.register_function("advance_to_agent", transition_func)

        await service._connect()
        assert len(sessions) == 1
        session_a = sessions[0]
        assert service._session is session_a

        # tool_call(X) arrives while bot is mid-utterance → deferred.
        service._bot_is_responding = True
        await service._handle_msg_tool_call(
            _make_tool_call_message(call_id="X-call-id", name="advance_to_agent")
        )

        # Bot's utterance ends; production dispatch path runs the handler.
        await service._set_bot_is_responding(False)
        await asyncio.wait_for(handler_done.wait(), timeout=2.0)

        session_at_result_time = service._session

        # Aggregator stand-in: `FunctionCallResultFrame` would normally
        # propagate through `LLMContextAggregator` to a context frame, which
        # `_handle_context` consumes to call `_tool_result`. Without a
        # `Pipeline` we invoke it directly.
        await service._tool_result(
            tool_call_id="X-call-id",
            tool_name="advance_to_agent",
            tool_result_message={"status": "ok"},
        )

        assert session_a.sent_tool_responses, (
            "send_tool_response(id=X) did not reach the session that emitted "
            "tool_call(id=X).\n"
            f"  Handler progress:        {handler_trace}\n"
            f"  Sessions opened:         {[s.label for s in sessions]}\n"
            f"  Session A closed:        {session_a.closed}\n"
            f"  Per-session sends:       "
            f"{[(s.label, len(s.sent_tool_responses)) for s in sessions]}\n"
            f"  service._session at _tool_result time: "
            f"{getattr(session_at_result_time, 'label', None)}\n"
        )
