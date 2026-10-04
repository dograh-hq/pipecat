"""Replay provider streams and playback/interruption ordering without network calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from google.genai import types

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    StopFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.asyncio.task_manager import TaskManager


@pytest_asyncio.fixture(params=["openai", "google"])
async def service(request):
    if request.param == "google":
        llm = GoogleLLMService(api_key="test-key")
    else:
        with patch.object(OpenAILLMService, "create_client"):
            llm = OpenAILLMService(api_key="test-key")
    await llm.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=SimpleNamespace(app_resources=None, worker_runner=None),
        )
    )
    llm.push_frame = AsyncMock()
    llm.run_function_calls = AsyncMock()
    for name in ("end_call", "transfer_agent"):
        llm.register_function(name, AsyncMock(), is_node_transition=True)
    llm.register_function("save_booking", AsyncMock())
    yield llm
    await llm.cleanup()


async def respond(service, names, text="Your booking is confirmed.", during_stream=None):
    if isinstance(service, GoogleLLMService):

        async def stream(context):
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(role="model", parts=[types.Part(text=text)])
                    )
                ]
            )
            if during_stream:
                await during_stream()
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=name, id=f"call-{i}", args={}
                                    )
                                )
                                for i, name in enumerate(names)
                            ],
                        )
                    )
                ]
            )

        service._stream_response = stream
    else:

        class Stream:
            def __aiter__(self):
                return self.iterate()

            async def iterate(self):
                yield SimpleNamespace(
                    usage=None,
                    model=None,
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
                )
                if during_stream:
                    await during_stream()
                for i, name in enumerate(names):
                    yield SimpleNamespace(
                        usage=None,
                        model=None,
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=i,
                                            id=f"call-{i}",
                                            function=SimpleNamespace(name=name, arguments="{}"),
                                        )
                                    ],
                                )
                            )
                        ],
                    )

            async def close(self):
                pass

        service.get_chat_completions = AsyncMock(return_value=Stream())
    await service._process_context(LLMContext())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names",
    [
        ["save_booking"],
        ["save_booking", "end_call"],
        ["end_call", "save_booking"],
        ["end_call", "transfer_agent"],
    ],
)
async def test_only_a_single_transition_can_be_deferred(service, names):
    await respond(service, names)
    service.run_function_calls.assert_awaited_once()
    assert [c.function_name for c in service.run_function_calls.await_args.args[0]] == names
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert service.run_function_calls.await_count == 1


@pytest.mark.asyncio
async def test_single_transition_waits_for_normal_playback_completion(service):
    await respond(service, ["end_call"])
    service.run_function_calls.assert_not_awaited()
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["interruption", "interrupted_stop", "cancel", "end", "stop"])
async def test_interrupted_or_cancelled_transition_never_executes(service, first):
    await respond(service, ["end_call"])
    service.run_function_calls.assert_not_awaited()
    if first == "interrupted_stop":
        stopped = BotStoppedSpeakingFrame()
        stopped.interrupted = True
        await service.process_frame(stopped, FrameDirection.UPSTREAM)
    else:
        frame = {
            "interruption": InterruptionFrame,
            "cancel": CancelFrame,
            "end": EndFrame,
            "stop": StopFrame,
        }[first]()
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_interruption_during_stream_cannot_arm_a_late_transition(service):
    async def interrupt():
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    await respond(service, ["end_call"], during_stream=interrupt)
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_not_awaited()
    # A fresh response can still decide to end the call.
    await respond(service, ["end_call"])
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "...", "\n"])
async def test_transition_without_speech_does_not_wait(service, text):
    await respond(service, ["end_call"], text=text)
    service.run_function_calls.assert_awaited_once()
