"""Response-scoped filters cover upstream aggregation and independent speech."""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.tts_service import TTSService, TTSSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.text.base_text_aggregator import AggregationType
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.spoken_text_filter import SpokenTextFilter


class OfflineTTS(TTSService):
    def __init__(self, mode="sentence", text_filter=None):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=mode == "sentence",
            pause_frame_processing=False,
            sample_rate=16000,
            settings=TTSSettings(model=None, voice=None, language=None),
            text_filters=[text_filter or SpokenTextFilter()],
        )
        self.mode = mode
        self.requests = []

    async def run_tts(self, text, context_id):
        self.requests.append(text)
        if self.mode == "words":
            await self.add_word_timestamps(
                [(word, index * 0.01) for index, word in enumerate(text.split())],
                context_id=context_id,
            )
        yield TTSAudioRawFrame(b"\x00\x00" * 320, 16000, 1, context_id=context_id)


class Capture(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


def normalized(text):
    return " ".join(text.split())


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sentence", "words", "missing_words"])
@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (['Ready. {"debug":"Private.', ' Internal."} Welcome.'], "Ready. Welcome."),
        (["Calling {{Example", " Studio}} now."], "Calling Example Studio now."),
        (["Start. ", "Done"], "Start. Done"),
    ],
)
async def test_aggregated_response_chunks_share_filter_state(mode, chunks, expected):
    tts = OfflineTTS(mode)
    context = LLMContext()
    capture = Capture()
    await run_test(
        Pipeline([tts, capture, LLMAssistantAggregator(context)]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            *[
                AggregatedTextFrame(text, AggregationType.SENTENCE, raw_text=text)
                for text in chunks
            ],
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.05),
        ],
        expected_down_frames=None,
    )
    assert normalized(" ".join(tts.requests)) == expected
    assert normalized(" ".join(m["content"] for m in context.messages)) == expected
    for frame in capture.frames:
        if isinstance(frame, (AggregatedTextFrame, TTSTextFrame)):
            for text in (frame.text, frame.raw_text or ""):
                assert not any(marker in text for marker in ("Private", "Internal", "{", "}"))


@pytest.mark.asyncio
async def test_independent_speech_does_not_finish_a_pending_aggregated_response():
    tts = OfflineTTS()
    await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            AggregatedTextFrame('Ready. {"debug":"Private.', AggregationType.SENTENCE),
            TTSSpeakFrame("One moment. {{unused_placeholder}}"),
            AggregatedTextFrame(' Internal."} Welcome.', AggregationType.SENTENCE),
            LLMFullResponseEndFrame(),
        ],
        expected_down_frames=None,
    )
    assert normalized(" ".join(tts.requests)) == "Ready. One moment. Welcome."


class InterruptibleFilter(BaseTextFilter):
    filter_before_aggregation = True

    def __init__(self):
        self.interrupted = False

    async def handle_interruption(self):
        self.interrupted = True

    async def reset_interruption(self):
        self.interrupted = False

    async def filter(self, text):
        return "" if self.interrupted else text


@pytest.mark.asyncio
async def test_new_response_and_independent_speech_reset_custom_filter_interruption():
    tts = OfflineTTS(text_filter=InterruptibleFilter())
    await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("First reply."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            LLMFullResponseStartFrame(),
            AggregatedTextFrame("Second reply.", AggregationType.SENTENCE),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            TTSSpeakFrame("Announcement."),
            SleepFrame(sleep=0.05),
        ],
        expected_down_frames=None,
    )
    assert normalized(" ".join(tts.requests)) == "First reply. Second reply. Announcement."


class FailingFilter(BaseTextFilter):
    def __init__(self, *, before_aggregation):
        self.filter_before_aggregation = before_aggregation

    async def filter(self, text):
        raise ValueError("filter failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stream", "aggregated", "independent"])
async def test_filter_failures_remain_application_errors_without_synthesis(path):
    tts = OfflineTTS(text_filter=FailingFilter(before_aggregation=path != "aggregated"))
    tts.push_error = AsyncMock()
    if path == "stream":
        await tts._handle_text(LLMTextFrame("Private text."))
    else:
        await tts._handle_aggregated_text(
            AggregatedTextFrame("Private text.", AggregationType.SENTENCE),
            whole=path == "independent",
        )

    assert tts.requests == []
    tts.push_error.assert_awaited_once()
    assert tts.push_error.await_args.kwargs["category"] == ErrorCategory.APPLICATION
    assert isinstance(tts.push_error.await_args.kwargs["exception"], ValueError)


@pytest.mark.asyncio
async def test_failed_filter_flush_reports_error_and_still_finishes_response():
    class FailedFlushFilter(BaseTextFilter):
        filter_before_aggregation = True

        async def filter(self, text):
            return ""

        async def flush(self):
            raise ValueError("flush failed")

    tts = OfflineTTS(text_filter=FailedFlushFilter())
    errors = []

    @tts.event_handler("on_error")
    async def on_error(_service, frame):
        errors.append(frame)

    down, _ = await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Private text."),
            LLMFullResponseEndFrame(),
        ],
        expected_down_frames=None,
    )

    assert tts.requests == []
    assert errors
    assert all(error.category == ErrorCategory.APPLICATION for error in errors)
    assert any(isinstance(frame, LLMFullResponseEndFrame) for frame in down)
