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
from pipecat.processors.filters.spoken_text_processor import SpokenTextProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.tts_service import TTSService, TTSSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.text.base_text_aggregator import TextType
from pipecat.utils.text.transforms.spoken_text import SpokenTextTransform


class OfflineTTS(TTSService):
    def __init__(self, mode="sentence"):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=mode == "sentence",
            pause_frame_processing=False,
            sample_rate=16000,
            settings=TTSSettings(model=None, voice=None, language=None),
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
        Pipeline([SpokenTextProcessor(), tts, capture, LLMAssistantAggregator(context)]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            *[AggregatedTextFrame(text, TextType.SENTENCE, raw_text=text) for text in chunks],
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
        Pipeline([SpokenTextProcessor(), tts]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            AggregatedTextFrame('Ready. {"debug":"Private.', TextType.SENTENCE),
            TTSSpeakFrame("One moment. {{unused_placeholder}}"),
            AggregatedTextFrame(' Internal."} Welcome.', TextType.SENTENCE),
            LLMFullResponseEndFrame(),
        ],
        expected_down_frames=None,
    )
    assert normalized(" ".join(tts.requests)) == "Ready. One moment. Welcome."


@pytest.mark.asyncio
async def test_new_response_and_independent_speech_reset_transform_state():
    tts = OfflineTTS()
    await run_test(
        Pipeline([SpokenTextProcessor(), tts]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("First reply."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            LLMFullResponseStartFrame(),
            AggregatedTextFrame("Second reply.", TextType.SENTENCE),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            TTSSpeakFrame("Announcement."),
            SleepFrame(sleep=0.05),
        ],
        expected_down_frames=None,
    )
    assert normalized(" ".join(tts.requests)) == "First reply. Second reply. Announcement."


class FailingTransform(SpokenTextTransform):
    async def __call__(self, text, text_type):
        raise ValueError("transform failed")

    async def transform_stream(self, text):
        raise ValueError("transform failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stream", "aggregated", "independent"])
async def test_transform_failures_remain_application_errors_without_synthesis(path):
    processor = SpokenTextProcessor(transform=FailingTransform())
    processor.push_error = AsyncMock()
    processor.push_frame = AsyncMock()
    frame = (
        TTSSpeakFrame("Private text.")
        if path == "independent"
        else (
            AggregatedTextFrame("Private text.", TextType.SENTENCE)
            if path == "aggregated"
            else LLMTextFrame("Private text.")
        )
    )
    await processor._transform_frame(frame, streaming=path != "independent")

    processor.push_frame.assert_not_awaited()
    processor.push_error.assert_awaited_once()
    assert processor.push_error.await_args.kwargs["category"] == ErrorCategory.APPLICATION
    assert isinstance(processor.push_error.await_args.kwargs["exception"], ValueError)


@pytest.mark.asyncio
async def test_failed_transform_flush_reports_error_and_still_finishes_response():
    class FailedFlushTransform(SpokenTextTransform):
        async def transform_stream(self, text):
            return ""

        async def flush(self):
            raise ValueError("flush failed")

    processor = SpokenTextProcessor(transform=FailedFlushTransform())
    tts = OfflineTTS()
    errors = []

    @processor.event_handler("on_error")
    async def on_error(_service, frame):
        errors.append(frame)

    down, _ = await run_test(
        Pipeline([processor, tts]),
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


@pytest.mark.asyncio
async def test_sanitizing_independent_speech_keeps_routing_and_uninterruptibility():
    speech = TTSSpeakFrame(
        "Hello {{Example Studio}}.", append_to_context=False, persist_to_logs=True
    )
    speech.interruptible = False
    speech.transport_destination = "caller"
    speech.metadata = {"speech_id": "announcement"}
    down, _ = await run_test(SpokenTextProcessor(), frames_to_send=[speech])

    rewritten = next(frame for frame in down if isinstance(frame, TTSSpeakFrame))
    assert rewritten.text == "Hello Example Studio."
    assert rewritten.id == speech.id
    assert rewritten.interruptible is False
    assert rewritten.transport_destination == "caller"
    assert rewritten.metadata == speech.metadata
    assert rewritten.append_to_context is False
    assert rewritten.persist_to_logs is True
    assert speech.text == "Hello {{Example Studio}}."


@pytest.mark.asyncio
async def test_recorded_transcripts_are_not_rewritten_after_audio_was_synthesized():
    recording = AggregatedTextFrame(
        "Literal {{title}}", TextType.SENTENCE, raw_text="Literal {{title}}"
    )
    recording.text_type = "recording"
    down, _ = await run_test(SpokenTextProcessor(), frames_to_send=[recording])

    assert recording in down
    assert recording.text == recording.raw_text == "Literal {{title}}"


@pytest.mark.asyncio
async def test_buffered_transform_flushes_before_the_response_ends():
    tts = OfflineTTS()
    processor = SpokenTextProcessor(
        transform=SpokenTextTransform(custom_patterns=[r"INTERNAL.*?END"])
    )
    await run_test(
        Pipeline([processor, tts]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello INTERNAL. "),
            LLMTextFrame("Private planning END world."),
            LLMFullResponseEndFrame(),
        ],
    )

    assert normalized(" ".join(tts.requests)) == "Hello world."


@pytest.mark.asyncio
async def test_failed_stream_drops_remaining_chunks_and_recovers_on_next_response():
    class FailOnceTransform(SpokenTextTransform):
        failed = False

        async def transform_stream(self, text):
            if not self.failed:
                self.failed = True
                raise ValueError("cannot sanitize")
            return await super().transform_stream(text)

    processor = SpokenTextProcessor(transform=FailOnceTransform())
    tts = OfflineTTS()
    await run_test(
        Pipeline([processor, tts]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Private prefix "),
            LLMTextFrame("private continuation."),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Welcome."),
            LLMFullResponseEndFrame(),
        ],
    )

    assert normalized(" ".join(tts.requests)) == "Welcome."
