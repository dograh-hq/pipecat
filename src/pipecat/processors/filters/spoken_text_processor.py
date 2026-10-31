"""Sanitize streamed speech before TTS aggregation, timing and context bookkeeping."""

from dataclasses import fields, replace

from pipecat.frames.frames import (
    AggregatedTextFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.text.base_text_aggregator import TextType
from pipecat.utils.text.transforms.spoken_text import SpokenTextTransform


class SpokenTextProcessor(FrameProcessor):
    """Remove model control syntax from speech and the context that records it.

    Place immediately before TTS. Raw text keeps its frame type so the TTS
    service still owns sentence/token aggregation and pronunciation transforms.
    The transform buffers incomplete syntax across response chunks, flushes at
    response end and resets on interruption. Independent ``TTSSpeakFrame``
    utterances use the transform's ordinary ``(text, text_type)`` callable
    without consuming the response buffer. Recorded audio and its transcript
    pass through unchanged.
    """

    def __init__(
        self,
        *,
        transform: SpokenTextTransform | None = None,
        skip_text_types: tuple[str, ...] = ("recording", "recording_router"),
        **kwargs,
    ):
        """Initialize a sanitizer owned by one generation pipeline.

        Args:
            transform: Sanitization transform with response-scoped streaming state.
            skip_text_types: Text belonging to audio that has already been synthesized.
            **kwargs: Arguments forwarded to the frame processor.
        """
        super().__init__(**kwargs)
        self._transform = transform or SpokenTextTransform()
        self._skip_text_types = skip_text_types
        self._receiving_response = False
        self._last_text_frame: TextFrame | None = None
        self._stream_failed = False

    async def _reset(self):
        await self._transform.reset()
        self._last_text_frame = None
        self._stream_failed = False
        self._receiving_response = False

    @staticmethod
    def _rewritten(frame: TextFrame | TTSSpeakFrame, text: str, *, new_frame: bool = False):
        rewritten = replace(frame, text=text)
        # Preserve routing, interruptibility and observer identity across a
        # rewrite. A flushed tail is an additional frame and needs its own ID.
        for item in fields(frame):
            if not item.init and (not new_frame or item.name not in ("id", "name")):
                setattr(rewritten, item.name, getattr(frame, item.name))
        if isinstance(rewritten, AggregatedTextFrame):
            rewritten.raw_text = text
        return rewritten

    async def _report_transform_error(self, exception: Exception):
        await self.push_error(
            error_msg=f"Error transforming spoken text: {exception}",
            exception=exception,
            category=ErrorCategory.APPLICATION,
        )

    async def _transform_frame(self, frame: TextFrame | TTSSpeakFrame, *, streaming: bool):
        if streaming:
            self._last_text_frame = frame
            if self._stream_failed:
                return
        try:
            text = (
                await self._transform.transform_stream(frame.text)
                if streaming
                else await self._transform(
                    frame.text, getattr(frame, "text_type", TextType.SENTENCE)
                )
            )
        except Exception as error:
            if streaming:
                self._stream_failed = True
            await self._report_transform_error(error)
            return
        if text or isinstance(frame, TTSSpeakFrame):
            await self.push_frame(self._rewritten(frame, text))

    async def _flush(self):
        try:
            if not self._stream_failed:
                text = await self._transform.flush()
                if text and self._last_text_frame is not None:
                    await self.push_frame(
                        self._rewritten(self._last_text_frame, text, new_frame=True)
                    )
        except Exception as error:
            await self._report_transform_error(error)
        finally:
            await self._reset()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Transform assistant text while preserving lifecycle and routing frames."""
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, CancelFrame, StartFrame)):
            await self._reset()
        elif direction == FrameDirection.UPSTREAM:
            pass
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._reset()
            self._receiving_response = True
        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self._flush()
        elif isinstance(frame, TTSSpeakFrame):
            await self._transform_frame(frame, streaming=False)
            return
        elif isinstance(frame, AggregatedTextFrame):
            if frame.text_type not in self._skip_text_types:
                await self._transform_frame(frame, streaming=self._receiving_response)
                return
        elif isinstance(frame, LLMTextFrame) or type(frame) is TextFrame:
            await self._transform_frame(frame, streaming=True)
            return
        await self.push_frame(frame, direction)
