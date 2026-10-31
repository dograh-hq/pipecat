"""Compatibility adapter for the deprecated TTS text-filter interface."""

from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.transforms.spoken_text import SpokenTextSanitizer, SpokenTextTransform

__all__ = ["SpokenTextFilter", "SpokenTextSanitizer"]


class SpokenTextFilter(SpokenTextTransform, BaseTextFilter):
    """Adapt complete-utterance sanitization to legacy text filters.

    Streaming speech is sanitized by ``SpokenTextProcessor`` before TTS.
    """

    filter_before_aggregation = True

    async def filter(self, text: str) -> str:
        """Sanitize an independent utterance through the transform callable."""
        return await self(text, "*")

    async def filter_stream(self, text: str) -> str:
        """Sanitize a chunk for callers of the legacy streaming helper."""
        return await self.transform_stream(text)

    async def handle_interruption(self):
        """Discard the current response's buffered control text."""
        await self.reset()
