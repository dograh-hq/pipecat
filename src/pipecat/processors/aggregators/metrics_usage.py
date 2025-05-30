import time
from collections import defaultdict
from typing import Dict, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    EndTaskFrame,
    Frame,
    MetricsFrame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.enums import EndTaskReason


class UsageMetricsAggregator(FrameProcessor):
    def __init__(self, max_call_duration_seconds=300):
        super().__init__()
        # Structure: {f"{processor}|||{model}": aggregated_metrics}
        # For LLM: aggregated_metrics is LLMTokenUsage
        # For TTS: aggregated_metrics is int (total characters)
        self._max_call_duration_seconds = max_call_duration_seconds

        self._call_duration = 0
        self._start_time: Optional[float] = None
        self._llm_usage_metrics: Dict[str, LLMTokenUsage] = {}
        self._tts_usage_metrics: Dict[str, int] = defaultdict(int)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            await self._stop(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)
        elif isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, LLMUsageMetricsData):
                    await self._handle_llm_usage_metrics(data)
                elif isinstance(data, TTSUsageMetricsData):
                    await self._handle_tts_usage_metrics(data)

        await self._check_call_duration()

        await self.push_frame(frame, direction)

    async def _check_call_duration(self):
        if self._start_time is not None:
            if time.time() - self._start_time > self._max_call_duration_seconds:
                await self.push_frame(
                    EndTaskFrame(metadata={"reason": EndTaskReason.CALL_DURATION_EXCEEDED.value}),
                    FrameDirection.UPSTREAM,
                )

    async def _start(self, _: StartFrame):
        """Start tracking call duration."""
        self._start_time = time.time()
        logger.debug("Started call duration tracking")

    async def _stop(self, _: EndFrame):
        """Stop tracking call duration."""
        if self._start_time is not None:
            self._call_duration = time.time() - self._start_time
            logger.debug(f"Call duration: {self._call_duration:.2f} seconds")
            self._start_time = None

    async def _cancel(self, _: CancelFrame):
        """Handle call cancellation - also stop tracking duration."""
        if self._start_time is not None:
            self._call_duration = time.time() - self._start_time
            logger.debug(f"Call cancelled, duration: {self._call_duration:.2f} seconds")
            self._start_time = None

    async def _handle_llm_usage_metrics(self, data: LLMUsageMetricsData):
        key = f"{data.processor}|||{data.model}"
        new_usage = data.value

        if key in self._llm_usage_metrics:
            # Aggregate with existing metrics
            existing = self._llm_usage_metrics[key]
            aggregated = LLMTokenUsage(
                prompt_tokens=existing.prompt_tokens + new_usage.prompt_tokens,
                completion_tokens=existing.completion_tokens + new_usage.completion_tokens,
                total_tokens=existing.total_tokens + new_usage.total_tokens,
                cache_read_input_tokens=(existing.cache_read_input_tokens or 0)
                + (new_usage.cache_read_input_tokens or 0),
                cache_creation_input_tokens=(existing.cache_creation_input_tokens or 0)
                + (new_usage.cache_creation_input_tokens or 0),
            )
            self._llm_usage_metrics[key] = aggregated
        else:
            # First occurrence for this processor+model combination
            self._llm_usage_metrics[key] = LLMTokenUsage(
                prompt_tokens=new_usage.prompt_tokens,
                completion_tokens=new_usage.completion_tokens,
                total_tokens=new_usage.total_tokens,
                cache_read_input_tokens=new_usage.cache_read_input_tokens,
                cache_creation_input_tokens=new_usage.cache_creation_input_tokens,
            )

        logger.debug(f"LLM usage metrics: {self._llm_usage_metrics}")

    async def _handle_tts_usage_metrics(self, data: TTSUsageMetricsData):
        key = f"{data.processor}|||{data.model}"
        self._tts_usage_metrics[key] += data.value
        logger.debug(f"TTS usage metrics: {self._tts_usage_metrics}")

    def get_llm_usage_metrics(self) -> Dict[str, LLMTokenUsage]:
        """Get the aggregated LLM usage metrics grouped by processor|||model."""
        return dict(self._llm_usage_metrics)

    def get_tts_usage_metrics(self) -> Dict[str, int]:
        """Get the aggregated TTS usage metrics grouped by processor|||model."""
        return dict(self._tts_usage_metrics)

    def get_call_duration(self) -> float:
        """Get the call duration in seconds."""
        return self._call_duration

    def get_all_usage_metrics_serialized(self) -> Dict[str, Dict[str, any]]:
        """Get all aggregated usage metrics in JSON-serializable format."""
        serialized_llm = {}
        for key, usage in self._llm_usage_metrics.items():
            serialized_llm[key] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            }

        return {
            "llm": serialized_llm,
            "tts": dict(self._tts_usage_metrics),
            "call_duration_seconds": self._call_duration,
        }

    def reset_metrics(self):
        """Reset all aggregated metrics."""
        self._llm_usage_metrics.clear()
        self._tts_usage_metrics.clear()
        self._call_duration = 0
        self._start_time = None
