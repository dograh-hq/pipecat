#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests for XMLFunctionTagFilter with TTS services."""

import asyncio

import pytest

from pipecat.frames.frames import (
    EndFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter


class MockTTSServiceWithFilter(TTSService):
    """Mock TTS service that demonstrates using XMLFunctionTagFilter."""
    
    def __init__(self, **kwargs):
        xml_function_tag_filter = XMLFunctionTagFilter()
        
        super().__init__(text_filters=[xml_function_tag_filter], **kwargs)
        
        self.received_texts = []
        
    async def run_tts(self, text: str):
        """Mock TTS processing that captures filtered text and generates frames."""

        self.received_texts.append(text)
        
        # Generate frames to simulate TTS processing
        yield TTSStartedFrame()
        
        # Only generate audio if we have text (to avoid empty frame issues)
        if text.strip():
            # Generate a simple audio frame
            dummy_audio = b"\x00\x01\x02\x03" * 1000  # 4KB of dummy audio
            audio_frame = TTSAudioRawFrame(
                audio=dummy_audio,
                sample_rate=24000,
                num_channels=1
            )
            yield audio_frame
                
        # Emit completion frames
        yield TTSStoppedFrame()
        yield TTSTextFrame(text=text)


@pytest.mark.asyncio
async def test_tts_service_with_xml_function_tag_filter():
    """Test that TTS service properly applies XMLFunctionTagFilter to incoming text."""
    
    tts_with_filter = MockTTSServiceWithFilter()
    
    # Set up frame collection
    received_down = asyncio.Queue()
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
    )
    
    # Create pipeline: TTS (with filter) -> sink
    pipeline = Pipeline([tts_with_filter, sink])
    task = PipelineTask(pipeline, cancel_on_idle_timeout=False)
    
    # Test data with function call tags
    original_text = ("Hello! I can help you schedule that meeting. "
                     "<function=schedule_interview>{\"date\": \"tomorrow\", \"time\": \"3 PM\"}</function> "
                     "The interview has been scheduled successfully!")
    
    expected_filtered_text = ("Hello! I can help you schedule that meeting. "
                             "The interview has been scheduled successfully!")
    
    async def send_frames():
        await asyncio.sleep(0.01)  # Small delay to let pipeline start
        
        # Send text frames with function call tags
        text_frame = LLMTextFrame(text=original_text)
        await task.queue_frame(text_frame)
        await task.queue_frame(EndFrame())
    
    # Run the pipeline
    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), send_frames())
    
    # Verify that TTS received the filtered text (without function call tags)
    assert len(tts_with_filter.received_texts) >= 1, f"TTS should have received text, got: {tts_with_filter.received_texts}"
    
    # Verify no function call tags in any received text
    for text in tts_with_filter.received_texts:
        assert "<function=" not in text, f"TTS should not receive function call tags: '{text}'"
        assert "</function>" not in text, f"TTS should not receive function call tags: '{text}'"
    
    # Verify the combined text matches our expectation (allowing for sentence splitting)
    all_received_text = " ".join(tts_with_filter.received_texts)
    assert all_received_text == expected_filtered_text, f"Expected: '{expected_filtered_text}', Got: '{all_received_text}'"
    
    # Collect frames to verify integration worked
    received_frames = []
    while not received_down.empty():
        frame = await received_down.get()
        if not isinstance(frame, EndFrame):
            received_frames.append(frame)
    
    # Verify that we got the expected frame types (showing TTS pipeline worked)
    frame_types = [type(f) for f in received_frames]
    assert TTSStartedFrame in frame_types, "Should receive TTSStartedFrame"
    assert TTSStoppedFrame in frame_types, "Should receive TTSStoppedFrame"
    assert TTSTextFrame in frame_types, "Should receive TTSTextFrame"
    
    print(f"Original: {original_text}")
    print(f"Filtered: {all_received_text}")
    print(f"Integration test successful")

if __name__ == "__main__":
    pytest.main([__file__])