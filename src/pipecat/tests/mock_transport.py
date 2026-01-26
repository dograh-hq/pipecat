#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mock transport implementation for testing Pipecat pipelines.

This module provides a simple mock transport that can be used in tests
to verify pipeline behavior without needing a real transport connection.
"""

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams


class MockInputTransport(FrameProcessor):
    """Mock input transport processor for testing.

    Can generate InputAudioRawFrame at regular intervals to simulate
    real audio input from a transport. Audio generation starts when
    StartFrame is received and stops when EndFrame or CancelFrame is received.
    """

    def __init__(
        self,
        params: Optional[TransportParams] = None,
        *,
        generate_audio: bool = False,
        audio_interval_ms: int = 20,
        sample_rate: int = 16000,
        num_channels: int = 1,
        **kwargs,
    ):
        """Initialize the mock input transport.

        Args:
            params: Optional transport parameters.
            generate_audio: If True, generates InputAudioRawFrame at regular intervals.
            audio_interval_ms: Interval between audio frames in milliseconds (default: 20ms).
            sample_rate: Audio sample rate in Hz (default: 16000).
            num_channels: Number of audio channels (default: 1).
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._params = params or TransportParams()
        self._generate_audio = generate_audio
        self._audio_interval_ms = audio_interval_ms
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._audio_task: Optional[asyncio.Task] = None
        self._running = False

    async def _generate_audio_frames(self):
        """Generate audio frames at regular intervals."""
        # Calculate bytes needed for the interval duration
        # PCM 16-bit audio: 2 bytes per sample per channel
        samples_per_frame = int(self._sample_rate * self._audio_interval_ms / 1000)
        bytes_per_frame = samples_per_frame * self._num_channels * 2

        # Generate silence (zeros) as the audio data
        silence_audio = bytes(bytes_per_frame)

        while self._running:
            try:
                frame = InputAudioRawFrame(
                    audio=silence_audio,
                    sample_rate=self._sample_rate,
                    num_channels=self._num_channels,
                )
                await self.push_frame(frame)
                await asyncio.sleep(self._audio_interval_ms / 1000)
            except asyncio.CancelledError:
                break

    def _start_audio_generation(self):
        """Start the audio generation task."""
        if self._generate_audio and not self._running:
            self._running = True
            self._audio_task = asyncio.create_task(self._generate_audio_frames())

    def _stop_audio_generation(self):
        """Stop the audio generation task."""
        self._running = False
        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
            self._audio_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames by passing them through.

        Starts audio generation on StartFrame and stops on EndFrame/CancelFrame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._start_audio_generation()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._stop_audio_generation()

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up resources."""
        self._stop_audio_generation()
        await super().cleanup()


class MockOutputTransport(FrameProcessor):
    """Mock output transport processor for testing.

    Simulates bot speaking behavior by emitting BotStartedSpeaking,
    BotSpeaking, and BotStoppedSpeaking frames when TTS frames are received.
    """

    def __init__(
        self,
        params: Optional[TransportParams] = None,
        *,
        emit_bot_speaking: bool = True,
        **kwargs,
    ):
        """Initialize the mock output transport.

        Args:
            params: Optional transport parameters.
            emit_bot_speaking: If True, emits BotSpeakingFrame on TTSAudioRawFrame.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._params = params or TransportParams()
        self._emit_bot_speaking = emit_bot_speaking

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and simulate bot speaking behavior.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            await self.push_frame(BotStartedSpeakingFrame())
            await self.push_frame(BotStartedSpeakingFrame(), direction=FrameDirection.UPSTREAM)
        elif isinstance(frame, TTSAudioRawFrame):
            if self._emit_bot_speaking:
                await self.push_frame(BotSpeakingFrame())
                await self.push_frame(BotSpeakingFrame(), direction=FrameDirection.UPSTREAM)
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(BotStoppedSpeakingFrame())
            await self.push_frame(BotStoppedSpeakingFrame(), direction=FrameDirection.UPSTREAM)

        await self.push_frame(frame, direction)


class MockTransport(BaseTransport):
    """Mock transport for testing Pipecat pipelines.

    Provides simple input and output transport processors that can be
    used in tests without needing actual WebSocket or WebRTC connections.
    Can optionally generate audio frames to simulate real input.
    """

    def __init__(
        self,
        params: Optional[TransportParams] = None,
        *,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        emit_bot_speaking: bool = True,
        generate_audio: bool = False,
        audio_interval_ms: int = 20,
        audio_sample_rate: int = 16000,
        audio_num_channels: int = 1,
    ):
        """Initialize the mock transport.

        Args:
            params: Optional transport parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
            emit_bot_speaking: If True, output transport emits BotSpeakingFrame.
            generate_audio: If True, input transport generates InputAudioRawFrame at intervals.
            audio_interval_ms: Interval between audio frames in milliseconds (default: 20ms).
            audio_sample_rate: Audio sample rate in Hz (default: 16000).
            audio_num_channels: Number of audio channels (default: 1).
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params or TransportParams()
        self._input = MockInputTransport(
            self._params,
            name=self._input_name,
            generate_audio=generate_audio,
            audio_interval_ms=audio_interval_ms,
            sample_rate=audio_sample_rate,
            num_channels=audio_num_channels,
        )
        self._output = MockOutputTransport(
            self._params,
            emit_bot_speaking=emit_bot_speaking,
            name=self._output_name,
        )

    def input(self) -> FrameProcessor:
        """Get the mock input transport processor.

        Returns:
            The mock input transport instance.
        """
        return self._input

    def output(self) -> FrameProcessor:
        """Get the mock output transport processor.

        Returns:
            The mock output transport instance.
        """
        return self._output
