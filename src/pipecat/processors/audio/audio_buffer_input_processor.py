#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
from typing import Optional

from loguru import logger

from pipecat.audio.utils import create_default_resampler, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioBufferInputProcessor(FrameProcessor):
    """Processes and buffers input audio frames from user sources.

    This processor manages input audio buffering with support for sample rate conversion
    and turn-based audio capture.

    Events:
        on_input_audio_data: Triggered when buffer_size is reached, providing input audio
        on_user_turn_audio_data: Triggered when user turn has ended, providing that turn's audio

    Args:
        sample_rate (Optional[int]): Desired output sample rate. If None, uses source rate
        buffer_size (int): Size of buffer before triggering events. 0 for no buffering
        enable_turn_audio (bool): Whether turn audio event handlers should be triggered
    """

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        buffer_size: int = 0,
        enable_turn_audio: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._audio_buffer_size_1s = 0
        self._buffer_size = buffer_size
        self._enable_turn_audio = enable_turn_audio

        self._audio_buffer = bytearray()
        self._turn_audio_buffer = bytearray()
        self._user_speaking = False

        # For handling intermittent audio streams
        self._last_frame_at = 0

        self._recording = False
        self._resampler = create_default_resampler()

        self._register_event_handler("on_input_audio_data")
        self._register_event_handler("on_user_turn_audio_data")

    @property
    def sample_rate(self) -> int:
        """Current sample rate of the audio processor."""
        return self._sample_rate

    async def start_recording(self):
        """Start recording audio from input."""
        # logger.debug(f"AudioBufferInputProcessor: Starting recording")
        self._recording = True
        self._reset_recording()

    async def stop_recording(self):
        """Stop recording and trigger final audio data handlers."""
        logger.debug(f"AudioBufferInputProcessor: Stopping recording")
        await self._call_audio_handler()
        self._recording = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming audio frames and manage audio buffers."""
        await super().process_frame(frame, direction)

        # Update sample rate if necessary
        if isinstance(frame, StartFrame):
            self._update_sample_rate(frame)

        if self._recording:
            if isinstance(frame, InputAudioRawFrame):
                await self._process_audio_frame(frame)

            if self._enable_turn_audio:
                await self._process_turn_recording(frame)
        elif isinstance(frame, InputAudioRawFrame):
            logger.warning(
                f"AudioBufferInputProcessor: Received InputAudioRawFrame but not recording!"
            )

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.stop_recording()

        await self.push_frame(frame, direction)

    def _update_sample_rate(self, frame: StartFrame):
        self._sample_rate = self._init_sample_rate or frame.audio_in_sample_rate
        self._audio_buffer_size_1s = self._sample_rate * 2
        logger.info(
            f"AudioBufferInputProcessor: Sample rate set to {self._sample_rate}Hz (init={self._init_sample_rate}, frame={frame.audio_in_sample_rate})"
        )

    async def _process_audio_frame(self, frame: InputAudioRawFrame):
        # Add silence if we need to
        silence = self._compute_silence(self._last_frame_at)
        self._audio_buffer.extend(silence)

        # Add audio
        resampled = await self._resample_audio(frame)
        self._audio_buffer.extend(resampled)

        # Save time of frame so we can compute silence
        self._last_frame_at = time.time()

        # Check if we need to trigger handlers
        if self._buffer_size > 0 and len(self._audio_buffer) > self._buffer_size:
            await self._call_audio_handler()

    async def _process_turn_recording(self, frame: Frame):
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._call_event_handler(
                "on_user_turn_audio_data", bytes(self._turn_audio_buffer), self._sample_rate, 1
            )
            self._user_speaking = False
            self._turn_audio_buffer = bytearray()

        if isinstance(frame, InputAudioRawFrame):
            resampled = await self._resample_audio(frame)
            self._turn_audio_buffer += resampled

            # Keep a short buffer when user is not speaking (for VAD delay)
            if (
                not self._user_speaking
                and len(self._turn_audio_buffer) > self._audio_buffer_size_1s
            ):
                discarded = len(self._turn_audio_buffer) - self._audio_buffer_size_1s
                self._turn_audio_buffer = self._turn_audio_buffer[discarded:]

    async def _call_audio_handler(self):
        if not self._buffer_has_audio() or not self._recording:
            return

        await self._call_event_handler(
            "on_input_audio_data",
            bytes(self._audio_buffer),
            self._sample_rate,
            1,  # Always mono for input
        )

        self._audio_buffer = bytearray()

    def _buffer_has_audio(self) -> bool:
        return len(self._audio_buffer) > 0

    def _reset_recording(self):
        self._audio_buffer = bytearray()
        self._turn_audio_buffer = bytearray()
        self._last_frame_at = time.time()

    async def _resample_audio(self, frame: AudioRawFrame) -> bytes:
        """Return PCM audio for frame at the processor's output rate."""
        target_rate = self._sample_rate or frame.sample_rate

        # Decode μ-law if required
        if getattr(frame, "metadata", {}).get("audio_encoding") == "ulaw":
            return await ulaw_to_pcm(frame.audio, frame.sample_rate, target_rate, self._resampler)

        # Default PCM path
        return await self._resampler.resample(frame.audio, frame.sample_rate, target_rate)

    def _compute_silence(self, from_time: float) -> bytes:
        quiet_time = time.time() - from_time
        # Introduce silence only if there's a big enough gap of 1s
        if from_time == 0 or quiet_time < 1.0:
            return b""
        num_bytes = int(quiet_time * self._sample_rate) * 2
        silence = b"\x00" * num_bytes
        return silence
