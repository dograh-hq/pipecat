#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio buffer processor for managing and synchronizing audio streams.

This module provides an AudioBufferProcessor that handles buffering and synchronization
of audio from both user input and bot output sources, with support for various audio
configurations and event-driven processing.
"""

import time
from typing import Optional

from loguru import logger

from pipecat.audio.utils import (
    create_stream_resampler,
    interleave_stereo_audio,
    mix_audio,
    ulaw_to_pcm,
)
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.audio.audio_buffer_input_processor import AudioBufferInputProcessor
from pipecat.processors.audio.audio_buffer_output_processor import AudioBufferOutputProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioBufferProcessor(FrameProcessor):
    """Processes and buffers audio frames from both input (user) and output (bot) sources.

    This processor manages audio buffering and synchronization, providing both merged and
    track-specific audio access through event handlers. It supports various audio configurations
    including sample rate conversion and mono/stereo output.

    Events:

    - on_audio_data: Triggered when buffer_size is reached, providing merged audio
    - on_track_audio_data: Triggered when buffer_size is reached, providing separate tracks
    - on_user_turn_audio_data: Triggered when user turn has ended, providing that user turn's audio
    - on_bot_turn_audio_data: Triggered when bot turn has ended, providing that bot turn's audio

    Audio handling:

    - Mono output (num_channels=1): User and bot audio are mixed
    - Stereo output (num_channels=2): User audio on left, bot audio on right
    - Automatic resampling of incoming audio to match desired sample_rate
    - Silence insertion for non-continuous audio streams
    - Buffer synchronization between user and bot audio
    """

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        num_channels: int = 1,
        buffer_size: int = 0,
        user_continuous_stream: Optional[bool] = None,
        enable_turn_audio: bool = False,
        **kwargs,
    ):
        """Initialize the audio buffer processor.

        Args:
            sample_rate: Desired output sample rate. If None, uses source rate.
            num_channels: Number of channels (1 for mono, 2 for stereo). Defaults to 1.
            buffer_size: Size of buffer before triggering events. 0 for no buffering.
            user_continuous_stream: Controls whether user audio is treated as a continuous
                stream for buffering purposes.

                .. deprecated:: 0.0.72
                    This parameter no longer has any effect and will be removed in a future version.

            enable_turn_audio: Whether turn audio event handlers should be triggered.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._audio_buffer_size_1s = 0
        self._num_channels = num_channels
        self._buffer_size = buffer_size
        self._enable_turn_audio = enable_turn_audio

        if user_continuous_stream is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter `user_continuous_stream` is deprecated.",
                    DeprecationWarning,
                )

        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

        self._user_speaking = False
        self._bot_speaking = False
        self._user_turn_audio_buffer = bytearray()
        self._bot_turn_audio_buffer = bytearray()

        # Intermittent (non continous user stream variables)
        self._last_user_frame_at = 0
        self._last_bot_frame_at = 0

        self._recording = False

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

        self._register_event_handler("on_audio_data")
        self._register_event_handler("on_track_audio_data")
        self._register_event_handler("on_user_turn_audio_data")
        self._register_event_handler("on_bot_turn_audio_data")

    @property
    def sample_rate(self) -> int:
        """Current sample rate of the audio processor.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Number of channels in the audio output.

        Returns:
            Number of channels (1 for mono, 2 for stereo).
        """
        return self._num_channels

    def has_audio(self) -> bool:
        """Check if both user and bot audio buffers contain data.

        Returns:
            True if both buffers contain audio data.
        """
        return self._buffer_has_audio(self._user_audio_buffer) and self._buffer_has_audio(
            self._bot_audio_buffer
        )

    def merge_audio_buffers(self) -> bytes:
        """Merge user and bot audio buffers into a single audio stream.

        For mono output, audio is mixed. For stereo output, user audio is placed
        on the left channel and bot audio on the right channel.

        Returns:
            Mixed audio data as bytes.
        """
        if self._num_channels == 1:
            return mix_audio(bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer))
        elif self._num_channels == 2:
            return interleave_stereo_audio(
                bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer)
            )
        else:
            return b""

    async def start_recording(self):
        """Start recording audio from both user and bot.

        Initializes recording state and resets audio buffers.
        """
        self._recording = True
        self._reset_recording()

    async def stop_recording(self):
        """Stop recording and trigger final audio data handlers.

        Calls audio handlers with any remaining buffered audio before stopping.
        """
        await self._call_on_audio_data_handler()
        self._recording = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming audio frames and manage audio buffers.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Update output sample rate if necessary.
        if isinstance(frame, StartFrame):
            self._update_sample_rate(frame)

        if self._recording:
            await self._process_recording(frame)
            if self._enable_turn_audio:
                await self._process_turn_recording(frame)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.stop_recording()

        await self.push_frame(frame, direction)

    def _update_sample_rate(self, frame: StartFrame):
        """Update the sample rate from the start frame."""
        self._sample_rate = self._init_sample_rate or frame.audio_out_sample_rate
        self._audio_buffer_size_1s = self._sample_rate * 2

    async def _process_recording(self, frame: Frame):
        """Process audio frames for recording."""
        if isinstance(frame, InputAudioRawFrame):
            # Add silence if we need to.
            silence = self._compute_silence(self._last_user_frame_at)
            self._user_audio_buffer.extend(silence)
            # Add user audio.
            resampled = await self._resample_input_audio(frame)
            self._user_audio_buffer.extend(resampled)
            # Save time of frame so we can compute silence.
            self._last_user_frame_at = time.time()
        elif self._recording and isinstance(frame, OutputAudioRawFrame):
            # Add silence if we need to.
            silence = self._compute_silence(self._last_bot_frame_at)
            self._bot_audio_buffer.extend(silence)
            # Add bot audio.
            resampled = await self._resample_output_audio(frame)
            self._bot_audio_buffer.extend(resampled)
            # Save time of frame so we can compute silence.
            self._last_bot_frame_at = time.time()

        if self._buffer_size > 0 and len(self._user_audio_buffer) > self._buffer_size:
            await self._call_on_audio_data_handler()

    async def _process_turn_recording(self, frame: Frame):
        """Process frames for turn-based audio recording."""
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._call_event_handler(
                "on_user_turn_audio_data", self._user_turn_audio_buffer, self.sample_rate, 1
            )
            self._user_speaking = False
            self._user_turn_audio_buffer = bytearray()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._call_event_handler(
                "on_bot_turn_audio_data", self._bot_turn_audio_buffer, self.sample_rate, 1
            )
            self._bot_speaking = False
            self._bot_turn_audio_buffer = bytearray()

        if isinstance(frame, InputAudioRawFrame):
            resampled = await self._resample_input_audio(frame)
            self._user_turn_audio_buffer += resampled
            # In the case of the user, we need to keep a short buffer of audio
            # since VAD notification of when the user starts speaking comes
            # later.
            if (
                not self._user_speaking
                and len(self._user_turn_audio_buffer) > self._audio_buffer_size_1s
            ):
                discarded = len(self._user_turn_audio_buffer) - self._audio_buffer_size_1s
                self._user_turn_audio_buffer = self._user_turn_audio_buffer[discarded:]
        elif self._bot_speaking and isinstance(frame, OutputAudioRawFrame):
            resampled = await self._resample_output_audio(frame)
            self._bot_turn_audio_buffer += resampled

    async def _call_on_audio_data_handler(self):
        """Call the audio data event handlers with buffered audio."""
        if not self.has_audio() or not self._recording:
            return

        # Call original handler with merged audio
        merged_audio = self.merge_audio_buffers()
        await self._call_event_handler(
            "on_audio_data", merged_audio, self._sample_rate, self._num_channels
        )

        # Call new handler with separate tracks
        await self._call_event_handler(
            "on_track_audio_data",
            bytes(self._user_audio_buffer),
            bytes(self._bot_audio_buffer),
            self._sample_rate,
            self._num_channels,
        )

        self._reset_audio_buffers()

    def _buffer_has_audio(self, buffer: bytearray) -> bool:
        """Check if a buffer contains audio data."""
        return buffer is not None and len(buffer) > 0

    def _reset_recording(self):
        """Reset recording state and buffers."""
        self._reset_audio_buffers()
        self._last_user_frame_at = time.time()
        self._last_bot_frame_at = time.time()

    def _reset_audio_buffers(self):
        """Reset all audio buffers to empty state."""
        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()
        self._user_turn_audio_buffer = bytearray()
        self._bot_turn_audio_buffer = bytearray()

    async def _resample_input_audio(self, frame: InputAudioRawFrame) -> bytes:
        """Resample audio frame to the target sample rate."""
        # Decode μ-law if required
        target_rate = self._sample_rate or frame.sample_rate
        if getattr(frame, "metadata", {}).get("audio_encoding") == "ulaw":
            return await ulaw_to_pcm(frame.audio, frame.sample_rate, target_rate, self._resampler)

        return await self._input_resampler.resample(
            frame.audio, frame.sample_rate, self._sample_rate
        )

    async def _resample_output_audio(self, frame: OutputAudioRawFrame) -> bytes:
        """Resample audio frame to the target sample rate."""
        return await self._output_resampler.resample(
            frame.audio, frame.sample_rate, self._sample_rate
        )

    def _compute_silence(self, from_time: float) -> bytes:
        """Compute silence to insert based on time gap."""
        quiet_time = time.time() - from_time
        # We should get audio frames very frequently. We introduce silence only
        # if there's a big enough gap of 1s.
        if from_time == 0 or quiet_time < 1.0:
            return b""
        num_bytes = int(quiet_time * self._sample_rate) * 2
        silence = b"\x00" * num_bytes
        return silence


class AudioBuffer:
    """Factory for creating and managing split audio buffer processors.

    Provides unified access to input and output audio buffer processors with the ability
    to record audio separately for better pipeline efficiency.

    Example:
        ```python
        # Create split audio buffers
        audio_buffer = AudioBuffer(sample_rate=16000, buffer_size=24000)

        # Create synchronizer for merged audio (outside pipeline)
        audio_synchronizer = AudioSynchronizer(sample_rate=16000, buffer_size=24000)
        audio_synchronizer.register_processors(
            audio_buffer.input(),
            audio_buffer.output()
        )

        pipeline = Pipeline([
            transport.input(),
            stt_mute_filter,
            audio_buffer.input(),  # Only processes InputAudioRawFrame
            stt,  # audio_passthrough=False
            # ... rest of pipeline ...
            tts,
            audio_buffer.output(),  # Only processes OutputAudioRawFrame
            transport.output(),
        ])

        # Register handler for merged audio
        @audio_synchronizer.event_handler("on_merged_audio")
        async def handle_merged_audio(synchronizer, pcm, sample_rate, num_channels):
            # Process merged audio
            pass
        ```
    """

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        buffer_size: int = 0,
        enable_turn_audio: bool = False,
    ):
        """Initialize factory with shared configuration.

        Args:
            sample_rate: Desired output sample rate. If None, uses source rate
            buffer_size: Size of buffer before triggering events. 0 for no buffering
            enable_turn_audio: Whether turn audio event handlers should be triggered
        """
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._enable_turn_audio = enable_turn_audio
        self._input_processor = None
        self._output_processor = None

    def input(self, **kwargs) -> AudioBufferInputProcessor:
        """Get the input audio buffer processor.

        Args:
            **kwargs: Additional arguments specific to AudioBufferInputProcessor

        Returns:
            AudioBufferInputProcessor instance
        """
        if self._input_processor is None:
            self._input_processor = AudioBufferInputProcessor(
                sample_rate=self._sample_rate,
                buffer_size=self._buffer_size,
                enable_turn_audio=self._enable_turn_audio,
                **kwargs,
            )
        else:
            logger.debug(f"AudioBuffer: Returning existing AudioBufferInputProcessor")
        return self._input_processor

    def output(self, **kwargs) -> AudioBufferOutputProcessor:
        """Get the output audio buffer processor.

        Args:
            **kwargs: Additional arguments specific to AudioBufferOutputProcessor

        Returns:
            AudioBufferOutputProcessor instance
        """
        if self._output_processor is None:
            self._output_processor = AudioBufferOutputProcessor(
                sample_rate=self._sample_rate,
                buffer_size=self._buffer_size,
                enable_turn_audio=self._enable_turn_audio,
                **kwargs,
            )
        else:
            logger.debug(f"AudioBuffer: Returning existing AudioBufferOutputProcessor")
        return self._output_processor

    async def start_recording(self):
        """Start recording on both input and output processors."""
        if self._input_processor:
            await self._input_processor.start_recording()
        else:
            logger.warning(f"AudioBuffer: No input processor created yet!")
        if self._output_processor:
            await self._output_processor.start_recording()
        else:
            logger.warning(f"AudioBuffer: No output processor created yet!")

    async def stop_recording(self):
        """Stop recording on both input and output processors."""
        if self._input_processor:
            await self._input_processor.stop_recording()
        if self._output_processor:
            await self._output_processor.stop_recording()
