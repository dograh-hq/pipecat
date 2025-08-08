#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio synchronizer for merging input and output audio streams."""

import asyncio
from typing import Callable, Dict, List

from loguru import logger

from pipecat.audio.utils import create_default_resampler, interleave_stereo_audio, mix_audio


class AudioSynchronizer:
    """Synchronizes audio from separate input and output processors without being in the pipeline.

    This class subscribes to events from AudioBufferInputProcessor and AudioBufferOutputProcessor,
    buffers the audio data, and emits merged audio when both buffers have sufficient data.

    Events:
        on_merged_audio: Triggered when synchronized audio is available

    Args:
        sample_rate (int): The sample rate for audio processing
        buffer_size (int): Size of buffer before triggering merged audio events
        num_channels (int): Number of channels for merged output (1 for mono, 2 for stereo)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_size: int = 8000,  # 0.5 seconds at 16kHz
        num_channels: int = 1,
    ):
        """Initialize audio synchronizer.

        Args:
            sample_rate: The sample rate for audio processing in Hz.
            buffer_size: Size of buffer before triggering merged audio events.
            num_channels: Number of channels for merged output (1 for mono, 2 for stereo).
        """
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._num_channels = num_channels

        # CORE AUDIO BUFFERS: Store incoming audio data
        self._input_buffer = bytearray()   # User microphone audio
        self._output_buffer = bytearray()  # Bot TTS audio

        # BUFFER SIZE LIMITS: Prevent memory issues from runaway accumulation
        # Maximum allowed buffer size before emergency cleanup (10x normal buffer)
        self._max_buffer_size = buffer_size * 10
        # Warning threshold for detecting accumulation issues (5x normal buffer) 
        self._warning_buffer_size = buffer_size * 5

        self._event_handlers: Dict[str, List[Callable]] = {"on_merged_audio": []}
        self._resampler = create_default_resampler()

        self._input_processor = None
        self._output_processor = None

        # RECORDING STATE: Track if we're currently recording audio
        self._recording = False

    def register_processors(self, input_processor, output_processor):
        """Register input and output processors to synchronize.

        Args:
            input_processor: AudioBufferInputProcessor instance
            output_processor: AudioBufferOutputProcessor instance
        """
        logger.info(f"AudioSynchronizer: Registering processors")
        self._input_processor = input_processor
        self._output_processor = output_processor

        # Subscribe to events from both processors
        if input_processor:
            input_processor.add_event_handler("on_input_audio_data", self._handle_input_audio)
        else:
            logger.warning(f"AudioSynchronizer: No input processor provided!")

        if output_processor:
            output_processor.add_event_handler("on_output_audio_data", self._handle_output_audio)
        else:
            logger.warning(f"AudioSynchronizer: No output processor provided!")

    async def start_recording(self):
        """Start recording and synchronizing audio."""
        self._recording = True
        self._reset_buffers()

    async def stop_recording(self):
        """Stop recording and flush remaining audio."""
        await self._call_audio_handler()
        self._recording = False

    def _has_audio(self) -> bool:
        """Check if both buffers contain audio data."""
        return self._buffer_has_audio(self._input_buffer) and self._buffer_has_audio(
            self._output_buffer
        )

    def _buffer_has_audio(self, buffer: bytearray) -> bool:
        """Check if a buffer contains audio data."""
        return buffer is not None and len(buffer) > 0

    async def _handle_input_audio(self, processor, pcm: bytes, sample_rate: int, num_channels: int):
        """Handle incoming input audio data."""
        if not self._recording:
            return

        # Add audio to input buffer
        self._input_buffer.extend(pcm)
        logger.trace(f"AudioSynchronizer: Input buffer size: {len(self._input_buffer)}")

        # BUFFER LIMIT ENFORCEMENT: Check for emergency cleanup conditions
        await self._check_buffer_limits()

        # STANDARD PROCESSING: Check if we should emit based on normal threshold
        if self._buffer_size > 0 and len(self._input_buffer) > self._buffer_size:
            await self._call_audio_handler()

    async def _handle_output_audio(
        self, processor, pcm: bytes, sample_rate: int, num_channels: int
    ):
        """Handle incoming output audio data."""
        if not self._recording:
            return

        # Add audio to output buffer
        self._output_buffer.extend(pcm)
        logger.trace(f"AudioSynchronizer: Output buffer size: {len(self._output_buffer)}")

        # BUFFER LIMIT ENFORCEMENT: Check for emergency cleanup conditions
        await self._check_buffer_limits()

        # CRITICAL FIX: Add symmetric triggering for output buffer
        # 
        # PROBLEM: Previously, only input buffer could trigger audio merging.
        # Output buffer would accumulate indefinitely, causing exponential growth.
        #
        # SOLUTION: Now both input AND output buffers can trigger merging when they
        # exceed the threshold, ensuring balanced processing and preventing accumulation.
        if self._buffer_size > 0 and len(self._output_buffer) > self._buffer_size:
            await self._call_audio_handler()

    async def _call_audio_handler(self):
        """Call the audio data event handler with merged audio."""
        if not self._has_audio() or not self._recording:
            logger.trace(
                f"AudioSynchronizer: Not calling handler - has_audio={self._has_audio()}, recording={self._recording}"
            )
            return

        # ENHANCED BUFFER PROCESSING: Prevent accumulation with better monitoring
        #
        # STEP 1: Capture current buffer states for analysis
        input_size = len(self._input_buffer)
        output_size = len(self._output_buffer)
        
        # STEP 2: Log buffer states to track accumulation patterns
        # This helps identify when buffers become unbalanced
        logger.debug(f"AudioSynchronizer: Processing buffers - input={input_size}, output={output_size}")
        
        # STEP 3: Calculate merge size using minimum of both buffers
        # WHY: We can only merge audio when we have data from BOTH user and bot
        # This is the core synchronization logic
        merge_size = min(input_size, output_size)

        # STEP 4: Ensure even byte alignment for 16-bit audio samples
        # WHY: Audio samples are 2 bytes each, odd numbers would corrupt audio
        if merge_size % 2 != 0:
            merge_size -= 1

        # STEP 5: Handle edge case where no merging is possible
        if merge_size == 0:
            # EARLY WARNING SYSTEM: Detect buffer imbalances before they become critical
            # If one buffer is much larger than normal while other is empty,
            # this indicates the accumulation problem is occurring
            max_buffer_size = max(input_size, output_size)
            if max_buffer_size > self._buffer_size * 2:  # More than 2x normal buffer size
                logger.warning(
                    f"AudioSynchronizer: Large buffer accumulation detected! "
                    f"input={input_size}, output={output_size}, threshold={self._buffer_size}"
                )
            return

        # STEP 6: Extract synchronized audio chunks from both buffers
        # We take exactly the same amount from each buffer to maintain sync
        input_chunk = bytes(self._input_buffer[:merge_size])
        output_chunk = bytes(self._output_buffer[:merge_size])

        # STEP 7: Remove processed data from buffers (sliding window approach)
        # CRITICAL: This is where residual data can remain if buffers are unequal
        # After this operation, larger buffer will have leftover data
        self._input_buffer = self._input_buffer[merge_size:]
        self._output_buffer = self._output_buffer[merge_size:]

        # STEP 8: Merge the synchronized audio chunks based on channel configuration
        # MONO (1 channel): Mix user and bot audio together (additive)
        # STEREO (2 channels): User on left channel, bot on right channel
        if self._num_channels == 1:
            merged_audio = mix_audio(input_chunk, output_chunk)  # Additive mixing
        elif self._num_channels == 2:
            merged_audio = interleave_stereo_audio(input_chunk, output_chunk)  # Stereo separation
        else:
            merged_audio = b""  # Fallback for unsupported channel configs

        # STEP 9: Emit the merged audio to registered handlers
        # This sends the synchronized audio to the InMemoryAudioBuffer for recording
        await self._emit_event(
            "on_merged_audio", merged_audio, self._sample_rate, self._num_channels
        )
        
        # STEP 10: POST-MERGE MONITORING - Check for continued accumulation
        # After processing, check if buffers are still critically large
        # This catches cases where the accumulation problem persists
        remaining_input = len(self._input_buffer)
        remaining_output = len(self._output_buffer)
        
        # RECURSIVE PROCESSING: If buffers are still very large (3x normal),
        # attempt additional processing to drain them
        if remaining_input > self._buffer_size * 3 or remaining_output > self._buffer_size * 3:
            logger.warning(
                f"AudioSynchronizer: Post-merge buffer accumulation still high! "
                f"remaining_input={remaining_input}, remaining_output={remaining_output}"
            )
            # Attempt recursive processing if both buffers still have data
            # This prevents runaway accumulation by processing multiple chunks in one cycle
            if self._has_audio():
                await self._call_audio_handler()

    def _reset_buffers(self):
        """Reset all audio buffers to empty state."""
        self._input_buffer = bytearray()
        self._output_buffer = bytearray()

    async def _emit_event(self, event_name: str, *args):
        """Emit an event to all registered handlers."""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    await handler(self, *args)
                except Exception as e:
                    logger.error(f"Error in {event_name} handler: {e}")

    def add_event_handler(self, event_name: str, handler: Callable):
        """Add an event handler for the specified event.

        Args:
            event_name: Name of the event ("on_merged_audio")
            handler: Async callable to handle the event
        """
        if event_name not in self._event_handlers:
            logger.warning(f"AudioSynchronizer: Unknown event: {event_name}")
            return

        logger.debug(f"AudioSynchronizer: Adding handler for event '{event_name}'")
        self._event_handlers[event_name].append(handler)

    def event_handler(self, event_name: str):
        """Decorator for registering event handlers.

        Example:
            ```python
            synchronizer = AudioSynchronizer()

            @synchronizer.event_handler("on_merged_audio")
            async def handle_merged_audio(sync, pcm, sample_rate, num_channels):
                # Process merged audio
                pass
            ```
        """

        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def clear_buffers(self):
        """Clear all internal audio buffers.
        
        PURPOSE: Complete buffer reset - used when starting new recording sessions
        or when forced cleanup is needed.
        """
        logger.debug(f"AudioSynchronizer: Clearing buffers - input={len(self._input_buffer)}, output={len(self._output_buffer)}")
        self._input_buffer.clear()
        self._output_buffer.clear()

    async def flush_buffers_at_turn_boundary(self):
        """TURN BOUNDARY CLEANUP: Force flush remaining audio to prevent cross-turn contamination.
        
        PROBLEM: Audio data can accumulate across conversation turns, causing:
        - Previous turn's audio bleeding into new turn's recording
        - User and bot speech appearing to overlap when they didn't
        - Exponential buffer growth across multiple turns
        
        SOLUTION: At the end of each conversation turn, this method:
        1. Processes any remaining synchronized audio
        2. Flushes unmatched audio with silence padding
        3. Completely clears buffers for clean turn separation
        
        WHEN TO CALL: This should be called when:
        - User stops speaking (end of user turn)
        - Bot finishes response (end of bot turn)
        - Call/session ends
        """
        if not self._recording:
            return
            
        input_size = len(self._input_buffer)
        output_size = len(self._output_buffer)
        
        # STEP 1: Log turn boundary flush for debugging
        if input_size > 0 or output_size > 0:
            logger.info(
                f"AudioSynchronizer: Turn boundary flush - input={input_size}, output={output_size}"
            )
            
            # STEP 2: Process any remaining synchronized audio first
            # If both buffers have data, merge what we can normally
            if self._has_audio():
                await self._call_audio_handler()
            
            # STEP 3: Handle remaining unmatched audio by padding with silence
            # This prevents audio loss while maintaining synchronization
            remaining_input = len(self._input_buffer)
            remaining_output = len(self._output_buffer)
            
            # CASE A: User audio without matching bot audio
            # Happens when user speaks but bot hasn't responded yet
            if remaining_input > 0 and remaining_output == 0:
                silence_padding = b'\x00' * remaining_input  # Create matching silence
                merged_audio = mix_audio(bytes(self._input_buffer), silence_padding)
                await self._emit_event("on_merged_audio", merged_audio, self._sample_rate, self._num_channels)
                self._input_buffer.clear()
                logger.debug("AudioSynchronizer: Flushed remaining input audio with silence padding")
                
            # CASE B: Bot audio without matching user audio  
            # Happens when bot is speaking but user is silent
            elif remaining_output > 0 and remaining_input == 0:
                silence_padding = b'\x00' * remaining_output  # Create matching silence
                merged_audio = mix_audio(silence_padding, bytes(self._output_buffer))
                await self._emit_event("on_merged_audio", merged_audio, self._sample_rate, self._num_channels)
                self._output_buffer.clear()
                logger.debug("AudioSynchronizer: Flushed remaining output audio with silence padding")
            
            # STEP 4: Final cleanup - ensure completely clean state for next turn
            self.clear_buffers()

    async def _check_buffer_limits(self):
        """EMERGENCY BUFFER MONITORING: Enforce size limits to prevent memory issues.
        
        PURPOSE: This method acts as a safety net against runaway buffer accumulation.
        It implements a tiered response system:
        
        TIER 1 - WARNING (5x normal): Log warnings to alert about growing buffers
        TIER 2 - EMERGENCY (10x normal): Force cleanup to prevent memory exhaustion
        
        WHY NEEDED: Even with fixes, edge cases or bugs could still cause accumulation.
        This provides defense-in-depth against memory issues.
        """
        input_size = len(self._input_buffer)
        output_size = len(self._output_buffer)
        
        # TIER 1: WARNING LEVEL - Early detection of accumulation
        if input_size > self._warning_buffer_size or output_size > self._warning_buffer_size:
            logger.warning(
                f"AudioSynchronizer: Buffer size warning! "
                f"input={input_size} (limit={self._warning_buffer_size}), "
                f"output={output_size} (limit={self._warning_buffer_size})"
            )
            # Attempt normal processing to reduce buffer sizes
            if self._has_audio():
                await self._call_audio_handler()
        
        # TIER 2: EMERGENCY LEVEL - Prevent memory exhaustion  
        if input_size > self._max_buffer_size or output_size > self._max_buffer_size:
            logger.error(
                f"AudioSynchronizer: EMERGENCY buffer cleanup! "
                f"input={input_size} (max={self._max_buffer_size}), "
                f"output={output_size} (max={self._max_buffer_size}). "
                f"Forcing buffer flush to prevent memory issues."
            )
            # Emergency flush with silence padding to prevent audio loss
            await self.flush_buffers_at_turn_boundary()
