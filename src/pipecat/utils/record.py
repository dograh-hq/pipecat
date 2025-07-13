"""Audio recording utilities for debugging and testing."""

import asyncio
import os
import tempfile
import wave
from datetime import datetime
from typing import Optional

from loguru import logger


class AudioRecorder:
    """Records incoming and outgoing PCM audio for a specified duration."""

    def __init__(
        self,
        sample_rate: int = 8000,
        num_channels: int = 1,
        record_duration: float = 10.0,
        output_dir: Optional[str] = None,
    ):
        """Initialize the audio recorder.

        Args:
            sample_rate: Sample rate of the audio
            num_channels: Number of audio channels
            record_duration: Duration to record in seconds
            output_dir: Directory to save recordings (defaults to temp dir)
        """
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._record_duration = record_duration
        self._output_dir = output_dir or tempfile.gettempdir()

        # Recording state
        self._is_recording = False
        self._start_time: Optional[float] = None

        # Audio buffers
        self._incoming_audio: list[bytes] = []
        self._outgoing_audio: list[bytes] = []

        # File paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._incoming_filename = f"incoming_audio_{timestamp}.wav"
        self._outgoing_filename = f"outgoing_audio_{timestamp}.wav"

        logger.info(f"AudioRecorder initialized - will record for {record_duration}s")

    def start_recording(self):
        """Start recording audio."""
        if self._is_recording:
            logger.warning("Recording already in progress")
            return

        self._is_recording = True
        self._start_time = asyncio.get_event_loop().time()
        self._incoming_audio.clear()
        self._outgoing_audio.clear()

        logger.info("Started audio recording")

        # Schedule automatic stop after duration
        asyncio.create_task(self._auto_stop_recording())

    async def _auto_stop_recording(self):
        """Automatically stop recording after the specified duration."""
        await asyncio.sleep(self._record_duration)
        if self._is_recording:
            await self.stop_recording()

    def record_incoming_audio(self, pcm_data: bytes) -> bool:
        """Record incoming PCM audio data.

        Args:
            pcm_data: Raw PCM audio bytes

        Returns:
            True if audio was recorded, False if recording is not active or time exceeded
        """
        if not self._is_recording:
            return False

        current_time = asyncio.get_event_loop().time()
        if current_time - self._start_time > self._record_duration:
            # Recording time exceeded, stop recording
            asyncio.create_task(self.stop_recording())
            return False

        self._incoming_audio.append(pcm_data)
        return True

    def record_outgoing_audio(self, pcm_data: bytes) -> bool:
        """Record outgoing PCM audio data.

        Args:
            pcm_data: Raw PCM audio bytes

        Returns:
            True if audio was recorded, False if recording is not active or time exceeded
        """
        if not self._is_recording:
            return False

        current_time = asyncio.get_event_loop().time()
        if current_time - self._start_time > self._record_duration:
            # Recording time exceeded, stop recording
            asyncio.create_task(self.stop_recording())
            return False

        self._outgoing_audio.append(pcm_data)
        return True

    async def stop_recording(self):
        """Stop recording and generate wave files."""
        if not self._is_recording:
            logger.warning("No recording in progress")
            return

        self._is_recording = False

        logger.info("Stopping audio recording and generating wave files")

        # Generate wave files
        incoming_path = await self._generate_wave_file(
            self._incoming_audio, self._incoming_filename
        )
        outgoing_path = await self._generate_wave_file(
            self._outgoing_audio, self._outgoing_filename
        )

        logger.info(f"Recording complete:")
        logger.info(f"  Incoming audio: {incoming_path}")
        logger.info(f"  Outgoing audio: {outgoing_path}")

        return incoming_path, outgoing_path

    async def _generate_wave_file(self, audio_data: list[bytes], filename: str) -> str:
        """Generate a wave file from recorded audio data.

        Args:
            audio_data: List of PCM audio byte chunks
            filename: Name of the output file

        Returns:
            Full path to the generated wave file
        """
        output_path = os.path.join(self._output_dir, filename)

        try:
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(self._num_channels)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(self._sample_rate)

                # Write all audio data
                for chunk in audio_data:
                    wf.writeframes(chunk)

            logger.info(f"Generated wave file: {output_path} ({len(audio_data)} chunks)")

        except Exception as e:
            logger.error(f"Failed to generate wave file {output_path}: {e}")
            raise

        return output_path

    @property
    def is_recording(self) -> bool:
        """Check if recording is currently active."""
        return self._is_recording

    @property
    def recording_time_remaining(self) -> float:
        """Get remaining recording time in seconds."""
        if not self._is_recording or not self._start_time:
            return 0.0

        elapsed = asyncio.get_event_loop().time() - self._start_time
        return max(0.0, self._record_duration - elapsed)
