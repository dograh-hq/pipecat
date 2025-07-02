#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class InternalFrameSerializer(FrameSerializer):
    """Serializer for InternalTransport that filters frames between agents.

    This serializer ensures only audio frames are passed between agents,
    preventing control frames from creating infinite loops.
    """

    @property
    def type(self) -> FrameSerializerType:
        """Internal transport uses binary frames."""
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        """No setup required for internal transport."""
        pass

    async def serialize(self, frame: Frame) -> bytes | None:
        """Only serialize audio frames for transmission between agents."""
        # Only pass audio frames between agents
        if isinstance(frame, OutputAudioRawFrame):
            # Pack frame metadata along with audio data
            # Format: "AUDIO|sample_rate|num_channels|data"
            metadata = f"AUDIO|{frame.sample_rate}|{frame.num_channels}|".encode()
            return metadata + frame.audio

        # Don't pass control frames between agents
        return None

    async def deserialize(self, data: bytes) -> Frame | None:
        """Deserialize audio frames from partner agent."""
        if data.startswith(b"AUDIO"):
            try:
                # Find the end of metadata
                metadata_end = data.find(b"|", 6)  # Skip "AUDIO|"
                if metadata_end == -1:
                    return None

                metadata_end2 = data.find(b"|", metadata_end + 1)
                if metadata_end2 == -1:
                    return None

                metadata_end3 = data.find(b"|", metadata_end2 + 1)
                if metadata_end3 == -1:
                    return None

                # Extract metadata
                sample_rate = int(data[6:metadata_end])
                num_channels = int(data[metadata_end + 1 : metadata_end2])

                # Extract audio data
                audio_data = data[metadata_end3 + 1 :]

                # Convert to InputAudioRawFrame for the receiving agent
                return InputAudioRawFrame(
                    audio=audio_data, num_channels=num_channels, sample_rate=sample_rate
                )
            except Exception as e:
                logger.error(f"Failed to deserialize audio frame: {e}")
                return None

        return None
