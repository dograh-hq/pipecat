#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
import soxr
from loguru import logger

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler


class SOXRAudioResampler(BaseAudioResampler):
    """Audio resampler implementation using the SoX resampler library."""

    def __init__(self, **kwargs):
        pass

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        if in_rate == out_rate:
            return audio

        # Ensure audio length is even (for 16-bit samples)
        if len(audio) % 2 != 0:
            logger.debug(
                f"Audio buffer has odd length: {len(audio)}, padding with zero byte in_rate={in_rate}, out_rate={out_rate}"
            )
            # Pad with a zero byte to make it even
            audio = audio + b"\x00"

        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = soxr.resample(audio_data, in_rate, out_rate, quality="VHQ")
        result = resampled_audio.astype(np.int16).tobytes()
        return result
