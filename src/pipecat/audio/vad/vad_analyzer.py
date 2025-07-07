#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import calculate_audio_volume, exp_smoothing

VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.8
VAD_MIN_VOLUME = 0.6


class VADState(Enum):
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADParams(BaseModel):
    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME


class VADAnalyzer(ABC):
    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None):
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._params = params or VADParams()
        self._num_channels = 1

        self._vad_buffer = b""

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def params(self) -> VADParams:
        return self._params

    @abstractmethod
    def num_frames_required(self) -> int:
        pass

    @abstractmethod
    def voice_confidence(self, buffer) -> float:
        pass

    def set_sample_rate(self, sample_rate: int):
        self._sample_rate = self._init_sample_rate or sample_rate
        self.set_params(self._params)

    def set_params(self, params: VADParams):
        logger.debug(f"Setting VAD params to: {params}")
        self._params = params
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET
        
        # logger.debug(f"VAD config: frames={self._vad_frames}, bytes_required={self._vad_frames_num_bytes}, "
        #             f"start_frames={self._vad_start_frames}, stop_frames={self._vad_stop_frames}, "
        #             f"sample_rate={self.sample_rate}")
        
        # Check the actual default parameter values
        # logger.debug(f"VAD thresholds: confidence={self._params.confidence}, min_volume={self._params.min_volume}")

    def _get_smoothed_volume(self, audio: bytes) -> float:
        volume = calculate_audio_volume(audio, self.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    def analyze_audio(self, buffer) -> VADState:
        self._vad_buffer += buffer
        # logger.debug(f"VADAnalyzer: Buffer size: {len(buffer)}, Total buffered: {len(self._vad_buffer)}, Required: {self._vad_frames_num_bytes}")

        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        audio_frames = self._vad_buffer[:num_required_bytes]
        self._vad_buffer = self._vad_buffer[num_required_bytes:]

        confidence = self.voice_confidence(audio_frames)

        volume = self._get_smoothed_volume(audio_frames)
        self._prev_volume = volume
        
        # Convert numpy array to float if needed
        confidence_value = float(confidence) if hasattr(confidence, '__float__') else confidence
        # logger.debug(f"VADAnalyzer: Confidence: {confidence_value:.4f} (threshold: {self._params.confidence}), Volume: {volume:.4f} (min: {self._params.min_volume})")

        speaking = confidence_value >= self._params.confidence and volume >= self._params.min_volume

        if speaking:
            match self._vad_state:
                case VADState.QUIET:
                    self._vad_state = VADState.STARTING
                    self._vad_starting_count = 1
                case VADState.STARTING:
                    self._vad_starting_count += 1
                case VADState.STOPPING:
                    self._vad_state = VADState.SPEAKING
                    self._vad_stopping_count = 0
        else:
            match self._vad_state:
                case VADState.STARTING:
                    self._vad_state = VADState.QUIET
                    self._vad_starting_count = 0
                case VADState.SPEAKING:
                    self._vad_state = VADState.STOPPING
                    self._vad_stopping_count = 1
                case VADState.STOPPING:
                    self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            # logger.debug(f"VADAnalyzer: State transition STARTING -> SPEAKING")
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            # logger.debug(f"VADAnalyzer: State transition STOPPING -> QUIET")
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        # if speaking:
        #     logger.debug(f"VADAnalyzer: Speaking detected, state: {self._vad_state}")
        
        return self._vad_state
