#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy with provisional VAD pause before transcript confirmation."""

import asyncio

from loguru import logger
from pipecat.frames.frames import (
    BotOutputAudioPauseFrame,
    BotOutputAudioResumeFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class ProvisionalVADUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Pause bot audio on VAD, but start the user turn only after transcription.

    While the bot is speaking, a VAD speech-start signal pauses output audio
    without broadcasting an interruption. If an interim or final transcript
    arrives before ``pause_secs`` elapses, the strategy starts the user turn and
    the user aggregator emits the normal ``InterruptionFrame``. If no transcript
    arrives in that window, output audio resumes and further transcript-based
    starts are ignored until the bot stops speaking.

    When the bot is not speaking, VAD and transcript frames start the user turn
    immediately, matching the normal fast-turn behavior.
    """

    def __init__(self, *, pause_secs: float = 1.0, use_interim: bool = True, **kwargs):
        """Initialize the provisional VAD start strategy.

        Args:
            pause_secs: Seconds to pause bot output while waiting for a
                transcript confirmation.
            use_interim: Whether interim transcriptions confirm the user turn.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._pause_secs = pause_secs
        self._use_interim = use_interim

        self._bot_speaking = False
        self._provisional_pause_active = False
        self._locked_out_until_bot_stops = False
        self._resume_task: asyncio.Task | None = None

    async def cleanup(self):
        """Clean up strategy state."""
        await super().cleanup()
        await self._cancel_resume_task()

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        await self._cancel_resume_task()
        self._provisional_pause_active = False
        self._locked_out_until_bot_stops = False
        self._bot_speaking = False

    def _state_str(self) -> str:
        """Compact snapshot of the strategy's decision state (diagnostic)."""
        return (
            f"bot_speaking={self._bot_speaking} "
            f"pause_active={self._provisional_pause_active} "
            f"locked_out={self._locked_out_until_bot_stops}"
        )

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to detect user turn start."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            logger.debug(f"[provisional-vad] {self} BotStartedSpeaking → {self._state_str()}")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug(
                f"[provisional-vad] {self} BotStoppedSpeaking (before) → {self._state_str()}"
            )
            await self._handle_bot_stopped_speaking()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            logger.debug(f"[provisional-vad] {self} VADUserStartedSpeaking → {self._state_str()}")
            return await self._handle_vad_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Keep the fixed provisional window measured from speech start.
            logger.debug(f"[provisional-vad] {self} VADUserStoppedSpeaking → {self._state_str()}")
        elif isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            logger.debug(f"[provisional-vad] {self} InterimTranscription → {self._state_str()}")
            return await self._handle_transcription()
        elif isinstance(frame, TranscriptionFrame):
            logger.debug(f"[provisional-vad] {self} Transcription → {self._state_str()}")
            return await self._handle_transcription()

        return ProcessFrameResult.CONTINUE

    async def _handle_vad_started(self) -> ProcessFrameResult:
        if not self._bot_speaking:
            logger.debug(
                f"[provisional-vad] {self} vad_started: bot NOT speaking "
                "→ trigger user turn started immediately"
            )
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP

        if self._locked_out_until_bot_stops:
            logger.debug(
                f"[provisional-vad] {self} vad_started: locked out until bot stops → ignore"
            )
            return ProcessFrameResult.CONTINUE

        if not self._provisional_pause_active:
            self._provisional_pause_active = True
            logger.debug(
                f"[provisional-vad] {self} vad_started: arming provisional pause "
                "→ pushing BotOutputAudioPauseFrame downstream"
            )
            await self.push_frame(BotOutputAudioPauseFrame(), FrameDirection.DOWNSTREAM)
            self._resume_task = self.create_task(
                self._resume_after_timeout(), name="resume_after_timeout"
            )
        else:
            logger.debug(
                f"[provisional-vad] {self} vad_started: provisional pause already active → no-op"
            )

        return ProcessFrameResult.CONTINUE

    async def _handle_transcription(self) -> ProcessFrameResult:
        if self._locked_out_until_bot_stops and self._bot_speaking:
            logger.debug(
                f"[provisional-vad] {self} transcription: locked out + bot speaking "
                "→ reset aggregation (ignore transcript)"
            )
            await self.trigger_reset_aggregation()
            return ProcessFrameResult.CONTINUE

        if self._provisional_pause_active:
            logger.debug(
                f"[provisional-vad] {self} transcription: confirms active provisional pause "
                "→ trigger user turn started"
            )
            self._provisional_pause_active = False
            await self._cancel_resume_task()
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP

        # Fall-through: no pause was armed (and not locked out). This fires
        # the user turn directly even if the bot is still speaking, because
        # the pause is only ever armed from _handle_vad_started.
        logger.debug(
            f"[provisional-vad] {self} transcription: FALL-THROUGH → trigger user turn started "
            f"({self._state_str()})"
        )
        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP

    async def _handle_bot_stopped_speaking(self):
        if self._provisional_pause_active:
            logger.debug(
                f"[provisional-vad] {self} bot stopped while provisional pause active "
                "→ resume output audio"
            )
            await self._resume_output_audio()

        self._bot_speaking = False
        self._locked_out_until_bot_stops = False
        self._provisional_pause_active = False
        await self._cancel_resume_task()

    async def _resume_after_timeout(self):
        try:
            await asyncio.sleep(self._pause_secs)
            if not self._provisional_pause_active:
                return
            logger.debug(
                f"[provisional-vad] {self} provisional pause timed out after "
                f"{self._pause_secs}s with no transcript → resume + lock out until bot stops"
            )
            self._provisional_pause_active = False
            self._locked_out_until_bot_stops = True
            await self._resume_output_audio()
        except asyncio.CancelledError:
            return
        finally:
            current_task = asyncio.current_task()
            if self._resume_task is current_task:
                self._resume_task = None

    async def _resume_output_audio(self):
        await self.push_frame(BotOutputAudioResumeFrame(), FrameDirection.DOWNSTREAM)

    async def _cancel_resume_task(self):
        if self._resume_task and not self._resume_task.done():
            await self.cancel_task(self._resume_task)
        self._resume_task = None
