#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    EndOfTurnState,
)
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    BotInterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    EndFrame,
    FilterUpdateSettingsFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    MetricsFrame,
    StartFrame,
    StartInterruptionFrame,
    StopFrame,
    StopInterruptionFrame,
    SystemFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADParamsUpdateFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams

AUDIO_INPUT_TIMEOUT_SECS = 0.5


class BaseInputTransport(FrameProcessor):
    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)

        self._params = params

        # Input sample rate. It will be initialized on StartFrame.
        self._sample_rate = 0

        # Track bot speaking state for interruption logic
        self._bot_speaking = False

        # Track user speaking state for interruption logic
        self._user_speaking = False

        # We read audio from a single queue one at a time and we then run VAD in
        # a thread. Therefore, only one thread should be necessary.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Till we figure out how to solve for task cancelled error, lets keep a flag
        # to determine whether we have received EndFrame or is it client disconnect
        self._received_end_frame = None

        # Task to process incoming audio (VAD) and push audio frames downstream
        # if passthrough is enabled.
        self._audio_task = None

        # If the transport is stopped with `StopFrame` we might still be
        # receiving frames from the transport but we really don't want to push
        # them downstream until we get another `StartFrame`.
        self._paused = False

        if self._params.vad_enabled:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'vad_enabled' is deprecated, use 'audio_in_enabled' and 'vad_analyzer' instead.",
                    DeprecationWarning,
                )
            self._params.audio_in_enabled = True

        if self._params.vad_audio_passthrough:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'vad_audio_passthrough' is deprecated, audio passthrough is now always enabled. Use 'audio_in_passthrough' to disable.",
                    DeprecationWarning,
                )
            self._params.audio_in_passthrough = True

        if self._params.camera_in_enabled:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameters 'camera_*' are deprecated, use 'video_*' instead.",
                    DeprecationWarning,
                )
            self._params.video_in_enabled = self._params.camera_in_enabled
            self._params.video_out_enabled = self._params.camera_out_enabled
            self._params.video_out_is_live = self._params.camera_out_is_live
            self._params.video_out_width = self._params.camera_out_width
            self._params.video_out_height = self._params.camera_out_height
            self._params.video_out_bitrate = self._params.camera_out_bitrate
            self._params.video_out_framerate = self._params.camera_out_framerate
            self._params.video_out_color_format = self._params.camera_out_color_format

    def enable_audio_in_stream_on_start(self, enabled: bool) -> None:
        logger.debug(f"Enabling audio on start. {enabled}")
        self._params.audio_in_stream_on_start = enabled

    async def start_audio_in_streaming(self):
        pass

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def vad_analyzer(self) -> Optional[VADAnalyzer]:
        return self._params.vad_analyzer

    @property
    def turn_analyzer(self) -> Optional[BaseTurnAnalyzer]:
        return self._params.turn_analyzer

    async def start(self, frame: StartFrame):
        self._paused = False
        self._user_speaking = False

        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate

        logger.info(
            f"BaseInputTransport: Setting sample rate to {self._sample_rate}Hz "
            f"(params={self._params.audio_in_sample_rate}, frame={frame.audio_in_sample_rate})"
        )

        # Configure VAD analyzer.
        if self._params.vad_analyzer:
            logger.debug(f"BaseInputTransport: Configuring VAD with {self._sample_rate}Hz")
            self._params.vad_analyzer.set_sample_rate(self._sample_rate)

        # Configure End of turn analyzer.
        if self._params.turn_analyzer:
            self._params.turn_analyzer.set_sample_rate(self._sample_rate)

        # Start audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.start(self._sample_rate)

    async def stop(self, frame: EndFrame):
        # Cancel and wait for the audio input task to finish.
        logger.debug(f"Stopping audio input task in BaseInputTransport")
        await self._cancel_audio_task()
        # Stop audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.stop()
        logger.debug(f"Audio input task stopped in BaseInputTransport")

    async def pause(self, frame: StopFrame):
        self._paused = True
        # Cancel task so we clear the queue
        await self._cancel_audio_task()
        # Retart the task
        self._create_audio_task()

    async def cancel(self, frame: CancelFrame):
        # Cancel and wait for the audio input task to finish.
        await self._cancel_audio_task()

    async def set_transport_ready(self, frame: StartFrame):
        """To be called when the transport is ready to stream."""
        # Create audio input queue and task if needed.
        self._create_audio_task()

    async def push_video_frame(self, frame: InputImageRawFrame):
        if self._params.video_in_enabled and not self._paused:
            await self.push_frame(frame)

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        if self._params.audio_in_enabled and not self._paused:
            await self._audio_in_queue.put(frame)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotInterruptionFrame):
            await self._handle_bot_interruption(frame)
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EmulateUserStartedSpeakingFrame):
            # If we've already received an EndFrame (i.e. the pipeline is shutting down),
            # ignore any late emulated VAD events so we don't spuriously restart the
            # interruption logic.
            if self._received_end_frame:
                logger.debug("Ignoring EmulateUserStartedSpeakingFrame received after EndFrame")
            else:
                logger.debug("Emulating user started speaking")
                await self._handle_user_interruption(UserStartedSpeakingFrame(emulated=True))
        elif isinstance(frame, EmulateUserStoppedSpeakingFrame):
            logger.debug("Emulating user stopped speaking")
            await self._handle_user_interruption(UserStoppedSpeakingFrame(emulated=True))
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, EndFrame):
            logger.debug(f"Received EndFrame, stopping {self}")
            self._received_end_frame = frame

            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self.stop(frame)
        elif isinstance(frame, StopFrame):
            await self.push_frame(frame, direction)
            await self.pause(frame)
        elif isinstance(frame, VADParamsUpdateFrame):
            if self.vad_analyzer:
                self.vad_analyzer.set_params(frame.params)
        elif isinstance(frame, FilterUpdateSettingsFrame) and self._params.audio_in_filter:
            await self._params.audio_in_filter.process_frame(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    #
    # Handle interruptions
    #

    async def _handle_bot_interruption(self, frame: BotInterruptionFrame):
        logger.debug("Bot interruption")
        if self.interruptions_allowed:
            await self._start_interruption()
            await self.push_frame(StartInterruptionFrame())

    async def _handle_user_interruption(self, frame: Frame):
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking")
            self._user_speaking = True
            await self.push_frame(frame)

            # Only push StartInterruptionFrame if:
            # 1. No interruption config is set, OR
            # 2. Interruption config is set but bot is not speaking
            should_push_immediate_interruption = (
                not self.interruption_strategies or not self._bot_speaking
            )

            # Make sure we notify about interruptions quickly out-of-band.
            if should_push_immediate_interruption and self.interruptions_allowed:
                await self._start_interruption()
                # Push an out-of-band frame (i.e. not using the ordered push
                # frame task) to stop everything, specially at the output
                # transport.
                await self.push_frame(StartInterruptionFrame())
            elif self.interruption_strategies and self._bot_speaking:
                logger.debug(
                    "User started speaking while bot is speaking with interruption config - "
                    "deferring interruption to aggregator"
                )
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking")
            self._user_speaking = False
            await self.push_frame(frame)
            if self.interruptions_allowed:
                await self._stop_interruption()
                await self.push_frame(StopInterruptionFrame())

    #
    # Handle bot speaking state
    #

    async def _handle_bot_started_speaking(self, frame: BotStartedSpeakingFrame):
        self._bot_speaking = True

    async def _handle_bot_stopped_speaking(self, frame: BotStoppedSpeakingFrame):
        self._bot_speaking = False

    #
    # Audio input
    #

    def _create_audio_task(self):
        if not self._audio_task and self._params.audio_in_enabled:
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.create_task(self._audio_task_handler())

    async def _cancel_audio_task(self):
        logger.debug(f"Cancelling audio task in BaseInputTransport")
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None
        logger.debug(f"Audio task cancelled in BaseInputTransport")

    async def _vad_analyze(self, audio_frame: InputAudioRawFrame) -> VADState:
        state = VADState.QUIET
        if self.vad_analyzer:
            state = await self.get_event_loop().run_in_executor(
                self._executor, self.vad_analyzer.analyze_audio, audio_frame.audio
            )
        return state

    async def _handle_vad(self, audio_frame: InputAudioRawFrame, vad_state: VADState):
        new_vad_state = await self._vad_analyze(audio_frame)
        if (
            new_vad_state != vad_state
            and new_vad_state != VADState.STARTING
            and new_vad_state != VADState.STOPPING
        ):
            frame = None
            # If the turn analyser is enabled, this will prevent:
            # - Creating the UserStoppedSpeakingFrame
            # - Creating the UserStartedSpeakingFrame multiple times
            can_create_user_frames = (
                self._params.turn_analyzer is None
                or not self._params.turn_analyzer.speech_triggered
            )
            if new_vad_state == VADState.SPEAKING:
                await self.push_frame(VADUserStartedSpeakingFrame())
                if can_create_user_frames:
                    frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                await self.push_frame(VADUserStoppedSpeakingFrame())
                if can_create_user_frames:
                    frame = UserStoppedSpeakingFrame()

            if frame:
                await self._handle_user_interruption(frame)

            vad_state = new_vad_state
        return vad_state

    async def _handle_end_of_turn(self):
        if self.turn_analyzer:
            state, prediction = await self.turn_analyzer.analyze_end_of_turn()
            await self._handle_prediction_result(prediction)
            await self._handle_end_of_turn_complete(state)

    async def _handle_end_of_turn_complete(self, state: EndOfTurnState):
        if state == EndOfTurnState.COMPLETE:
            await self._handle_user_interruption(UserStoppedSpeakingFrame())

    async def _run_turn_analyzer(
        self, frame: InputAudioRawFrame, vad_state: VADState, previous_vad_state: VADState
    ):
        is_speech = vad_state == VADState.SPEAKING or vad_state == VADState.STARTING
        # If silence exceeds threshold, we are going to receive EndOfTurnState.COMPLETE
        end_of_turn_state = self._params.turn_analyzer.append_audio(frame.audio, is_speech)
        if end_of_turn_state == EndOfTurnState.COMPLETE:
            # Check if the turn analyzer has timeout metrics to emit
            if (
                hasattr(self._params.turn_analyzer, "_timeout_metrics")
                and self._params.turn_analyzer._timeout_metrics
            ):
                logger.debug("Sending event _timeout_metrics")
                await self._handle_prediction_result(self._params.turn_analyzer._timeout_metrics)
                self._params.turn_analyzer._timeout_metrics = None
            await self._handle_end_of_turn_complete(end_of_turn_state)
        # Otherwise we are going to trigger to check if the turn is completed based on the VAD
        elif vad_state == VADState.QUIET and vad_state != previous_vad_state:
            await self._handle_end_of_turn()

    async def _audio_task_handler(self):
        vad_state: VADState = VADState.QUIET
        while True:
            try:
                frame: InputAudioRawFrame = await asyncio.wait_for(
                    self._audio_in_queue.get(), timeout=AUDIO_INPUT_TIMEOUT_SECS
                )

                # If an audio filter is available, run it before VAD.
                if self._params.audio_in_filter:
                    frame.audio = await self._params.audio_in_filter.filter(frame.audio)

                # Check VAD and push event if necessary. We just care about
                # changes from QUIET to SPEAKING and vice versa.
                previous_vad_state = vad_state
                if self._params.vad_analyzer:
                    vad_state = await self._handle_vad(frame, vad_state)

                if self._params.turn_analyzer:
                    await self._run_turn_analyzer(frame, vad_state, previous_vad_state)

                # Push audio downstream if passthrough is set.
                if self._params.audio_in_passthrough:
                    await self.push_frame(frame)

                self._audio_in_queue.task_done()
            except asyncio.TimeoutError:
                if self._user_speaking:
                    logger.warning(
                        "Forcing user stopped speaking due to timeout receiving audio frame!"
                    )
                    vad_state = VADState.QUIET
                    if self._params.turn_analyzer:
                        self._params.turn_analyzer.clear()
                    await self._handle_user_interruption(UserStoppedSpeakingFrame())
            finally:
                self.reset_watchdog()

    async def _handle_prediction_result(self, result: MetricsData):
        """Handle a prediction result event from the turn analyzer.

        Args:
            result: The prediction result MetricsData.
        """
        await self.push_frame(MetricsFrame(data=[result]))
