#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Awaitable, Callable

from pipecat.frames.frames import (
    BotSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BotIdleProcessor(FrameProcessor):
    """Monitors bot inactivity after user stops speaking and triggers callback if bot doesn't respond.

    Starts the timer only when the user stops speaking. Resets if the bot starts speaking.
    Calls the callback only once per idle event.

    Args:
        callback: Function to call when bot is idle after user stops speaking.
            Signature: callback(processor) -> None
        timeout: Seconds to wait after user stops speaking before considering bot idle
        **kwargs: Additional arguments passed to FrameProcessor

    Example:
        async def handle_bot_idle(processor: "BotIdleProcessor") -> None:
            await prompt_bot_to_respond()

        processor = BotIdleProcessor(
            callback=handle_bot_idle,
            timeout=3.0
        )
    """

    def __init__(
        self,
        *,
        callback: Callable[["BotIdleProcessor"], Awaitable[None]],
        timeout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._callback = callback
        self._timeout = timeout
        self._conversation_started = False
        self._idle_task = None
        self._waiting_for_bot = False

    async def _stop_idle_timer(self) -> None:
        """Stops the idle timer if it's running."""
        if self._idle_task:
            await self.cancel_task(self._idle_task)
            self._idle_task = None
        self._waiting_for_bot = False

    def _start_idle_timer(self) -> None:
        """Starts the idle timer to wait for bot response."""
        if not self._idle_task and not self._waiting_for_bot:
            self._waiting_for_bot = True
            self._idle_task = self.create_task(self._idle_timer())

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes incoming frames and manages bot idle monitoring.

        Args:
            frame: The frame to process
            direction: Direction of the frame flow
        """
        await super().process_frame(frame, direction)

        # Check for end frames before processing
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop_idle_timer()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

        # Start monitoring on first conversation activity
        if not self._conversation_started and isinstance(
            frame, (UserStartedSpeakingFrame, BotSpeakingFrame)
        ):
            self._conversation_started = True

        # Only process these events if conversation has started
        if self._conversation_started:
            if isinstance(frame, UserStoppedSpeakingFrame):
                # User stopped speaking - start waiting for bot response
                self._start_idle_timer()
            elif isinstance(frame, BotSpeakingFrame):
                # Bot started speaking - stop the idle timer
                await self._stop_idle_timer()
            elif isinstance(frame, UserStartedSpeakingFrame):
                # User started speaking again - stop waiting for bot
                await self._stop_idle_timer()

    async def cleanup(self) -> None:
        """Cleans up resources when processor is shutting down."""
        await super().cleanup()
        await self._stop_idle_timer()

    async def _idle_timer(self) -> None:
        """Timer that waits for bot response after user stops speaking."""
        try:
            await asyncio.sleep(self._timeout)
            # Timeout occurred - bot is idle, call callback once
            if self._waiting_for_bot:
                await self._callback(self)
        except asyncio.CancelledError:
            # Timer was cancelled (bot started speaking or user started speaking again)
            pass
        finally:
            self._waiting_for_bot = False
            self._idle_task = None
