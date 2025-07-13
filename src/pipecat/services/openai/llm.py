#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM service implementation with context aggregators."""

import json
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    StartInterruptionFrame,
    UserImageRawFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService


@dataclass
class OpenAIContextAggregatorPair:
    """Pair of OpenAI context aggregators for user and assistant messages.

    Parameters:
        _user: User context aggregator for processing user messages.
        _assistant: Assistant context aggregator for processing assistant messages.
    """

    _user: "OpenAIUserContextAggregator"
    _assistant: "OpenAIAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> "OpenAIAssistantContextAggregator":
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class OpenAILLMService(BaseOpenAILLMService):
    """OpenAI LLM service implementation.

    Provides a complete OpenAI LLM service with context aggregation support.
    Uses the BaseOpenAILLMService for core functionality and adds OpenAI-specific
    context aggregator creation.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        params: Optional[BaseOpenAILLMService.InputParams] = None,
        **kwargs,
    ):
        """Initialize OpenAI LLM service.

        Args:
            model: The OpenAI model name to use. Defaults to "gpt-4.1".
            params: Input parameters for model configuration.
            **kwargs: Additional arguments passed to the parent BaseOpenAILLMService.
        """
        super().__init__(model=model, params=params, **kwargs)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Create OpenAI-specific context aggregators.

        Creates a pair of context aggregators optimized for OpenAI's message format,
        including support for function calls, tool usage, and image handling.

        Args:
            context: The LLM context to create aggregators for.
            user_params: Parameters for user message aggregation.
            assistant_params: Parameters for assistant message aggregation.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())
        user = OpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)


class OpenAIUserContextAggregator(LLMUserContextAggregator):
    """OpenAI-specific user context aggregator.

    Handles aggregation of user messages for OpenAI LLM services.
    Inherits all functionality from the base LLMUserContextAggregator.
    """

    pass


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    """OpenAI-specific assistant context aggregator.

    Handles aggregation of assistant messages for OpenAI LLM services,
    with specialized support for OpenAI's function calling format,
    tool usage tracking, and image message handling.
    """

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Handle a function call in progress.

        Adds the function call to the context with an IN_PROGRESS status
        to track ongoing function execution.

        Args:
            frame: Frame containing function call progress information.
        """
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments),
                        },
                        "type": "function",
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": frame.tool_call_id,
            }
        )

        # Track the tool_call_id so we can later reorder the messages related
        # to this tool call. Using stable IDs rather than indices makes the
        # tracking resilient to subsequent insertions/deletions.
        if self._current_llm_response_id:
            self._response_function_messages.setdefault(self._current_llm_response_id, set()).add(
                frame.tool_call_id
            )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle the result of a function call.

        Updates the context with the function call result, replacing any
        previous IN_PROGRESS status.

        Args:
            frame: Frame containing the function call result.
        """
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Handle a cancelled function call.

        Updates the context to mark the function call as cancelled.

        Args:
            frame: Frame containing the function call cancellation information.
        """
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in self._context.messages:
            if (
                message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        """Handle a user image frame from a function call request.

        Marks the associated function call as completed and adds the image
        to the context for processing.

        Args:
            frame: Frame containing the user image and request context.
        """
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )

    # ------------------------------------------------------------------
    # Interruption handling
    # ------------------------------------------------------------------

    async def _handle_interruptions(self, frame: StartInterruptionFrame):
        """Handle user interruption while minimizing iterations over the context.

        The new implementation performs **two** passes instead of three over the
        message list by:
        1. Collecting *CANCELLED* tool call ids in a first light pass.
        2. Building the filtered message list **and** computing the remaining
           tracked ids in a single second pass.
        """

        # Lets not reorder if interrupted, so that the last thing the model
        # sees is assistant message getting interrupted
        SHOULD_REORDER = False

        current_llm_response_id = self._current_llm_response_id
        messages = self._context.get_messages()

        # Pass 1 – gather ids for calls that were either still in progress or
        # already cancelled.
        cancelled_ids = {
            m.get("tool_call_id")
            for m in messages
            if m.get("role") == "tool" and m.get("content") == "CANCELLED"
        }
        pending_ids: set[str] = set(self._function_calls_in_progress.keys()).union(cancelled_ids)

        # Prepare variables required for Pass 2 (executed regardless of whether
        # *pending_ids* is empty so we can still compute *remaining_ids*).
        tracked_ids: set[str] = (
            self._response_function_messages.get(current_llm_response_id, set())
            if current_llm_response_id
            else set()
        )
        remaining_ids: set[str] = set()

        # Pass 2 – rebuild message list while collecting *remaining_ids* for the
        # current response in a single sweep.
        if pending_ids:
            new_msgs: list = []
            for m in messages:
                role = m.get("role")

                # Skip assistant tool_call stubs that reference a pending id.
                if role == "assistant" and m.get("tool_calls"):
                    tool_calls = m["tool_calls"]
                    if any(tc["id"] in pending_ids for tc in tool_calls):
                        continue
                    # Track surviving ids for later re-ordering
                    for tc in tool_calls:
                        if tc["id"] in tracked_ids:
                            remaining_ids.add(tc["id"])
                    new_msgs.append(m)
                    continue

                # Skip tool responses that reference a pending id.
                if role == "tool" and m.get("tool_call_id") in pending_ids:
                    continue

                # Track surviving ids appearing in tool responses.
                if role == "tool" and m.get("tool_call_id") in tracked_ids:
                    remaining_ids.add(m.get("tool_call_id"))

                new_msgs.append(m)

            self._context.set_messages(new_msgs)

            # Update mapping for the current response so that only ids still
            # present in the context remain tracked.
            if (
                current_llm_response_id
                and current_llm_response_id in self._response_function_messages
            ):
                if remaining_ids:
                    self._response_function_messages[current_llm_response_id] = remaining_ids
                else:
                    self._response_function_messages.pop(current_llm_response_id, None)

        # ------------------------------------------------------------------
        # Add interruption marker & eventual reordering of remaining messages
        # ------------------------------------------------------------------
        content_to_add = self._aggregation.strip() if self._aggregation else ""
        if content_to_add:
            content_to_add += " <<interrupted_by_user>>"
            text_msg_index = len(self._context.get_messages())
            self._context.add_message({"role": "assistant", "content": content_to_add})
            self._aggregation = ""

            if (
                current_llm_response_id
                and current_llm_response_id in self._response_function_messages
                and SHOULD_REORDER
            ):
                await self._reorder_context_for_response(current_llm_response_id, text_msg_index)

        # ------------------------------------------------------------------
        # Final cleanup (mirrors previous behaviour)
        # ------------------------------------------------------------------
        if current_llm_response_id:
            self._cleanup_response_session(current_llm_response_id)

        self._function_calls_in_progress.clear()
        self._started = 0  # Reset state for current response

        await self.push_context_frame()
        await self.reset()
        return
