#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
    _user: "OpenAIUserContextAggregator"
    _assistant: "OpenAIAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        return self._user

    def assistant(self) -> "OpenAIAssistantContextAggregator":
        return self._assistant


class OpenAILLMService(BaseOpenAILLMService):
    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        params: Optional[BaseOpenAILLMService.InputParams] = None,
        **kwargs,
    ):
        super().__init__(model=model, params=params, **kwargs)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Create an instance of OpenAIContextAggregatorPair from an
        OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_params (LLMUserAggregatorParams, optional): User aggregator parameters.
            assistant_params (LLMAssistantAggregatorParams, optional): User aggregator parameters.

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
    pass


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        # Track message indices before adding for potential reordering
        messages_before = len(self._context.get_messages())

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
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
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
        """Override default interruption handling to:
        1. Append a marker indicating the bot was interrupted.
        2. Remove any pending function/tool call messages associated with the
           current response.
        3. Suppress the `on_push_aggregation` event so deferred transitions are
           not flushed by the engine.
        """

        # Store current response id before any cleanup actions
        current_id = self._current_llm_response_id

        # Determine tool_call_ids that are still pending
        pending_ids = set(self._function_calls_in_progress.keys())

        # Also capture any calls already marked as CANCELLED in the context
        msgs = self._context.get_messages()

        logger.debug(f"Pending IDs: {pending_ids} {msgs}")

        for m in msgs:
            if m.get("role") == "tool" and m.get("content") == "CANCELLED":
                logger.debug(f"Got cancelled tool call id {m.get('tool_call_id')}")
                pending_ids.add(m.get("tool_call_id"))

        if pending_ids:
            new_msgs = []
            for m in msgs:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    if any(tc["id"] in pending_ids for tc in m["tool_calls"]):
                        continue  # skip this assistant tool_call stub
                if m.get("role") == "tool" and m.get("tool_call_id") in pending_ids:
                    continue  # skip corresponding tool response
                new_msgs.append(m)
            self._context.set_messages(new_msgs)
        # Refresh mapping for the current response so that only surviving
        # tool_call_ids remain. This keeps reordering logic working even after
        # we deleted some messages.
        if current_id and current_id in self._response_function_messages:
            tracked_ids = self._response_function_messages[current_id]

            remaining_ids = set()
            for m in self._context.get_messages():
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        if tc["id"] in tracked_ids:
                            remaining_ids.add(tc["id"])
                elif m.get("role") == "tool" and m.get("tool_call_id") in tracked_ids:
                    remaining_ids.add(m["tool_call_id"])

            if remaining_ids:
                self._response_function_messages[current_id] = remaining_ids
            else:
                self._response_function_messages.pop(current_id, None)

        # Determine if there is any aggregated content that warrants adding
        # a textual assistant message. If none exists, we skip adding the
        # interruption tag and any subsequent reordering since there is no
        # anchor message to attach tool call messages to.

        content_to_add = self._aggregation.strip() if self._aggregation else ""

        if content_to_add:
            content_to_add += " <<interrupted_by_user>>"

            text_msg_index = len(self._context.get_messages())
            self._context.add_message({"role": "assistant", "content": content_to_add})
            self._aggregation = ""

            # Reorder any remaining function messages tied to this response id
            if current_id and current_id in self._response_function_messages:
                await self._reorder_context_for_response(current_id, text_msg_index)

        # Cleanup response session bookkeeping regardless of whether we added
        # a message or not.
        if current_id:
            self._cleanup_response_session(current_id)

        self._function_calls_in_progress.clear()
        self._started = 0  # Reset state for current response

        # Push the updated context frame (without emitting on_push_aggregation)
        await self.push_context_frame()

        # Final cleanup similar to original behaviour
        await self.reset()

        # Do NOT call super()._handle_interruptions or on_push_aggregation –
        # simply exit so the caller's process_frame will still push the frame
        # downstream/upstream as usual.
        return
