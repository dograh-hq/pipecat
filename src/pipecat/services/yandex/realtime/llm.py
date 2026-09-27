#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Yandex Cloud Realtime LLM service implementation."""

import json

from websockets.asyncio.client import connect as websocket_connect

from pipecat.services.openai._constants import OPENAI_SAMPLE_RATE
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.utils.types import is_given

YANDEX_REALTIME_BASE_URL = "wss://ai.api.cloud.yandex.net/v1/realtime/openai"


def _normalize_pcm_rate(message: str) -> str:
    """Fill in the PCM sample rate Yandex omits from session events.

    Yandex's ``session.created``/``session.updated`` events echo the audio
    format as ``{"type": "audio/pcm", "rate": null}`` instead of the fixed
    24000 Hz OpenAI's wire format uses. The base service's ``PCMAudioFormat.rate``
    is ``Literal[24000]``, so the explicit ``null`` (as opposed to an omitted
    key, which would fall back to the field's default) fails validation in
    ``events.parse_server_event`` before it is ever seen.
    """
    try:
        data = json.loads(message)
    except (TypeError, ValueError):
        return message

    audio = data.get("session", {}).get("audio") if isinstance(data, dict) else None
    if not isinstance(audio, dict):
        return message

    patched = False
    for direction in ("input", "output"):
        side = audio.get(direction)
        fmt = side.get("format") if isinstance(side, dict) else None
        if isinstance(fmt, dict) and fmt.get("type") == "audio/pcm" and fmt.get("rate") is None:
            fmt["rate"] = OPENAI_SAMPLE_RATE
            patched = True

    return json.dumps(data) if patched else message


# Yandex implements a pre-GA revision of the OpenAI Realtime wire protocol,
# where a new conversation item is announced via "conversation.item.created".
# The base service's event parser only knows the GA rename,
# "conversation.item.added" (see pipecat.services.openai.realtime.events);
# the payload shape (previous_item_id + item) is unchanged, so a straight
# type rename is enough to make it validate against the base schema.
_EVENT_TYPE_ALIASES = {
    "conversation.item.created": "conversation.item.added",
}


def _normalize_event_type(message: str) -> str:
    """Rewrite Yandex's pre-GA event type names to the base service's GA names."""
    try:
        data = json.loads(message)
    except (TypeError, ValueError):
        return message

    if not isinstance(data, dict):
        return message

    alias = _EVENT_TYPE_ALIASES.get(data.get("type"))
    if alias is None:
        return message

    data["type"] = alias
    return json.dumps(data)


class _RateNormalizingWebSocket:
    """Wraps a realtime websocket connection to patch Yandex protocol quirks.

    Patches the null PCM rate in session events and renames pre-GA event
    types to their GA equivalents. Only the receive path needs patching;
    outgoing messages and lifecycle calls pass straight through to the
    underlying connection.
    """

    def __init__(self, websocket):
        self._websocket = websocket

    async def __aiter__(self):
        async for message in self._websocket:
            yield _normalize_event_type(_normalize_pcm_rate(message))

    async def send(self, message):
        await self._websocket.send(message)

    async def close(self):
        await self._websocket.close()


class YandexRealtimeLLMService(OpenAIRealtimeLLMService):
    """OpenAI Realtime wire-protocol client pointed at Yandex Cloud.

    Yandex Cloud's Realtime API (``wss://ai.api.cloud.yandex.net/v1/realtime/openai``)
    speaks the same event protocol as OpenAI's Realtime API — confirmed against a
    user-supplied example script exercising session setup, audio streaming,
    barge-in, and function calling. Besides the default connection URL and the
    auth header (``Api-Key`` instead of ``Bearer``), the only behavioral delta
    from :class:`OpenAIRealtimeLLMService` is normalizing the wire-format quirks
    above; everything else — session management, event parsing, tool-calling
    glue, and turn-taking — is inherited unmodified.
    """

    def __init__(self, **kwargs):
        """Initialize the Yandex Realtime LLM service.

        Args:
            **kwargs: Arguments passed to :class:`OpenAIRealtimeLLMService`.
                ``base_url`` defaults to Yandex Cloud's realtime endpoint.
                ``model`` (or an equivalent ``settings``/``session_properties``
                override) is required: a Yandex model resource such as
                ``"gpt://<folder-id>/yandexgpt/latest"``. Without it, this
                would silently inherit :class:`OpenAIRealtimeLLMService`'s
                OpenAI model default, which Yandex Cloud's endpoint rejects.

        Raises:
            ValueError: If no model was given.
        """
        kwargs.setdefault("base_url", YANDEX_REALTIME_BASE_URL)
        if not self._explicit_model_given(kwargs):
            raise ValueError(
                "YandexRealtimeLLMService requires an explicit Yandex model "
                "resource, e.g. model=\"gpt://<folder-id>/yandexgpt/latest\" "
                "(or settings=Settings(model=...)). Without one, this would "
                "silently inherit OpenAIRealtimeLLMService's default OpenAI "
                "model, which is not a valid Yandex Cloud model resource."
            )
        super().__init__(**kwargs)

    @staticmethod
    def _explicit_model_given(kwargs: dict) -> bool:
        """Check whether any of the model-carrying init args were set.

        Mirrors :class:`OpenAIRealtimeLLMService`'s own precedence: the
        deprecated top-level ``model``/``session_properties`` args, or the
        canonical ``settings=Settings(model=...)``.
        """
        if kwargs.get("model") is not None:
            return True

        settings = kwargs.get("settings")
        if settings is not None and is_given(settings.model) and settings.model is not None:
            return True

        session_properties = kwargs.get("session_properties")
        if session_properties is not None and session_properties.model is not None:
            return True

        return False

    async def _connect(self):
        try:
            if self._websocket:
                # Here we assume that if we have a websocket, we are connected. We
                # handle disconnections in the send/recv code paths.
                return
            self._websocket = _RateNormalizingWebSocket(
                await websocket_connect(
                    uri=self.base_url,
                    additional_headers={
                        "Authorization": f"Api-Key {self.api_key}",
                    },
                )
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None
