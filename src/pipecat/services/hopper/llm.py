#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hopper LLM service implementation.

Hopper (https://withhopper.com) serves open-weight models such as Gemma 4 31B
behind an OpenAI-compatible chat-completions API at
``https://api.withhopper.com/v1``.
"""

from dataclasses import dataclass

from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.http import connection_limits

HOPPER_BASE_URL = "https://api.withhopper.com/v1"
HOPPER_DEFAULT_MODEL = "gemma-4-31b"


def _http2_available() -> bool:
    """Report whether the HTTP client can negotiate HTTP/2.

    Both httpx families need the ``h2`` package for HTTP/2, which the
    ``pipecat-ai[hopper]`` extra installs.
    """
    try:
        import h2  # noqa: F401
    except ImportError:
        logger.debug(
            "HopperLLMService: h2 is not installed, using HTTP/1.1. "
            "Install pipecat-ai[hopper] to enable HTTP/2."
        )
        return False
    return True


@dataclass
class HopperLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for HopperLLMService."""

    pass


class HopperLLMService(OpenAILLMService):
    """OpenAI-compatible LLM service for Hopper's hosted open models.

    Hopper accepts the ``system``, ``user``, ``assistant`` and ``tool`` roles,
    so ``developer`` messages are converted to ``user`` before sending.

    The HTTP client negotiates HTTP/2 when the ``h2`` package is installed
    and keeps idle connections open across the gaps between conversation
    turns, so each turn reuses a warm connection to Hopper.
    """

    Settings = HopperLLMSettings
    _settings: Settings
    supports_developer_role = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = HOPPER_BASE_URL,
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Hopper LLM service.

        Args:
            api_key: Hopper API key (``sk_hopper_...``).
            base_url: Hopper OpenAI-compatible API base URL.
            model: Model identifier to use. Defaults to ``gemma-4-31b``.

                .. deprecated:: 0.0.105
                    Use ``settings=HopperLLMService.Settings(model=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside
                deprecated parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model=HOPPER_DEFAULT_MODEL)

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create an AsyncOpenAI client for Hopper.

        Args:
            api_key: Hopper API key.
            base_url: Hopper API base URL.
            organization: Unused; accepted for interface compatibility.
            project: Unused; accepted for interface compatibility.
            default_headers: Additional HTTP headers.
            **kwargs: Additional client configuration arguments.

        Returns:
            Configured AsyncOpenAI client instance.
        """
        logger.debug(f"Creating Hopper client with api {base_url}")
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            http_client=DefaultAsyncHttpxClient(
                http2=_http2_available(),
                limits=connection_limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                ),
            ),
            default_headers=default_headers,
        )
