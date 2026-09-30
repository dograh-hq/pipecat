#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import patch

from pipecat.services.hopper import llm as hopper_llm
from pipecat.services.hopper.llm import HopperLLMService, HopperLLMSettings


def test_hopper_llm_defaults():
    with patch.object(HopperLLMService, "create_client", return_value=object()) as create_client:
        service = HopperLLMService(api_key="sk_hopper_test")

    assert service.supports_developer_role is False
    assert service._settings.model == "gemma-4-31b"
    kwargs = create_client.call_args.kwargs
    assert kwargs["api_key"] == "sk_hopper_test"
    assert kwargs["base_url"] == "https://api.withhopper.com/v1"


def test_hopper_llm_settings_and_base_url_override():
    with patch.object(HopperLLMService, "create_client", return_value=object()) as create_client:
        service = HopperLLMService(
            api_key="sk_hopper_test",
            base_url="https://hopper.example.test/v1",
            settings=HopperLLMSettings(model="gemma-4-31b-custom", temperature=0.2),
        )

    assert service._settings.model == "gemma-4-31b-custom"
    assert service._settings.temperature == 0.2
    assert create_client.call_args.kwargs["base_url"] == "https://hopper.example.test/v1"


def _capture_client_kwargs(http2_available: bool):
    captured = {}

    def fake_http_client(**kwargs):
        captured["http_client"] = kwargs
        return object()

    def fake_async_openai(**kwargs):
        captured["openai"] = kwargs
        return object()

    with (
        patch.object(hopper_llm, "_http2_available", return_value=http2_available),
        patch.object(hopper_llm, "DefaultAsyncHttpxClient", side_effect=fake_http_client),
        patch.object(hopper_llm, "AsyncOpenAI", side_effect=fake_async_openai),
    ):
        HopperLLMService(api_key="sk_hopper_test")

    return captured


def test_hopper_llm_client_uses_http2_when_h2_is_installed():
    captured = _capture_client_kwargs(http2_available=True)

    assert captured["http_client"]["http2"] is True
    assert captured["openai"]["api_key"] == "sk_hopper_test"
    assert captured["openai"]["base_url"] == "https://api.withhopper.com/v1"


def test_hopper_llm_client_falls_back_to_http1_without_h2():
    captured = _capture_client_kwargs(http2_available=False)

    assert captured["http_client"]["http2"] is False
    assert captured["openai"]["base_url"] == "https://api.withhopper.com/v1"
