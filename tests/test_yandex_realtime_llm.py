#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for YandexRealtimeLLMService's required-model validation.

Without an explicit Yandex model resource, this service would silently
inherit OpenAIRealtimeLLMService's OpenAI model default and send an invalid
``?model=`` query to Yandex Cloud's endpoint.
"""

import pytest

from pipecat.services.openai.realtime.events import SessionProperties
from pipecat.services.yandex.realtime.llm import YandexRealtimeLLMService

YANDEX_MODEL = "gpt://folder-1/speech-realtime-deepseek-v4-flash/latest"


def test_missing_model_raises():
    with pytest.raises(ValueError, match="requires an explicit Yandex model"):
        YandexRealtimeLLMService(api_key="test-key")


def test_explicit_settings_model_is_accepted():
    service = YandexRealtimeLLMService(
        api_key="test-key",
        settings=YandexRealtimeLLMService.Settings(model=YANDEX_MODEL),
    )
    assert service._settings.model == YANDEX_MODEL


def test_deprecated_top_level_model_is_accepted():
    service = YandexRealtimeLLMService(api_key="test-key", model=YANDEX_MODEL)
    assert service._settings.model == YANDEX_MODEL


def test_deprecated_session_properties_model_is_accepted():
    service = YandexRealtimeLLMService(
        api_key="test-key",
        session_properties=SessionProperties(model=YANDEX_MODEL),
    )
    assert service._settings.model == YANDEX_MODEL
