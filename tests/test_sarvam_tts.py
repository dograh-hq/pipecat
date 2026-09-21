#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Sarvam TTS text gating and bulbul:v4-flash support."""

import json
import unittest

import pytest
from websockets.protocol import State

from pipecat.services.sarvam.tts import SarvamTTSService, _SpeakableTextFilter

V4_FLASH = "bulbul:v4-flash"


class TestSpeakableTextFilter(unittest.IsolatedAsyncioTestCase):
    """Sarvam rejects requests holding no target-language character."""

    async def test_punctuation_only_text_is_emptied(self):
        filter = _SpeakableTextFilter()

        # A doubled full stop is split by sentence aggregation, stranding this.
        self.assertEqual(await filter.filter("."), "")
        self.assertEqual(await filter.filter(".."), "")
        self.assertEqual(await filter.filter("…"), "")
        self.assertEqual(await filter.filter(" -- "), "")
        self.assertEqual(await filter.filter("🙂"), "")

    async def test_speakable_text_passes_through_unchanged(self):
        filter = _SpeakableTextFilter()

        for text in (
            "hamare teeno models alag-alag power requirements ke liye bane hain.",
            "दूसरी तरफ, इलेक्ट्रिक मोटर में सीधे बैटरी से ताकत मिलती है।",
            "X45",
            "123",
        ):
            self.assertEqual(await filter.filter(text), text)

    async def test_script_is_not_assumed_to_be_latin(self):
        """Devanagari, Han and Cyrillic are speakable exactly as Latin is."""
        filter = _SpeakableTextFilter()

        for text in ("हाँ", "你好", "да"):
            self.assertEqual(await filter.filter(text), text)


class _FakeWebsocket:
    def __init__(self, *, state=State.OPEN):
        self.state = state
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        self.state = State.CLOSED


def _service(**settings_kwargs) -> SarvamTTSService:
    settings_kwargs.setdefault("model", V4_FLASH)
    return SarvamTTSService(
        api_key="test-key", settings=SarvamTTSService.Settings(**settings_kwargs)
    )


def _sent_config(service: SarvamTTSService) -> dict:
    """Read back the config the service put on the wire."""
    messages = [json.loads(m) for m in service._websocket.sent]
    configs = [m["data"] for m in messages if m["type"] == "config"]
    assert configs, "no config message was sent"
    return configs[-1]


def test_v4_flash_defaults():
    """The service has to land on the same speaker and rate the API defaults to."""
    service = _service()

    assert service._settings.voice == "shubh_en_narration_gentle"
    assert service._init_sample_rate == 24000
    assert service._settings.enable_preprocessing is True


@pytest.mark.asyncio
async def test_v4_flash_sends_pitch_and_loudness():
    """v4-flash applies both; the v3 models ignore them."""
    service = _service(pitch=0.3, loudness=1.5, pace=1.2)
    service._websocket = _FakeWebsocket()

    await service._send_config()
    config = _sent_config(service)

    assert config["pitch"] == 0.3
    assert config["loudness"] == 1.5
    assert config["pace"] == 1.2


@pytest.mark.asyncio
async def test_v4_flash_drops_temperature():
    """The API pins it to 0.6, so sending a value would only mislead."""
    service = _service(temperature=0.9)
    service._websocket = _FakeWebsocket()

    await service._send_config()

    assert "temperature" not in _sent_config(service)


def test_v4_flash_clamps_pace_to_its_own_range():
    service = _service(pace=2.5)

    assert service._settings.pace == 2.0


if __name__ == "__main__":
    unittest.main()
