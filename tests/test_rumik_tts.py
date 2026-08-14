#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smoke tests for Rumik Silk TTS services.

These only cover construction and settings defaults — they don't hit the
network. End-to-end behavior against the live Rumik API is verified via
manual smoke testing.
"""

import unittest

from pipecat.services.rumik.tts import (
    RumikHttpTTSService,
    RumikTTSService,
    RumikTTSSettings,
)


class TestRumikTTSSettings(unittest.TestCase):
    def test_defaults(self):
        service = RumikTTSService(api_key="test-key")
        self.assertEqual(service._settings.model, "muga")
        self.assertIsNone(service._settings.voice)
        # sample_rate is only finalized on start(); pre-start, the requested
        # init value lives in _init_sample_rate.
        self.assertEqual(service._init_sample_rate, 24000)

    def test_mulberry_model_selection(self):
        service = RumikTTSService(api_key="test-key", settings=RumikTTSSettings(model="mulberry"))
        self.assertEqual(service._settings.model, "mulberry")

    def test_settings_override(self):
        settings = RumikTTSSettings(description="a calm, reassuring narrator")
        service = RumikTTSService(api_key="test-key", settings=settings)
        self.assertEqual(service._settings.description, "a calm, reassuring narrator")

    def test_default_gateway_url(self):
        service = RumikTTSService(api_key="test-key")
        self.assertEqual(service._gateway_url, "https://silk-api.rumik.ai")


class TestRumikHttpTTSService(unittest.TestCase):
    def test_defaults(self):
        service = RumikHttpTTSService(api_key="test-key", aiohttp_session=None)
        self.assertEqual(service._settings.model, "muga")
        self.assertEqual(service._init_sample_rate, 24000)


if __name__ == "__main__":
    unittest.main()
