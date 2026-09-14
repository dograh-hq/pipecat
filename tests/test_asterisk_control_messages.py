#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AsteriskFrameSerializer control message parsing.

chan_websocket sends control messages as text frames in one of two formats.
JSON is preferred from Asterisk 20.18.0, 22.8.0 and 23.2.0; plain text is the
default everywhere and is deprecated. Which one arrives depends on the
operator's chan_websocket.conf or f(<format>) dialstring, so both have to be
read - a deployment on defaults sends the plain-text form and nothing here
controls that.

https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
"""

import io
import json
import unittest

from loguru import logger

from pipecat.serializers.asterisk import AsteriskFrameSerializer, _parse_control_message
from tests.frame_processor_helpers import frame_processor_setup

SAMPLE_RATE = 8000

# Copied verbatim from a production Asterisk 22.11.0 call.
MEDIA_START_PLAIN = (
    "MEDIA_START connection_id:dograh channel:WebSocket/dograh/0x78b36c003130 "
    "channel_id:dograh-ext-98fc90e9-3f20-4244-b20a-ecb5e52da23e "
    "format:ulaw optimal_frame_size:160 ptime:20"
)


class TestParseControlMessage(unittest.TestCase):
    def test_plain_text_media_start(self):
        message = _parse_control_message(MEDIA_START_PLAIN)

        assert message == {
            "event": "MEDIA_START",
            "connection_id": "dograh",
            "channel": "WebSocket/dograh/0x78b36c003130",
            "channel_id": "dograh-ext-98fc90e9-3f20-4244-b20a-ecb5e52da23e",
            "format": "ulaw",
            "optimal_frame_size": "160",
            "ptime": "20",
        }

    def test_json_form_is_returned_unchanged(self):
        payload = {"event": "MEDIA_START", "format": "ulaw", "ptime": 20}

        assert _parse_control_message(json.dumps(payload)) == payload

    def test_events_without_parameters(self):
        assert _parse_control_message("QUEUE_DRAINED") == {"event": "QUEUE_DRAINED"}

    def test_events_carrying_a_channel_id(self):
        # The flow-control and DTMF events all take this shape.
        for name in ("MEDIA_XOFF", "MEDIA_XON"):
            assert _parse_control_message(f"{name} channel_id:abc") == {
                "event": name,
                "channel_id": "abc",
            }

    def test_dtmf_digit_is_preserved(self):
        assert _parse_control_message("DTMF_END channel_id:abc digit:5") == {
            "event": "DTMF_END",
            "channel_id": "abc",
            "digit": "5",
        }

    def test_values_may_contain_colons(self):
        # Splitting on the first colon only, so a value keeps any it contains.
        message = _parse_control_message("STATUS channel_id:a:b:c queue_length:3")

        assert message["channel_id"] == "a:b:c"
        assert message["queue_length"] == "3"

    def test_unparseable_input_returns_none(self):
        assert _parse_control_message("") is None
        assert _parse_control_message("   ") is None
        assert _parse_control_message("{not json") is None
        # Valid JSON that is not an object tells us nothing about an event.
        assert _parse_control_message("[1, 2]") is None


class TestAsteriskDeserializeControlMessages(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = AsteriskFrameSerializer(
            channel_id="test-channel",
            ari_endpoint="http://asterisk:8088",
            app_name="dograh",
            app_password="secret",
        )
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_plain_text_control_message_is_consumed_quietly(self):
        # Control messages carry no audio, so None is the right result - the
        # point is that it no longer arrives via a JSONDecodeError.
        assert await self.serializer.deserialize(MEDIA_START_PLAIN) is None

    async def test_json_control_message_is_consumed_quietly(self):
        payload = json.dumps({"event": "MEDIA_START", "format": "ulaw", "ptime": 20})

        assert await self.serializer.deserialize(payload) is None

    async def _warnings_from(self, message: str) -> str:
        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            await self.serializer.deserialize(message)
        finally:
            logger.remove(handler_id)
        return sink.getvalue()

    async def test_declared_format_mismatch_is_reported(self):
        # Every sample would be decoded wrong, which otherwise looks like a bad
        # line rather than a configuration error.
        warnings = await self._warnings_from(
            "MEDIA_START channel_id:abc format:alaw optimal_frame_size:160 ptime:20"
        )

        assert "alaw" in warnings

    async def test_declared_ulaw_does_not_warn(self):
        assert await self._warnings_from(MEDIA_START_PLAIN) == ""

    async def test_unrecognised_message_is_reported(self):
        assert "Unrecognised" in await self._warnings_from("!!! not a control message")
