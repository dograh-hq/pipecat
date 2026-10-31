#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Contract tests for Dograh-owned telephony serializers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import CancelFrame, EndFrame, OutputTransportMessageFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.asterisk import AsteriskFrameSerializer
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.serializers.vobiz import VobizFrameSerializer
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.enums import EndTaskReason


def _setup(*, sample_rate: int = 16000) -> FrameProcessorSetup:
    return FrameProcessorSetup(
        audio_in_sample_rate=sample_rate,
        clock=SystemClock(),
        task_manager=TaskManager(),
        pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_vobiz_uses_v18_setup_and_filters_rtvi_messages():
    serializer = VobizFrameSerializer(
        stream_id="stream",
        params=VobizFrameSerializer.InputParams(auto_hang_up=False),
    )

    await serializer.setup(_setup())

    assert serializer._sample_rate == 16000
    assert (
        await serializer.serialize(
            OutputTransportMessageFrame(message={"label": "rtvi-ai", "type": "test"})
        )
        is None
    )


@pytest.mark.asyncio
async def test_asterisk_uses_v18_setup_contract():
    serializer = AsteriskFrameSerializer(
        channel_id="channel",
        ari_endpoint="http://asterisk.invalid",
        app_name="app",
        app_password="secret",
        params=AsteriskFrameSerializer.InputParams(sample_rate=24000),
    )

    await serializer.setup(_setup(sample_rate=16000))

    assert serializer._sample_rate == 24000


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["twilio", "telnyx", "plivo"])
@pytest.mark.parametrize("first_frame", [EndFrame, CancelFrame])
@pytest.mark.parametrize("repeated_frame", [EndFrame, CancelFrame])
async def test_repeated_transfer_shutdown_never_falls_through_to_hangup(
    provider, first_frame, repeated_frame
):
    cls, credentials = {
        "twilio": (
            TwilioFrameSerializer,
            dict(
                stream_sid="stream", call_sid="call", account_sid="account", auth_token="test-token"
            ),
        ),
        "telnyx": (
            TelnyxFrameSerializer,
            dict(
                stream_id="stream",
                call_control_id="call",
                api_key="test-key",
                outbound_encoding="PCMU",
                inbound_encoding="PCMU",
            ),
        ),
        "plivo": (
            PlivoFrameSerializer,
            dict(stream_id="stream", call_id="call", auth_id="account", auth_token="test-token"),
        ),
    }[provider]
    transfer, hangup = AsyncMock(), AsyncMock()
    serializer = cls(transfer_strategy=transfer, hangup_strategy=hangup, **credentials)
    reason = EndTaskReason.TRANSFER_CALL.value

    await serializer.serialize(first_frame(reason=reason))
    await serializer.serialize(repeated_frame(reason=reason))

    transfer.execute_transfer.assert_awaited_once()
    hangup.execute_hangup.assert_not_awaited()
