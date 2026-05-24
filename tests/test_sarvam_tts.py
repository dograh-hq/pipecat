#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SarvamTTSService."""

import pytest

from pipecat.services.sarvam.tts import SarvamTTSService


def _make_service():
    return SarvamTTSService(api_key="test-key")


async def _drain(agen):
    """Collect everything yielded by an async generator."""
    return [frame async for frame in agen]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ['"', "   ", '"  "', "।", "...", "—"])
async def test_run_tts_skips_unsynthesizable_text(monkeypatch, text):
    """Punctuation/whitespace-only chunks must not be sent to Sarvam.

    Sarvam returns "400: Text must contain at least one character from the
    allowed languages." for such fragments and tears down the websocket, so
    run_tts should skip them entirely. Sentence aggregation can isolate a
    trailing closing quote as its own chunk, which previously broke the
    end-of-call goodbye line.
    """
    service = _make_service()

    sent = []
    connected = []

    async def fake_send_text(t):
        sent.append(t)

    async def fake_connect():
        connected.append(True)

    monkeypatch.setattr(service, "_send_text", fake_send_text)
    monkeypatch.setattr(service, "_connect", fake_connect)

    frames = await _drain(service.run_tts(text, context_id="ctx-1"))

    assert sent == [], f"unsynthesizable text {text!r} should not be sent"
    assert connected == [], "should not even open a connection for skipped text"
    assert frames == [], "no frames should be produced for skipped text"


@pytest.mark.asyncio
async def test_run_tts_sends_synthesizable_text(monkeypatch):
    """Text with at least one alphanumeric character is sent for synthesis."""
    service = _make_service()

    sent = []

    async def fake_send_text(t):
        sent.append(t)

    async def fake_connect():
        return None

    async def fake_metrics(_t):
        return None

    monkeypatch.setattr(service, "_send_text", fake_send_text)
    monkeypatch.setattr(service, "_connect", fake_connect)
    monkeypatch.setattr(service, "start_tts_usage_metrics", fake_metrics)

    await _drain(service.run_tts("धन्यवाद, आपका दिन शुभ हो।", context_id="ctx-1"))

    assert sent == ["धन्यवाद, आपका दिन शुभ हो।"]
