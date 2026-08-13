#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SpeechifyTTSService."""

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.speechify.tts import (
    SpeechifyTTSService,
    language_to_speechify_language,
    output_format_from_sample_rate,
)
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language


def test_output_format_from_sample_rate():
    assert output_format_from_sample_rate(8000) == "pcm_8000"
    assert output_format_from_sample_rate(16000) == "pcm_16000"
    assert output_format_from_sample_rate(24000) == "pcm_24000"
    assert output_format_from_sample_rate(48000) == "pcm_48000"
    # Unsupported rates are rejected: Speechify would return audio at a
    # different rate than the frames are labeled with.
    with pytest.raises(ValueError, match="unsupported sample rate 11025"):
        output_format_from_sample_rate(11025)


def test_language_to_speechify_language():
    assert language_to_speechify_language(Language.EN) == "en"
    assert language_to_speechify_language(Language.PT_BR) == "pt-BR"
    # Unmapped regional variants fall back to the base language code
    assert language_to_speechify_language(Language.EN_GB) == "en"


@pytest.mark.asyncio
async def test_run_speechify_tts_success(aiohttp_client):
    """Streams chunked PCM and checks request payload and frame ordering."""
    received_payloads = []

    async def handler(request):
        received_payloads.append(await request.json())

        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "audio/L16"},
        )
        await resp.prepare(request)
        # Second chunk has an odd length to exercise the 16-bit sample carry.
        await resp.write(b"\x00\x01" * 6000)
        await resp.write(b"\x02\x03" * 6000 + b"\x04")
        await resp.write(b"\x05")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/audio/stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = SpeechifyTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=16000,
        )

        frames_received = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello world.", append_to_context=False)],
        )
        down_frames = frames_received[0]
        frame_types = [type(f) for f in down_frames]

        assert AggregatedTextFrame in frame_types
        assert TTSStartedFrame in frame_types
        assert TTSStoppedFrame in frame_types
        assert TTSTextFrame in frame_types

        started_idx = frame_types.index(TTSStartedFrame)
        stopped_idx = frame_types.index(TTSStoppedFrame)
        text_idx = frame_types.index(TTSTextFrame)
        assert started_idx < text_idx < stopped_idx, (
            "Expected: TTSStartedFrame < TTSTextFrame < TTSStoppedFrame"
        )

        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) >= 1, "Expected at least one audio frame"
        for a_frame in audio_frames:
            assert a_frame.sample_rate == 16000
            assert len(a_frame.audio) % 2 == 0, "Audio frames must not split 16-bit samples"
        total_audio = sum(len(f.audio) for f in audio_frames)
        assert total_audio == 6000 * 2 + 6000 * 2 + 2, "All streamed bytes must be delivered"

    assert received_payloads == [
        {
            "input": "Hello world.",
            "voice_id": "beatrice_32",
            "model": "simba-3.2",
            "output_format": "pcm_16000",
        }
    ]


@pytest.mark.asyncio
async def test_run_speechify_tts_settings_and_language(aiohttp_client):
    """Custom settings reach the request payload; language is converted."""
    received_payloads = []

    async def handler(request):
        received_payloads.append(await request.json())
        return web.Response(status=200, body=b"\x00\x01" * 100)

    app = web.Application()
    app.router.add_post("/v1/audio/stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = SpeechifyTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=8000,
            settings=SpeechifyTTSService.Settings(
                model="simba-3.0",
                voice="geffen_32",
                language=Language.PT_BR,
            ),
        )

        await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Olá.", append_to_context=False)],
        )

    assert received_payloads == [
        {
            "input": "Olá.",
            "voice_id": "geffen_32",
            "model": "simba-3.0",
            "output_format": "pcm_8000",
            "language": "pt-BR",
        }
    ]


@pytest.mark.asyncio
async def test_run_speechify_tts_error(aiohttp_client):
    """A non-200 response yields an ErrorFrame with status details."""

    async def handler(_request):
        return web.Response(status=401, text="Unauthorized")

    app = web.Application()
    app.router.add_post("/v1/audio/stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = SpeechifyTTSService(
            api_key="bad-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=16000,
        )

        frames_received = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Error case.", append_to_context=False)],
            expected_down_frames=[
                AggregatedTextFrame,
                TTSStartedFrame,
                TTSTextFrame,
                TTSStoppedFrame,
            ],
            expected_up_frames=[ErrorFrame],
        )
        up_frames = frames_received[1]

        assert isinstance(up_frames[0], ErrorFrame)
        assert "status: 401" in up_frames[0].error


@pytest.mark.asyncio
async def test_speechify_tts_owned_session(aiohttp_client):
    """Without an injected session, the service creates and closes its own."""

    async def handler(_request):
        return web.Response(status=200, body=b"\x00\x01" * 100)

    app = web.Application()
    app.router.add_post("/v1/audio/stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    tts_service = SpeechifyTTSService(
        api_key="test-key",
        base_url=base_url,
        sample_rate=16000,
    )

    frames_received = await run_test(
        tts_service,
        frames_to_send=[TTSSpeakFrame(text="Own session.", append_to_context=False)],
    )
    down_frames = frames_received[0]
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) >= 1

    await tts_service.cleanup()
    assert tts_service._session is None
