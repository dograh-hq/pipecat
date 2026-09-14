#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for MiniMaxHttpTTSService error reporting.

MiniMax answers HTTP 200 even when it rejects a request and puts the real
outcome in ``base_resp``, which rides on a non-streamed body and on every
streaming chunk alike. A caller that checks only the HTTP status therefore
produces a call that is answered, billed and completely silent, recorded as a
healthy synthesis. These tests pin each place that report can arrive, and the
verdict each one carries about whether the service can speak again.
"""

import json

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.services.minimax.tts import MiniMaxHttpTTSService
from pipecat.tests.utils import run_test
from pipecat.utils.errors import ErrorCategory

AUDIO_HEX = "00ff" * 64
OK_RESP = {"status_code": 0, "status_msg": "success"}
ERROR_RESP = {"status_code": 2042, "status_msg": "you don't have access to this voice_id"}


def _sse(payload: dict) -> bytes:
    return b"data:" + json.dumps(payload).encode() + b"\n\n"


def _audio_chunk(base_resp=None) -> dict:
    return {
        "data": {"audio": AUDIO_HEX, "status": 1},
        "trace_id": "trace-test",
        "base_resp": base_resp or OK_RESP,
    }


def _summary_chunk(base_resp=None) -> dict:
    """The closing chunk: status 2, no audio, extra_info, and its own base_resp."""
    return {
        "data": {"audio": "", "status": 2},
        "trace_id": "trace-test",
        "extra_info": {"audio_length": 1500, "usage_characters": 3},
        "base_resp": base_resp or OK_RESP,
    }


async def _serve(handler):
    app = web.Application()
    app.router.add_post("/v1/t2a_v2", handler)
    return app


def _streaming_handler(chunks):
    async def handler(request):
        response = web.StreamResponse(status=200)
        await response.prepare(request)
        for chunk in chunks:
            await response.write(_sse(chunk))
        await response.write_eof()
        return response

    return handler


async def _frames_for(handler, aiohttp_client):
    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v1/t2a_v2"))

    async with aiohttp.ClientSession() as session:
        tts = MiniMaxHttpTTSService(
            api_key="test-key",
            group_id="test-group",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )
        down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    frames = list(down) + list(up)
    return (
        [f for f in frames if isinstance(f, TTSAudioRawFrame)],
        [f for f in frames if isinstance(f, ErrorFrame)],
    )


@pytest.mark.asyncio
async def test_healthy_stream_produces_audio_and_no_error(aiohttp_client):
    """The control: base_resp is 0 throughout, so nothing is reported."""
    handler = _streaming_handler([_audio_chunk(), _audio_chunk(), _summary_chunk()])

    audio, errors = await _frames_for(handler, aiohttp_client)

    assert audio
    assert not errors


@pytest.mark.asyncio
async def test_rejection_before_the_stream_is_reported(aiohttp_client):
    """A rejected request arrives as a plain JSON body, with a 200 on it.

    This is the shape that never reaches the data-block loop at all, so it used
    to end the generator with no audio, no error and no exception.
    """

    async def handler(_request):
        return web.json_response({"base_resp": ERROR_RESP, "trace_id": "t-1"}, status=200)

    audio, errors = await _frames_for(handler, aiohttp_client)

    assert not audio
    assert errors
    assert "2042" in errors[0].error
    assert "voice_id" in errors[0].error
    # trace_id is the first thing MiniMax support asks for.
    assert "t-1" in errors[0].error


@pytest.mark.asyncio
async def test_rejection_as_a_single_sse_event_is_reported(aiohttp_client):
    """The same rejection, framed as one event with nothing following it."""
    handler = _streaming_handler([{"base_resp": ERROR_RESP, "trace_id": "t-2"}])

    audio, errors = await _frames_for(handler, aiohttp_client)

    assert not audio
    assert errors
    assert "2042" in errors[0].error


@pytest.mark.asyncio
async def test_failure_part_way_through_a_stream_is_reported(aiohttp_client):
    """Audio starts, then a chunk carries a failure.

    The caller hears part of a sentence and then nothing, so the run has to say
    so rather than record a complete synthesis.
    """
    handler = _streaming_handler(
        [_audio_chunk(), _audio_chunk(ERROR_RESP), _summary_chunk(ERROR_RESP)]
    )

    _, errors = await _frames_for(handler, aiohttp_client)

    assert errors
    assert "2042" in errors[0].error


@pytest.mark.asyncio
async def test_failure_reported_only_in_the_closing_chunk_is_reported(aiohttp_client):
    """The closing chunk carries its own base_resp, and it is the last one.

    Nothing follows it to delimit the block, so reaching it means draining what
    the stream left behind.
    """
    handler = _streaming_handler([_audio_chunk(), _summary_chunk(ERROR_RESP)])

    _, errors = await _frames_for(handler, aiohttp_client)

    assert errors
    assert "2042" in errors[0].error


@pytest.mark.asyncio
async def test_http_level_error_is_still_reported(aiohttp_client):
    """The pre-existing check keeps working."""

    async def handler(_request):
        return web.json_response({"detail": "unauthorized"}, status=401)

    audio, errors = await _frames_for(handler, aiohttp_client)

    assert not audio
    assert errors
    assert "401" in errors[0].error
    assert errors[0].category is ErrorCategory.AUTHENTICATION


@pytest.mark.asyncio
async def test_permanent_rejection_costs_the_service_its_usability(aiohttp_client):
    """A code that rejects the configuration rejects every later turn too.

    Reporting it as permanent is what stops the pipeline handing the service
    more text to speak, instead of leaving the line silent turn after turn.
    """
    handler = _streaming_handler([{"base_resp": ERROR_RESP, "trace_id": "t-5"}])

    _, errors = await _frames_for(handler, aiohttp_client)

    assert errors[0].category is ErrorCategory.AUTHORIZATION
    assert not errors[0].processor.is_usable


@pytest.mark.asyncio
async def test_transient_rejection_leaves_the_service_usable(aiohttp_client):
    """A rate limit says nothing about whether the next turn will be spoken."""
    rate_limited = {"status_code": 1002, "status_msg": "rate limit"}
    handler = _streaming_handler([{"base_resp": rate_limited, "trace_id": "t-6"}])

    _, errors = await _frames_for(handler, aiohttp_client)

    assert errors[0].category is ErrorCategory.RATE_LIMIT
    assert errors[0].processor.is_usable


@pytest.mark.asyncio
async def test_absent_base_resp_is_not_treated_as_a_failure(aiohttp_client):
    """Chunks without base_resp are normal; only a non-zero status_code is not."""
    handler = _streaming_handler(
        [
            {"data": {"audio": AUDIO_HEX, "status": 1}},
            {"data": {"audio": "", "status": 2}, "extra_info": {}},
        ]
    )

    audio, errors = await _frames_for(handler, aiohttp_client)

    assert audio
    assert not errors
