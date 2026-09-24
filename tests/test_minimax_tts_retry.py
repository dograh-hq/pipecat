#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax retries preserve audio, deadlines, accounting and response ownership."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.services.minimax import tts as minimax
from pipecat.services.minimax.tts import MiniMaxHttpTTSService, MiniMaxSynthesisOutcome
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory
from tests.frame_processor_helpers import frame_processor_setup

pytestmark = pytest.mark.asyncio
PCM = b"\x01\x02" * 64


def _success():
    return web.Response(
        body=(
            b"data:"
            + json.dumps({"data": {"audio": PCM.hex(), "status": 1}}).encode()
            + b'\n\ndata:{"data":{"status":2}}\n\n'
        ),
        content_type="text/event-stream",
    )


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(minimax, "_RETRY_BASE_DELAY_S", 0.02)
    monkeypatch.setattr(minimax, "_RETRY_JITTER_S", 0.01)


@asynccontextmanager
async def _service(aiohttp_client, handler, **kwargs):
    app = web.Application()
    app.router.add_post("/tts", handler)
    client = await aiohttp_client(app)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1)) as session:
        service = MiniMaxHttpTTSService(
            api_key="test-key",
            group_id="test-group",
            base_url=str(client.make_url("/tts")),
            aiohttp_session=session,
            **kwargs,
        )
        manager = TaskManager()
        await service.setup(frame_processor_setup(manager))
        service._audio_sample_rate = 24000
        service.start_tts_usage_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        try:
            yield service
        finally:
            await service.cleanup()
            assert not manager.current_tasks()


async def _collect(service):
    outcome = MiniMaxSynthesisOutcome()
    async with asyncio.timeout(2):
        frames = [
            frame
            async for frame in service._run_tts_request(
                service._build_request("Hello."), "context", outcome
            )
        ]
    return frames, outcome


@pytest.mark.parametrize("code", [1001, 1002, 1024, 1033, 1039, 1041, 2045])
@pytest.mark.parametrize("sse", [False, True])
async def test_transient_refusals_recover_without_errors_or_duplicate_usage(
    aiohttp_client, code, sse
):
    requests = []

    async def handler(request):
        requests.append(await request.json())
        if len(requests) < 3:
            body = {"base_resp": {"status_code": code, "status_msg": "try later"}}
            if sse:
                return web.Response(body=b"data:" + json.dumps(body).encode() + b"\n\n")
            return web.json_response(body)
        return _success()

    async with _service(aiohttp_client, handler) as service:
        frames, outcome = await _collect(service)
        assert outcome.completed
        assert len(requests) == 3
        assert requests[0] == requests[1] == requests[2]
        assert not any(isinstance(f, ErrorFrame) for f in frames)
        assert b"".join(f.audio for f in frames) == PCM
        service.start_tts_usage_metrics.assert_awaited_once_with("Hello.")


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_http_transient_failure_recovers(aiohttp_client, status):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(status=status) if calls == 1 else _success()

    async with _service(aiohttp_client, handler) as service:
        frames, outcome = await _collect(service)
        assert outcome.completed and calls == 2
        assert b"".join(f.audio for f in frames) == PCM


@pytest.mark.parametrize("http_status", [200, 429, 503])
async def test_retry_after_is_a_minimum_delay(aiohttp_client, http_status):
    attempts = []

    async def handler(request):
        attempts.append(asyncio.get_running_loop().time())
        if len(attempts) == 1:
            return web.json_response(
                {"base_resp": {"status_code": 1002}},
                status=http_status,
                headers={"Retry-After": "1"},
            )
        return _success()

    async with _service(aiohttp_client, handler) as service:
        frames, outcome = await _collect(service)
        assert outcome.completed and len(attempts) == 2
        assert attempts[1] - attempts[0] >= 1
        assert b"".join(f.audio for f in frames) == PCM


@pytest.mark.parametrize("date_header", [False, True])
async def test_retry_after_beyond_budget_surfaces_refusal_without_another_attempt(
    aiohttp_client, date_header
):
    calls = 0
    retry_after = format_datetime(datetime.now(UTC) + timedelta(minutes=1)) if date_header else "60"

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(status=429, headers={"Retry-After": retry_after})

    async with _service(aiohttp_client, handler) as service:
        frames, outcome = await _collect(service)
        assert calls == 1 and not outcome.completed
        assert len(frames) == 1 and frames[0].category is ErrorCategory.RATE_LIMIT


@pytest.mark.parametrize(
    "retry_after", ["invalid", "nan", "inf", "-1", "Sun, 06 Nov 1994 08:49:37 GMT"]
)
async def test_invalid_or_expired_retry_after_does_not_prevent_recovery(
    aiohttp_client, retry_after
):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.Response(status=429, headers={"Retry-After": retry_after})
        return _success()

    async with _service(aiohttp_client, handler) as service:
        frames, outcome = await _collect(service)
        assert calls == 2 and outcome.completed
        assert b"".join(f.audio for f in frames) == PCM


@pytest.mark.parametrize("code", [1004, 1008, 2013, 2042, 2049, 2056, 20132, 1026])
async def test_permanent_quota_and_unclassified_refusals_are_not_retried(aiohttp_client, code):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.json_response({"base_resp": {"status_code": code}})

    async with _service(aiohttp_client, handler) as service:
        frames, outcome = await _collect(service)
        assert calls == 1 and not outcome.completed
        assert len(frames) == 1 and isinstance(frames[0], ErrorFrame)
        service.start_tts_usage_metrics.assert_not_awaited()


async def test_exhausted_attempts_emit_only_the_last_error(aiohttp_client):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.json_response({"base_resp": {"status_code": 1002}, "trace_id": str(calls)})

    async with _service(aiohttp_client, handler) as service:
        frames, outcome = await _collect(service)
        assert calls == 3 and not outcome.completed
        assert len(frames) == 1 and "trace_id=3" in frames[0].error
        assert frames[0].category is ErrorCategory.RATE_LIMIT
        service.start_tts_usage_metrics.assert_not_awaited()


async def test_partial_audio_is_never_retried(aiohttp_client):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(
            body=(
                b"data:"
                + json.dumps({"data": {"audio": PCM.hex(), "status": 1}}).encode()
                + b'\n\ndata:{"base_resp":{"status_code":1002}}\n\n'
            )
        )

    async with _service(aiohttp_client, handler) as service:
        frames, outcome = await _collect(service)
        assert calls == 1 and not outcome.completed
        assert b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame)) == PCM
        assert sum(isinstance(f, ErrorFrame) for f in frames) == 1
        service.start_tts_usage_metrics.assert_awaited_once_with("Hello.")


async def test_nonstreaming_request_retries_without_duplicate_audio(aiohttp_client):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        assert (await request.json())["stream"] is False
        if calls == 1:
            return web.json_response({"base_resp": {"status_code": 1002}})
        return web.json_response({"data": {"audio": PCM.hex(), "status": 2}})

    async with _service(aiohttp_client, handler, stream=False) as service:
        frames, outcome = await _collect(service)
        assert calls == 2 and outcome.completed
        assert b"".join(f.audio for f in frames) == PCM
        service.start_tts_usage_metrics.assert_awaited_once_with("Hello.")


async def test_closing_partial_synthesis_releases_the_response(aiohttp_client):
    release = asyncio.Event()

    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        chunk = b"data:" + json.dumps({"data": {"audio": PCM.hex(), "status": 1}}).encode()
        await response.write(chunk + b"\n\n")
        await asyncio.wait_for(release.wait(), 2)
        return response

    try:
        async with _service(aiohttp_client, handler) as service:
            outcome = MiniMaxSynthesisOutcome()
            frames = service._run_tts_request(service._build_request("Hello."), "context", outcome)
            try:
                frame = await asyncio.wait_for(anext(frames), 1)
                assert isinstance(frame, TTSAudioRawFrame)
            finally:
                await asyncio.wait_for(frames.aclose(), 1)
            assert not outcome.completed
            assert not service._session.connector._acquired
    finally:
        release.set()


@pytest.mark.parametrize(
    "failure", [TimeoutError(), aiohttp.ServerDisconnectedError(), aiohttp.ClientPayloadError()]
)
async def test_network_exceptions_are_classified_before_retry(aiohttp_client, failure):
    async def handler(request):
        return _success()

    async with _service(aiohttp_client, handler) as service:
        original = service._session.post
        with patch.object(
            service._session,
            "post",
            side_effect=[
                failure,
                original(service._base_url, json=service._build_request("Hello.")),
            ],
        ) as post:
            frames, outcome = await _collect(service)
        assert post.call_count == 2 and outcome.completed
        assert b"".join(f.audio for f in frames) == PCM


async def test_retry_closes_an_unfinished_response_before_reusing_the_connector(aiohttp_client):
    calls = 0
    release = asyncio.Event()

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls > 1:
            return _success()
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b'data:{"base_resp":{"status_code":1002}}\n\n')
        await asyncio.wait_for(release.wait(), 2)
        return response

    try:
        async with _service(aiohttp_client, handler) as service:
            frames, outcome = await _collect(service)
            assert calls == 2 and outcome.completed
            assert b"".join(f.audio for f in frames) == PCM
    finally:
        release.set()


async def test_first_audio_deadline_closes_a_stalled_request(aiohttp_client):
    release = asyncio.Event()
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        response = web.StreamResponse()
        await response.prepare(request)
        await asyncio.wait_for(release.wait(), 2)
        return response

    try:
        async with _service(aiohttp_client, handler, retry_timeout_secs=0.06) as service:
            frames, outcome = await _collect(service)
            assert calls == 1 and not outcome.completed
            assert len(frames) == 1 and "timed out before first audio" in frames[0].error
            assert frames[0].category is ErrorCategory.CONNECTIVITY
            assert not service._session.connector._acquired
    finally:
        release.set()


async def test_budget_does_not_restart_on_retry(aiohttp_client, monkeypatch):
    monkeypatch.setattr(minimax, "_RETRY_JITTER_S", 0)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.2)
        return web.json_response({"base_resp": {"status_code": 1002}})

    async with _service(aiohttp_client, handler, retry_timeout_secs=0.35) as service:
        frames, outcome = await _collect(service)
        assert calls == 2 and not outcome.completed
        assert len(frames) == 1 and "timed out before first audio" in frames[0].error


async def test_first_audio_deadline_does_not_cut_off_streaming_speech(aiohttp_client):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        response = web.StreamResponse()
        await response.prepare(request)
        chunk = b"data:" + json.dumps({"data": {"audio": PCM.hex(), "status": 1}}).encode()
        await response.write(chunk + b"\n\n")
        await asyncio.sleep(0.6)
        await response.write(chunk + b'\n\ndata:{"data":{"status":2}}\n\n')
        return response

    async with _service(aiohttp_client, handler, retry_timeout_secs=0.3) as service:
        frames, outcome = await _collect(service)
        assert calls == 1 and outcome.completed
        assert b"".join(f.audio for f in frames) == PCM * 2
        service.start_tts_usage_metrics.assert_awaited_once_with("Hello.")


@pytest.mark.parametrize("during_backoff", [False, True])
async def test_cancellation_closes_requests_and_never_retries(
    aiohttp_client, monkeypatch, during_backoff
):
    monkeypatch.setattr(minimax, "_RETRY_BASE_DELAY_S", 0.3)
    requested = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        requested.set()
        if during_backoff:
            return web.json_response({"base_resp": {"status_code": 1002}})
        await asyncio.wait_for(release.wait(), 2)
        return _success()

    try:
        async with _service(aiohttp_client, handler) as service:
            task = asyncio.create_task(_collect(service))
            try:
                await asyncio.wait_for(requested.wait(), 1)
                if during_backoff:
                    await asyncio.sleep(0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
                assert calls == 1
                assert not service._session.connector._acquired
                service.start_tts_usage_metrics.assert_not_awaited()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    finally:
        release.set()
