#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for MiniMaxTTSService, the bidirectional WebSocket MiniMax service.

MiniMax's bidirectional API carries no client-chosen request id, so the service
matches audio to replies by order. The sequences fed in below are the ones
observed against the live API on 2026-10-11: after ``task_cancel`` the cancelled
sentence's audio keeps arriving until ``task_canceled``, a sentence that had
started but not yet sent audio can deliver a block under a brand-new trace id,
and every flush pending at the cancel is acknowledged after ``task_canceled``.
"""

import asyncio
import json

import pytest
import websockets
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.minimax.tts import MiniMaxTTSService, _subtitle_words
from pipecat.tests.utils import run_test

OK = {"status_code": 0, "status_msg": "success"}


def _event(event: str, trace: str = "session", **extra) -> dict:
    return {"event": event, "trace_id": trace, "session_id": "s", "base_resp": OK, **extra}


def _audio(trace: str, n_bytes: int = 4, subtitle: dict | None = None) -> dict:
    data = {"audio": "01" * n_bytes}
    if subtitle:
        data["subtitle"] = subtitle
    return _event("task_continued", trace, data=data, is_final=False)


def _final(trace: str) -> dict:
    return _event("task_continued", trace, data={"audio": ""}, is_final=True)


def _subtitle(text: str, pieces: list[tuple[str, int, int, int, int]]) -> dict:
    return {
        "text": text,
        "text_begin": 0,
        "text_end": len(text),
        "time_begin": 0,
        "time_end": pieces[-1][4],
        "timestamped_words": [
            {"word": w, "word_begin": b, "word_end": e, "time_begin": tb, "time_end": te}
            for w, b, e, tb, te in pieces
        ],
    }


# Anonymized subtitle pieces from the live API (speech-2.8-turbo, Italian voice).
BUONGIORNO = _subtitle(
    "Buongiorno, sono Filippo di Acme.",
    [
        ("B", 0, 1, 33, 131), ("u", 1, 2, 131, 197), ("ong", 2, 5, 197, 263), ("ior", 5, 8, 263, 361),
        ("no", 8, 10, 361, 492), (",", 10, 11, 492, 591), (" ", 11, 12, 591, 624), ("son", 12, 15, 624, 886),
        ("o", 15, 16, 886, 1017), (" ", 16, 17, 1017, 1050), ("F", 17, 18, 1050, 1149), ("ili", 18, 21, 1149, 1214),
        ("p", 21, 22, 1214, 1247), ("po", 22, 24, 1247, 1313), (" ", 24, 25, 1313, 1346), ("di", 25, 27, 1346, 1444),
        (" ", 27, 28, 1444, 1477), ("A", 28, 29, 1477, 1510), ("cm", 29, 31, 1510, 1543), ("e", 31, 32, 1543, 1641),
        (".", 32, 33, 1641, 1871),
    ],
)  # fmt: skip


class _FakeSocket:
    def __init__(self):
        self.sent: list[dict] = []
        self.state = State.OPEN

    async def send(self, raw: str):
        self.sent.append(json.loads(raw))

    async def close(self):
        self.state = State.CLOSED


@pytest.fixture
def tts(monkeypatch):
    """A service wired to a fake socket, recording what it hands to its audio contexts."""
    service = MiniMaxTTSService(api_key="test-key", sample_rate=8000)
    service._sample_rate = 8000
    service._websocket = _FakeSocket()
    service.audio, service.words, service.stopped, service.errors = [], [], [], []

    async def append_to_audio_context(context_id, frame):
        if isinstance(frame, TTSAudioRawFrame):
            service.audio.append((context_id, len(frame.audio)))
        elif isinstance(frame, TTSStoppedFrame):
            service.stopped.append(context_id)

    async def add_word_timestamps(word_times, context_id=None, **_):
        service.words.extend((context_id, word, round(t, 3)) for word, t in word_times)

    async def remove_audio_context(context_id):
        pass

    async def push_error_frame(error, force_treat_as_permanent=False):
        service.errors.append(error)

    async def no_metrics(*_args, **_kwargs):
        pass

    monkeypatch.setattr(service, "append_to_audio_context", append_to_audio_context)
    monkeypatch.setattr(service, "add_word_timestamps", add_word_timestamps)
    monkeypatch.setattr(service, "remove_audio_context", remove_audio_context)
    monkeypatch.setattr(service, "push_error_frame", push_error_frame)
    monkeypatch.setattr(service, "audio_context_available", lambda _context_id: True)
    monkeypatch.setattr(service, "start_tts_usage_metrics", no_metrics)
    monkeypatch.setattr(service, "stop_all_metrics", no_metrics)
    return service


async def _say(tts: MiniMaxTTSService, text: str, context_id: str, flush: bool = True):
    async for _ in tts.run_tts(text, context_id):
        pass
    if flush:
        await tts.flush_audio(context_id)


async def _feed(tts: MiniMaxTTSService, *messages: dict):
    for message in messages:
        await tts._on_message(message)


def _sent_events(tts) -> list[str]:
    return [m["event"] for m in tts._websocket.sent]


def test_subtitle_pieces_are_rebuilt_into_whole_words():
    assert _subtitle_words(BUONGIORNO) == [
        ("Buongiorno,", 0.033, 0.591),
        ("sono", 0.624, 1.017),
        ("Filippo", 1.05, 1.313),
        ("di", 1.346, 1.444),
        ("Acme.", 1.477, 1.871),
    ]


@pytest.mark.asyncio
async def test_audio_follows_send_order_and_each_flush_ends_its_reply(tts):
    await _say(tts, "Uno. ", "A")
    await _say(tts, "Due. ", "B")
    await _feed(
        tts,
        _event("sentence_start"), _audio("t1", 4), _final("t1"), _event("sentence_end", "t1"), _event("task_flushed"),
        _event("sentence_start"), _audio("t2", 6), _final("t2"), _event("sentence_end", "t2"), _event("task_flushed"),
    )  # fmt: skip

    assert _sent_events(tts) == ["task_continue", "task_flush", "task_continue", "task_flush"]
    assert tts.audio == [("A", 4), ("B", 6)]
    assert tts.stopped == ["A", "B"]


@pytest.mark.asyncio
async def test_a_reply_that_never_flushed_is_flushed_before_the_next_one(tts):
    await _say(tts, "Uno. ", "A", flush=False)
    await _say(tts, "Due. ", "B", flush=False)

    assert _sent_events(tts) == ["task_continue", "task_flush", "task_continue"]


@pytest.mark.asyncio
async def test_cancel_drops_in_flight_audio_and_the_stale_flush_ack(tts):
    """Barge-in mid-delivery, with the next reply sent before task_canceled arrives."""
    await _say(tts, "Vecchia frase. ", "old")
    await _feed(tts, _event("sentence_start"), _audio("t-old", 4))

    await tts._cancel_outstanding()
    await _say(tts, "Nuova frase. ", "new")
    await _feed(
        tts,
        _audio("t-old", 8),  # in flight when the cancel landed
        _event("task_canceled"),
        _event("task_flushed"),  # the cancelled reply's flush, acknowledged late
        _audio("t-old", 2),  # an old trace after the ack is still old
        _event("sentence_start"),
        _audio("t-new", 6),
        _final("t-new"),
        _event("task_flushed"),
    )

    assert _sent_events(tts) == [
        "task_continue",
        "task_flush",
        "task_cancel",
        "task_continue",
        "task_flush",
    ]
    assert tts.audio == [("old", 4), ("new", 6)]
    assert tts.stopped == ["new"]


@pytest.mark.asyncio
async def test_a_sentence_that_started_before_the_cancel_cannot_leak_under_a_new_trace(tts):
    """Experiment e004: sentence 2 had started but sent no audio when the caller barged in.

    Its first block then arrived after the cancel, under a trace id never seen
    before, so only the ordering (before task_canceled) identifies it as stale.
    """
    await _say(tts, "Ho visto che la vostra sede si trova a Camposampiero. ", "old", flush=False)
    await _say(tts, "Proprio lì abbiamo seguito una falegnameria. ", "old")
    await _feed(
        tts,
        _event("sentence_start"), _audio("t-s1", 10), _final("t-s1"), _event("sentence_end", "t-s1"),
        _event("sentence_start"),  # sentence 2 under way, no audio yet
    )  # fmt: skip

    await tts._cancel_outstanding()
    await _feed(tts, _audio("t-s2", 12), _event("task_canceled"), _event("task_flushed"))
    await _say(tts, "Ha ragione, sarò rapidissimo. ", "new")
    await _feed(
        tts, _event("sentence_start"), _audio("t-new", 6), _final("t-new"), _event("task_flushed")
    )

    assert tts.audio == [("old", 10), ("new", 6)]
    assert tts.stopped == ["new"]


@pytest.mark.asyncio
async def test_audio_between_task_canceled_and_the_next_sentence_start_is_dropped(tts):
    await _say(tts, "Vecchia frase. ", "old")
    await tts._cancel_outstanding()
    await _feed(tts, _event("task_canceled"), _event("task_flushed"), _audio("t-unknown", 8))
    await _say(tts, "Nuova frase. ", "new")
    await _feed(tts, _event("sentence_start"), _audio("t-new", 6), _event("task_flushed"))

    assert tts.audio == [("new", 6)]


@pytest.mark.asyncio
async def test_nothing_is_cancelled_once_every_reply_is_delivered(tts):
    await _say(tts, "Uno. ", "A")
    await _feed(tts, _event("sentence_start"), _audio("t1", 4), _event("task_flushed"))

    await tts._cancel_outstanding()

    assert "task_cancel" not in _sent_events(tts)
    await _say(tts, "Due. ", "B")
    await _feed(tts, _event("sentence_start"), _audio("t2", 6))
    assert tts.audio == [("A", 4), ("B", 6)]


@pytest.mark.asyncio
async def test_word_times_are_offset_by_the_sentences_before_them(tts):
    """Times restart at 0 for each sentence, and a word counts once 80 ms (or half of it) has played."""
    await _say(tts, "Buongiorno, sono Filippo di Acme. ", "A", flush=False)
    await _say(tts, "Parlo con lei? ", "A")
    second = _subtitle(
        "Parlo con lei?",
        [("Parlo", 0, 5, 20, 300), (" ", 5, 6, 300, 320), ("con", 6, 9, 320, 450), (" ", 9, 10, 450, 470),
         ("lei", 10, 13, 470, 700), ("?", 13, 14, 700, 760)],
    )  # fmt: skip
    await _feed(
        tts,
        _event("sentence_start"), _audio("t1", 16000), _audio("t1", 16000, subtitle=BUONGIORNO), _final("t1"),
        _event("sentence_start"), _audio("t2", 8000, subtitle=second), _final("t2"),
        _event("task_flushed"),
    )  # fmt: skip

    # Sentence 1 is 32000 bytes of 8 kHz PCM: 2.0 s, so sentence 2 starts at 2.0 s.
    assert tts.words == [
        ("A", "Buongiorno,", 0.113),
        ("A", "sono", 0.704),
        ("A", "Filippo", 1.13),
        ("A", "di", 1.395),
        ("A", "Acme.", 1.557),
        ("A", "Parlo", 2.1),
        ("A", "con", 2.385),
        ("A", "lei?", 2.55),
    ]


@pytest.mark.asyncio
async def test_pcm_split_on_an_odd_byte_is_reassembled(tts):
    await _say(tts, "Uno. ", "A")
    await _feed(tts, _event("sentence_start"), _audio("t1", 3), _audio("t1", 5))

    assert tts.audio == [("A", 2), ("A", 6)]


@pytest.mark.asyncio
async def test_a_failed_task_ends_every_reply_and_an_empty_balance_disables_the_service(tts):
    await _say(tts, "Uno. ", "A")
    await _feed(
        tts,
        _event(
            "task_failed", base_resp={"status_code": 1008, "status_msg": "insufficient balance"}
        ),
    )

    assert tts.stopped == ["A"]
    assert [e.error for e in tts.errors] == [
        "MiniMax TTS error: 1008 insufficient balance (trace_id=session)"
    ]
    assert not tts.is_usable


@pytest.mark.asyncio
async def test_a_reply_whose_text_cannot_be_sent_is_ended(tts):
    async def closed(_raw):
        raise ConnectionError("socket closed")

    tts._websocket.send = closed
    frames = [frame async for frame in tts.run_tts("Uno. ", "A")]

    assert [type(f) for f in frames] == [ErrorFrame]
    assert tts.stopped == ["A"]
    assert not tts._syntheses


class _FakeStream(_FakeSocket):
    def __init__(self, messages: list[dict]):
        super().__init__()
        self._messages = messages

    async def __aiter__(self):
        for message in self._messages:
            yield json.dumps(message)


@pytest.mark.asyncio
async def test_a_malformed_event_is_skipped_and_the_session_kept(tts):
    await _say(tts, "Uno. ", "A")
    tts._websocket = _FakeStream(
        [
            _event("sentence_start"),
            _event("task_continued", "t1", data={"audio": "not hex"}, is_final=False),
            _audio("t1", 4, subtitle={"text": "Uno.", "timestamped_words": [{"word": "Uno"}]}),
            _audio("t1", 6),
            _event("task_flushed"),
        ]
    )

    await tts._receive_messages()

    assert tts.audio == [("A", 4), ("A", 6)]
    assert tts.stopped == ["A"]


def _unconnected_service(monkeypatch, port: int):
    """A service pointed at a local server, still holding a session MiniMax already closed."""
    tts = MiniMaxTTSService(api_key="test-key", url=f"ws://127.0.0.1:{port}")
    tts._sample_rate = 8000
    tts._websocket = _FakeSocket()
    tts._websocket.state = State.CLOSED
    tts.errors, tts.events = [], []

    async def push_error_frame(error, force_treat_as_permanent=False):
        tts.errors.append(error)

    async def call_event_handler(name, *args):
        tts.events.append(name)

    monkeypatch.setattr(tts, "push_error_frame", push_error_frame)
    monkeypatch.setattr(tts, "_call_event_handler", call_event_handler)
    return tts


@pytest.mark.asyncio
async def test_a_rejected_task_start_leaves_no_socket_behind(monkeypatch):
    closed = asyncio.Event()

    async def reject(websocket):
        await websocket.send(json.dumps(_event("connected_success")))
        await websocket.recv()
        failed = {"status_code": 2054, "status_msg": "voice id not exist"}
        await websocket.send(json.dumps(_event("task_failed", base_resp=failed)))
        await websocket.wait_closed()
        closed.set()

    async with websockets.serve(reject, "127.0.0.1", 0) as server:
        tts = _unconnected_service(monkeypatch, server.sockets[0].getsockname()[1])
        await tts._connect_websocket()
        await asyncio.wait_for(closed.wait(), timeout=5)

    assert tts._websocket is None
    assert tts.events == ["on_connection_error"]
    assert [e.error for e in tts.errors] == [
        "MiniMax TTS error: 2054 voice id not exist (trace_id=session)"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("replies", [[], ["connected_success"]], ids=["connect", "task_start"])
async def test_a_handshake_that_times_out_closes_its_socket(monkeypatch, replies):
    monkeypatch.setattr("pipecat.services.minimax.tts._HANDSHAKE_TIMEOUT_SECS", 0.2)
    closed = asyncio.Event()

    async def stall(websocket):
        for event in replies:
            await websocket.send(json.dumps(_event(event)))
        await websocket.wait_closed()
        closed.set()

    async with websockets.serve(stall, "127.0.0.1", 0) as server:
        tts = _unconnected_service(monkeypatch, server.sockets[0].getsockname()[1])
        await tts._connect_websocket()
        await asyncio.wait_for(closed.wait(), timeout=5)

    assert tts._websocket is None
    assert tts.events == ["on_connection_error"]


async def _fake_minimax(websocket):
    """Just enough of the bidirectional protocol: one sentence per flush, with word subtitles."""
    await websocket.send(json.dumps(_event("connected_success")))
    start = json.loads(await websocket.recv())
    assert start["event"] == "task_start" and start["subtitle_type"] == "word"
    await websocket.send(json.dumps(_event("task_started")))
    buffered = ""
    async for raw in websocket:
        message = json.loads(raw)
        if message["event"] == "task_continue":
            buffered += message["text"]
        elif message["event"] == "task_flush":
            text, buffered = buffered.strip(), ""
            if text:
                pieces, at = [], 0
                for i, word in enumerate(text.split(" ")):
                    begin = text.index(word, at)
                    pieces.append((word, begin, begin + len(word), 100 + 300 * i, 350 + 300 * i))
                    at = begin + len(word)
                await websocket.send(json.dumps(_event("sentence_start")))
                await websocket.send(json.dumps(_audio("t1", 8000)))
                await websocket.send(
                    json.dumps(_audio("t1", 8000, subtitle=_subtitle(text, pieces)))
                )
                await websocket.send(json.dumps(_final("t1")))
                await websocket.send(json.dumps(_event("sentence_end", "t1")))
            await websocket.send(json.dumps(_event("task_flushed")))
        elif message["event"] == "task_cancel":
            await websocket.send(json.dumps(_event("task_canceled")))


@pytest.mark.asyncio
async def test_a_spoken_reply_produces_audio_and_word_frames_end_to_end():
    async with websockets.serve(_fake_minimax, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        # No output transport here to send BotStoppedSpeakingFrame, which lifts the pause.
        tts = MiniMaxTTSService(
            api_key="test-key",
            url=f"ws://127.0.0.1:{port}",
            sample_rate=8000,
            pause_frame_processing=False,
        )
        down, up = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Buongiorno, sono Filippo.")]
        )

    frames = list(down) + list(up)
    assert sum(len(f.audio) for f in frames if isinstance(f, TTSAudioRawFrame)) == 16000
    assert [f.text for f in frames if isinstance(f, TTSTextFrame)] == [
        "Buongiorno,",
        "sono",
        "Filippo.",
    ]
    assert not [f for f in frames if isinstance(f, ErrorFrame)]
