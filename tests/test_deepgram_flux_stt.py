#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for DeepgramFluxSTTBase language_hints wire format.

Covers two distinct wire shapes Deepgram uses for the same logical concept:
- Connect-time URL: singular ``language_hint=<code>`` repeated per code.
- Mid-stream Configure message: plural ``language_hints`` JSON array.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.services.deepgram.flux.base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
)
from pipecat.transcriptions.language import Language


class _StubFlux(DeepgramFluxSTTBase):
    """Concrete subclass that stubs the abstract transport interface so we can
    instantiate via __new__ and test the wire-format helpers in isolation."""

    async def _transport_send_audio(self, audio: bytes):
        pass

    async def _transport_send_json(self, message: dict):
        pass

    def _transport_is_active(self) -> bool:
        return False

    async def _connect(self):
        pass

    async def _disconnect(self):
        pass

    async def run_stt(self, audio):
        if False:
            yield None


def _make_settings(**overrides) -> DeepgramFluxSTTSettings:
    """Construct a fully-populated Settings instance for query/Configure tests.

    Iterable fields (keyterm, language_hints) and threshold fields must be set
    explicitly because ``_build_query_string`` iterates the lists and emits any
    threshold that is ``is not None``. The websocket service's __init__ does
    this in production; we replicate the same shape here.
    """
    # `language` is inherited from STTSettings and tags resulting
    # TranscriptionFrames; it is not read by _build_query_string or
    # _send_configure, so any concrete enum value works for these tests.
    defaults = dict(
        model="flux-general-multi",
        language=Language.EN,
        eager_eot_threshold=None,
        eot_threshold=None,
        eot_timeout_ms=None,
        keyterm=[],
        language_hints=[],
        min_confidence=None,
    )
    defaults.update(overrides)
    return DeepgramFluxSTTSettings(**defaults)


def _make_stub(settings: DeepgramFluxSTTSettings) -> _StubFlux:
    """Bypass __init__ and wire the minimum attributes the helpers under test
    read (sample_rate, _name, _encoding, _tag, _mip_opt_out)."""
    instance = _StubFlux.__new__(_StubFlux)
    instance._settings = settings
    instance._encoding = "linear16"
    instance._tag = []
    instance._mip_opt_out = None
    instance._sample_rate = 16000  # backing attr; sample_rate is a read-only property
    instance._name = "TestFlux"  # used by __str__ in debug logging
    return instance


def test_build_query_string_emits_repeated_language_hint():
    """URL form is singular-per-code, repeated. Order is preserved."""
    settings = _make_settings(language_hints=["en", "hi"])
    base = _make_stub(settings)

    qs = base._build_query_string()
    parts = qs.split("&")

    # Both codes present, each with the singular `language_hint` key.
    assert "language_hint=en" in parts
    assert "language_hint=hi" in parts
    # Plural form must NOT appear in the URL — that shape is for Configure only.
    assert all(not p.startswith("language_hints=") for p in parts), qs


def test_build_query_string_omits_language_hint_when_empty():
    """Empty list means full auto-detect; the URL must not mention language_hint."""
    settings = _make_settings(language_hints=[])
    base = _make_stub(settings)

    qs = base._build_query_string()

    assert "language_hint" not in qs


def test_build_query_string_url_encodes_codes_with_subtags():
    """BCP-47 locale subtags like 'en-GB' must round-trip safely through urlencode."""
    settings = _make_settings(language_hints=["en-GB", "pt-BR"])
    base = _make_stub(settings)

    qs = base._build_query_string()

    # urlencode percent-escapes the hyphen to '-' (no escape needed) but the test
    # asserts the literal expected substrings to lock in wire format.
    assert "language_hint=en-GB" in qs
    assert "language_hint=pt-BR" in qs


@pytest.mark.asyncio
async def test_send_configure_uses_plural_language_hints():
    """Mid-stream update uses the PLURAL `language_hints` JSON array."""
    settings = _make_settings(language_hints=["en", "es", "fr"])
    instance = _make_stub(settings)
    instance._transport_send_json = AsyncMock()

    await instance._send_configure({"language_hints"})

    instance._transport_send_json.assert_awaited_once()
    sent = instance._transport_send_json.await_args.args[0]
    assert sent["type"] == "Configure"
    assert sent["language_hints"] == ["en", "es", "fr"]
    # Other configurable fields should not leak when only language_hints changed.
    assert "keyterms" not in sent
    assert "thresholds" not in sent


@pytest.mark.asyncio
async def test_send_configure_clears_hints_with_empty_list():
    """Per Deepgram docs, an empty array clears existing hints."""
    settings = _make_settings(language_hints=[])
    instance = _make_stub(settings)
    instance._transport_send_json = AsyncMock()

    await instance._send_configure({"language_hints"})

    sent = instance._transport_send_json.await_args.args[0]
    assert sent["language_hints"] == []


@pytest.mark.asyncio
async def test_send_configure_omits_language_hints_when_not_in_changed_set():
    """If only keyterm changed, the Configure body must not include language_hints."""
    settings = _make_settings(keyterm=["nophari"], language_hints=["en"])
    instance = _make_stub(settings)
    instance._transport_send_json = AsyncMock()

    await instance._send_configure({"keyterm"})

    sent = instance._transport_send_json.await_args.args[0]
    assert "language_hints" not in sent
    assert sent["keyterms"] == ["nophari"]


def test_configure_fields_membership():
    """language_hints must be in the configurable set so settings updates flow
    through _send_configure rather than triggering _warn_unhandled_updated_settings."""
    assert "language_hints" in DeepgramFluxSTTBase._CONFIGURE_FIELDS
