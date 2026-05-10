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


def test_configure_fields_did_not_lose_existing_entries():
    """Regression guard: adding `language_hints` must not displace the
    pre-existing configurable fields. Pin the full set so any future deletion
    fails loudly here rather than silently dropping a configurable field."""
    expected = {
        "keyterm",
        "eot_threshold",
        "eager_eot_threshold",
        "eot_timeout_ms",
        "language_hints",
    }
    assert DeepgramFluxSTTBase._CONFIGURE_FIELDS == expected


def test_build_query_string_preserves_hint_order():
    """Order matters as a contract: callers may pass primary-then-fallback codes
    (e.g. 'hi','en') and Deepgram processes them in order. Pin emission order."""
    settings = _make_settings(language_hints=["hi", "en", "es"])
    base = _make_stub(settings)

    qs = base._build_query_string()

    # Find positions of each hint in the query string and assert ascending.
    positions = [qs.find(f"language_hint={code}") for code in ("hi", "en", "es")]
    assert all(p >= 0 for p in positions), qs
    assert positions == sorted(positions), f"hint order not preserved: {qs}"


def test_build_query_string_url_encodes_special_chars():
    """A hint containing reserved URL chars must be percent-escaped, not
    interpolated raw — otherwise a stray '&' or space would corrupt the
    query string structure. Deepgram will reject the resulting code, but the
    URL itself must remain well-formed."""
    settings = _make_settings(language_hints=["en uncommon"])
    base = _make_stub(settings)

    qs = base._build_query_string()

    # urlencode emits '+' for space (form-encoded), and the literal space must
    # not appear inside the language_hint value.
    assert "language_hint=en+uncommon" in qs or "language_hint=en%20uncommon" in qs
    assert "language_hint=en uncommon" not in qs


def test_build_query_string_single_hint():
    """Single-element list still uses the repeated-singular form (no special
    casing as a scalar)."""
    settings = _make_settings(language_hints=["hi"])
    base = _make_stub(settings)

    qs = base._build_query_string()
    parts = qs.split("&")

    assert "language_hint=hi" in parts
    assert sum(1 for p in parts if p.startswith("language_hint=")) == 1


def test_build_query_string_keyterm_and_language_hints_coexist():
    """Both list-emitters write to the same query string; neither must clobber
    the other or interleave incorrectly."""
    settings = _make_settings(keyterm=["nophari", "samvadatmaka"], language_hints=["en", "hi"])
    base = _make_stub(settings)

    qs = base._build_query_string()
    parts = qs.split("&")

    assert "keyterm=nophari" in parts
    assert "keyterm=samvadatmaka" in parts
    assert "language_hint=en" in parts
    assert "language_hint=hi" in parts


def test_build_query_string_is_idempotent():
    """Building the query string must be a pure function of settings — calling
    it twice without mutating settings produces identical output. Catches
    accidental mutation of self._settings or self._tag during emission."""
    settings = _make_settings(language_hints=["en", "hi"], keyterm=["nophari"])
    base = _make_stub(settings)

    first = base._build_query_string()
    second = base._build_query_string()

    assert first == second


@pytest.mark.asyncio
async def test_send_configure_with_multiple_fields_groups_correctly():
    """A realistic mid-stream update touches several fields at once. The
    Configure body must group thresholds under the `thresholds` key while
    keeping `keyterms` and `language_hints` at the top level."""
    settings = _make_settings(
        keyterm=["nophari"],
        language_hints=["en", "es"],
        eot_threshold=0.8,
        eager_eot_threshold=0.5,
    )
    instance = _make_stub(settings)
    instance._transport_send_json = AsyncMock()

    await instance._send_configure({"keyterm", "language_hints", "eot_threshold", "eager_eot_threshold"})

    sent = instance._transport_send_json.await_args.args[0]
    assert sent["type"] == "Configure"
    assert sent["keyterms"] == ["nophari"]
    assert sent["language_hints"] == ["en", "es"]
    assert sent["thresholds"] == {"eot_threshold": 0.8, "eager_eot_threshold": 0.5}


@pytest.mark.asyncio
async def test_send_configure_copies_language_hints_list():
    """The Configure body must contain a *copy* of the hints list, not a live
    reference into self._settings — so a caller that mutates the original
    after the message is sent doesn't retroactively change the wire payload."""
    original = ["en", "hi"]
    settings = _make_settings(language_hints=original)
    instance = _make_stub(settings)
    instance._transport_send_json = AsyncMock()

    await instance._send_configure({"language_hints"})
    sent_list = instance._transport_send_json.await_args.args[0]["language_hints"]

    # Mutating the original list must not alter what was sent on the wire.
    original.append("es")
    assert sent_list == ["en", "hi"]
    assert sent_list is not original
