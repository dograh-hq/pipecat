"""Synthetic control syntax must never become speech or conversation text."""

import pytest

from pipecat.utils.text.spoken_text_filter import SpokenTextFilter
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter

CASES = [
    ('Goodbye.x{ "call_id": "default_api:', "Goodbye."),
    ('All set.op_update_record_84_a1:{"', "All set."),
    ("Goodbye. account_code: null, record_key: null.", "Goodbye."),
    ("Calling {{Example Workshop}} now.", "Calling Example Workshop now."),
    ("Calling {{Example Co. West}} now.", "Calling Example Co. West now."),
    ("Please ask {{contact_name}}.", "Please ask ."),
    ("{{profile.contact.name}}", ""),
    ("phase:no_pending_actions route:offer_other_slot|request_reference|finish_session", ""),
    ('Ready. {"debug": {"step": "Wait. Then proceed."}} Next question?', "Ready. Next question?"),
    ('Ready. ```json\n{"action": "update_record"}\n``` Next question?', "Ready. Next question?"),
    ('Hello.<function=update_record>{"key":"a"}</function> Welcome.', "Hello. Welcome."),
    ("Hello.<function=update_record/>", "Hello."),
    ("<think>Private planning. More planning.</think>Hello.", "Hello."),
    ("Hello.<call:default_api:finish_session/>", "Hello."),
    ("Hello.<|im_end|>", "Hello."),
    ('Ready. trace_label: "INTERNAL VALUE", retry_count: 3. Next?', "Ready. Next?"),
    ('Ready. [{"name":"update_record","arguments":{}}] Next?', "Ready. Next?"),
    ("</thinking>Hello.", "Hello."),
    ("Done. op_update_record_84_a1:", "Done."),
    ("Goodbye.<fun", "Goodbye."),
    ("<think", ""),
    ('Done. "item_code": null, "request_id": "INTERNAL".', "Done."),
    ("Ready. [true, false, null] Next?", "Ready. Next?"),
    ("Ready. [-1, -2.5, -3e2] Next?", "Ready. Next?"),
    ('Ready. [null, [false, -2], {"trace": true}] Next?', "Ready. Next?"),
    ("Ready. [", "Ready."),
    ("Ready. [  ", "Ready."),
    ("Ready. [tru", "Ready."),
    ("Ready. [-", "Ready."),
    (
        "Ready. <call:default_api:finish_session>Private instructions.</call:default_api:finish_session> Welcome.",
        "Ready. Welcome.",
    ),
    ("Ready. <call:default_api:finish_session>Private instructions.", "Ready."),
    ("</call:default_api:finish_session>Hello.", "Hello."),
]


def normalized(text):
    return " ".join(text.split())


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "expected"), CASES)
async def test_complete_utterance(text, expected):
    assert normalized(await SpokenTextFilter().filter(text)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "expected"), CASES)
async def test_every_stream_split(text, expected):
    for split in range(len(text) + 1):
        filter = SpokenTextFilter()
        spoken = await filter.filter_stream(text[:split])
        spoken += await filter.filter_stream(text[split:])
        spoken += await filter.flush()
        assert normalized(spoken) == expected, (split, spoken)
    filter = SpokenTextFilter()
    spoken = "".join([await filter.filter_stream(c) for c in text]) + await filter.flush()
    assert normalized(spoken) == expected


@pytest.mark.asyncio
async def test_interrupt_and_response_end_clear_incomplete_syntax():
    filter = SpokenTextFilter()
    assert await filter.filter_stream('{"debug":"unfinished') == ""
    await filter.handle_interruption()
    assert await filter.filter_stream("Hello. ") == "Hello. "
    assert await filter.flush() == ""
    assert await filter.filter_stream("{{unfinished") == ""
    assert await filter.flush() == ""
    assert await filter.filter_stream("Welcome. ") == "Welcome. "


@pytest.mark.asyncio
async def test_complete_utterances_do_not_consume_a_pending_response():
    filter = SpokenTextFilter()
    assert await filter.filter_stream("Calling {{Example") == "Calling "
    assert await filter.filter("One moment. {{trace_id}}") == "One moment. "
    assert await filter.filter_stream(" Workshop}} now.") == "Example Workshop now."
    assert await filter.flush() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Note: the office opens at 9:30.",
        "Answer: true. The opposite answer: false.",
        "The price is 19.95, with a 5% discount.",
        "Email sample_user@example.org or visit https://example.org/a_b.",
        "Your username is sample_user, correct?",
        "The option is called record_key in the settings.",
        "We meet on [Monday] at (around) ten.",
        "Choose [trueblue], [falsehood], [nullable], or [-placeholder].",
        "The value is < five, and 5 > 3.",
        "Bonjour. ¿En qué puedo ayudar? こんにちは。",
    ],
)
async def test_ordinary_speech_is_preserved(text):
    assert await SpokenTextFilter().filter(text) == text
    filter = SpokenTextFilter()
    streamed = "".join([await filter.filter_stream(c) for c in text]) + await filter.flush()
    assert streamed == text


@pytest.mark.asyncio
async def test_custom_patterns_cover_chunk_boundaries_and_reset():
    filter = SpokenTextFilter(custom_patterns=[r"INTERNAL\s+VALUE"])
    assert await filter.filter_stream("Hello INTERNAL ") == ""
    assert await filter.filter_stream("VALUE world.") == ""
    assert normalized(await filter.flush()) == "Hello world."
    assert await filter.filter("INTERNAL VALUE") == " "
    await filter.update_settings({"custom_patterns": []})
    assert await filter.filter_stream("Hello. ") == "Hello. "


@pytest.mark.asyncio
async def test_unterminated_fragment_is_bounded_and_next_turn_is_clean():
    filter = SpokenTextFilter()
    assert await filter.filter_stream('{"debug":"' + "x" * 20000) == ""
    assert await filter.filter_stream('"} trailing control') == ""
    assert await filter.flush() == ""
    assert await filter.filter_stream("Welcome. ") == "Welcome. "


@pytest.mark.asyncio
@pytest.mark.parametrize("patterns", [None, [], ["PRIVATE"]])
async def test_settings_changes_take_effect_after_the_active_stream(patterns):
    filter = SpokenTextFilter(custom_patterns=["INTERNAL"])
    assert await filter.filter_stream("First INTERNAL ") == ""
    await filter.update_settings({"custom_patterns": patterns})
    assert await filter.filter_stream("second PRIVATE.") == ""
    assert normalized(await filter.flush()) == "First second PRIVATE."
    next_reply = await filter.filter_stream("Next INTERNAL PRIVATE.") + await filter.flush()
    assert normalized(next_reply) == (
        "Next INTERNAL PRIVATE." if not patterns else "Next INTERNAL ."
    )


@pytest.mark.asyncio
async def test_enabling_patterns_mid_stream_does_not_filter_only_the_suffix():
    filter = SpokenTextFilter()
    output = await filter.filter_stream("First INTERNAL ")
    await filter.update_settings({"custom_patterns": [r"INTERNAL\s+VALUE"]})
    output += await filter.filter_stream("VALUE last.") + await filter.flush()
    assert output == "First INTERNAL VALUE last."
    output = await filter.filter_stream("Next INTERNAL VALUE reply.") + await filter.flush()
    assert normalized(output) == "Next reply."


@pytest.mark.asyncio
async def test_interruption_applies_pending_settings_without_releasing_old_text():
    filter = SpokenTextFilter(custom_patterns=["INTERNAL"])
    assert await filter.filter_stream("Discard this INTERNAL text.") == ""
    await filter.update_settings({"custom_patterns": [], "preserve_whitespace": False})
    await filter.handle_interruption()
    assert await filter.filter_stream("New reply.") + await filter.flush() == "New reply."


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_class", [SpokenTextFilter, XMLFunctionTagFilter])
@pytest.mark.parametrize("preserve", [True, False])
@pytest.mark.parametrize("tag", ["<function=sample/>", "<function=sample>hidden</function>"])
async def test_xml_removal_preserves_legacy_separator_policy(filter_class, preserve, tag):
    text = f"a{tag}b"
    expected = "a b" if preserve else "ab"
    filter = filter_class(preserve_whitespace=preserve)
    assert await filter.filter(text) == expected
    for split in range(len(text) + 1):
        output = await filter.filter_stream(text[:split])
        output += await filter.filter_stream(text[split:]) + await filter.flush()
        assert output == expected, (split, output)


@pytest.mark.asyncio
async def test_whitespace_setting_is_frozen_for_stream_but_independent_speech_uses_update():
    filter = SpokenTextFilter(custom_patterns=["HIDDEN"])
    assert await filter.filter_stream("a<function=sample>") == ""
    await filter.update_settings({"preserve_whitespace": False})
    assert await filter.filter("x<function=sample/>yHIDDENz") == "xyz"
    assert await filter.filter_stream("hidden</function>bHIDDENc") == ""
    assert await filter.flush() == "a b c"
    assert await filter.filter_stream("x<function=sample/>yHIDDENz") == ""
    assert await filter.flush() == "xyz"


@pytest.mark.asyncio
async def test_custom_regex_buffer_preserves_long_speech_and_crosses_the_fragment_limit():
    filter = SpokenTextFilter(custom_patterns=[r"INTERNAL\s+VALUE"])
    prefix = "Ordinary speech. " * 2000
    assert await filter.filter_stream(prefix + "INTERNAL ") == ""
    assert await filter.filter_stream("VALUE is removed. " + prefix) == ""
    assert normalized(await filter.flush()) == normalized(prefix + "is removed. " + prefix)
