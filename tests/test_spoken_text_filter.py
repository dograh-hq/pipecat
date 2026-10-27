"""Synthetic control syntax must never become speech or conversation text."""

import pytest

from pipecat.utils.text.spoken_text_filter import SpokenTextFilter

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
