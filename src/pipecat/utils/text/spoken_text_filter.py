"""Remove serialized control data from speech without executing actions."""

import json
import re
from collections.abc import Mapping
from typing import Any

from pipecat.utils.text.base_text_filter import BaseTextFilter


class SpokenTextSanitizer:
    """Incrementally separate speech from structured model output.

    Holds unfinished lexical tokens and structured fragments across chunks.
    Tool names and business field names are deliberately not part of the grammar.
    """

    MAX_PENDING_CHARS = 16384
    _TAGS = (
        "<function=",
        "<call:",
        "<think",
        "<thought",
        "<thinking",
        "<|",
        "</think",
        "</thought",
        "</thinking",
    )
    _ASSIGNMENT = re.compile(r'"?([\w]+)"?\s*:\s*(.*)', re.DOTALL)
    _CONTROL_VALUE = re.compile(
        r"(?:null|true|false|undefined|None|[\w]+_[\w:|.-]*)(?=$|[\s,;.!?])"
    )
    _VARIABLE = re.compile(r"(?:[a-z][a-z0-9]*|[\w]+[_.][\w.]*)(?:\s*\|.*)?$")

    def __init__(self):
        """Initialize an empty response buffer."""
        self.reset()

    def reset(self) -> None:
        """Discard pending text at an interruption or a new utterance."""
        self._pending = ""
        self._discard = False
        self._after_control = False

    def feed(self, text: str) -> str:
        """Release safe text, retaining incomplete syntax for the next chunk."""
        if self._discard:
            return ""
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> str:
        """Release a final word and discard unterminated structured fragments."""
        result = self._drain(final=True)
        self.reset()
        return result

    @staticmethod
    def _structure_end(text: str) -> int | None:
        stack, quoted, escaped = [], False, False
        for index, char in enumerate(text):
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                stack.append("}" if char == "{" else "]")
            elif char in "}]":
                if not stack or char != stack.pop():
                    return None
                if not stack:
                    return index + 1
        return None

    def _drain(self, *, final: bool) -> str:
        output = []
        while self._pending and not self._discard:
            text = self._pending
            if text[0].isspace():
                output.append(text[0])
                self._pending = text[1:]
                continue
            if text[0] in "\ufeff\u200b\u2060}":
                self._pending = text[1:]
                continue
            if self._after_control and text[0] in ",;.!?":
                self._pending = text[1:]
                continue

            if text.startswith("{{"):
                end = text.find("}}", 2)
                if end < 0:
                    if final:
                        self._pending = ""
                    break
                value = text[2:end].strip()
                # Keep a rendered display name, but never pronounce a variable path.
                if self._VARIABLE.fullmatch(value) or any(c in value for c in "{}<>`"):
                    value = ""
                self._pending = value + text[end + 2 :]
                continue
            if text.startswith("[") and not final and not text[1:].strip():
                break
            if text.startswith("{") or re.match(r'\[\s*[\[\{"\d\]]', text):
                if text == "{" and not final:
                    break
                end = self._structure_end(text)
                if end is None and not final:
                    break
                self._pending = text[end:] if end is not None else ""
                self._after_control = True
                continue

            if text.startswith("`"):
                if text in ("`", "``") and not final:
                    break
                fence = "```" if text.startswith("```") else "`"
                end = text.find(fence, len(fence))
                if end < 0 and not final:
                    break
                self._pending = text[end + len(fence) :] if end >= 0 else ""
                self._after_control = True
                continue

            if text.startswith("<"):
                lowered = text.lower()
                if any(tag.startswith(lowered) for tag in self._TAGS):
                    if final:
                        self._pending = ""
                    break
                tag = re.match(r"<(/?)(function=|call:|think\b|thinking\b|thought\b|\|)", lowered)
                if tag:
                    end = text.find(">")
                    if end < 0 and not final:
                        break
                    if (
                        end >= 0
                        and not tag[1]
                        and tag[2] not in ("call:", "|")
                        and text[end - 1] != "/"
                    ):
                        name = "function" if tag[2] == "function=" else tag[2]
                        closing = re.search(rf"</{name}\s*>", lowered[end + 1 :])
                        if closing is None and not final:
                            break
                        end = end + closing.end() if closing else len(text) - 1
                    self._pending = text[end + 1 :] if end >= 0 else ""
                    self._after_control = True
                    continue

            namespace = re.match(r"(?:[\w-]+:)?default_api[:.][\w.-]*", text)
            if namespace:
                if namespace.end() == len(text) and not final:
                    break
                self._pending = text[namespace.end() :]
                self._after_control = True
                continue

            # A lexical prefix glued to an object is serialized control syntax.
            prefix = re.match(r"[\w:-]+(?=\{)", text)
            if prefix:
                self._pending = text[prefix.end() :]
                self._after_control = True
                continue

            assignment = self._ASSIGNMENT.match(text)
            if assignment:
                value = assignment[2]
                if not value and not final:
                    break
                if not final and value and not re.search(r"[\s,;!?]", value):
                    # A partial ordinary-looking value can still become a control token.
                    break
                control = self._CONTROL_VALUE.match(value)
                if control and ("_" in assignment[1] or "_" in control[0]):
                    end = assignment.start(2) + control.end()
                    if end == len(text) and not final:
                        break
                    self._pending = text[end:]
                    self._after_control = True
                    continue
                if "_" in assignment[1] and value.startswith(('"', "{", "[")):
                    try:
                        _, length = json.JSONDecoder().raw_decode(value)
                    except ValueError:
                        if not final:
                            break
                        length = len(value)
                    self._pending = value[length:]
                    self._after_control = True
                    continue
                if "_" in assignment[1] and re.match(r"-?\d", value):
                    number = re.match(r"-?\d+(?:\.\d+)?(?=$|[\s,;.!?])", value)
                    if number:
                        self._pending = value[number.end() :]
                        self._after_control = True
                        continue
                if not value and "_" in assignment[1]:
                    self._pending = ""
                    continue

            # Hold the final token so a following ':' or '{' can classify it.
            boundary = re.search(r"[\s{}<>`!?;,.]", text)
            if boundary:
                end = boundary.start() + (boundary[0] in " \t\n\r!?;,.")
                end = max(end, 1)
            elif final:
                end = len(text)
            else:
                break
            output.append(text[:end])
            self._after_control = False
            self._pending = text[end:]

        if len(self._pending) > self.MAX_PENDING_CHARS:
            self._pending = ""
            self._discard = True
        return "".join(output)


class SpokenTextFilter(BaseTextFilter):
    """Sanitize speech before aggregation and context/word-timing bookkeeping."""

    filter_before_aggregation = True

    def __init__(
        self, *, custom_patterns: list[str] | None = None, preserve_whitespace: bool = True
    ):
        """Initialize the streaming sanitizer.

        Args:
            custom_patterns: Additional regexes evaluated on complete utterances.
                When supplied, streaming text is buffered until response end.
            preserve_whitespace: Replace custom-pattern matches with a space
                instead of removing them without a separator.
        """
        self._sanitizer = SpokenTextSanitizer()
        self._preserve_whitespace = preserve_whitespace
        self._patterns = [re.compile(p, re.DOTALL | re.IGNORECASE) for p in custom_patterns or []]
        self._custom_pending = ""
        self._custom_overflow = False

    def _custom(self, text: str) -> str:
        for pattern in self._patterns:
            text = pattern.sub(" " if self._preserve_whitespace else "", text)
        return text

    async def filter(self, text: str) -> str:
        """Sanitize a complete, independent utterance."""
        sanitizer = SpokenTextSanitizer()
        return self._custom(sanitizer.feed(text) + sanitizer.finish())

    async def filter_stream(self, text: str) -> str:
        """Sanitize one chunk of the current response."""
        clean = self._sanitizer.feed(text)
        if not self._patterns:
            return clean
        # Arbitrary regexes need the complete utterance to match across chunks.
        if not self._custom_overflow:
            self._custom_pending += clean
            if len(self._custom_pending) > SpokenTextSanitizer.MAX_PENDING_CHARS:
                self._custom_overflow = True
                self._custom_pending = ""
        return ""

    async def flush(self) -> str:
        """Finish and reset the current response."""
        text = self._custom_pending + self._sanitizer.finish()
        text = "" if self._custom_overflow else self._custom(text)
        self._custom_pending = ""
        self._custom_overflow = False
        return text

    async def handle_interruption(self):
        """Discard unfinished syntax and custom-pattern buffers."""
        self._sanitizer.reset()
        self._custom_pending = ""
        self._custom_overflow = False

    async def update_settings(self, settings: Mapping[str, Any]):
        """Update custom regexes and their whitespace replacement policy."""
        if "preserve_whitespace" in settings:
            self._preserve_whitespace = settings["preserve_whitespace"]
        if "custom_patterns" in settings:
            self._patterns = [
                re.compile(p, re.DOTALL | re.IGNORECASE) for p in settings["custom_patterns"]
            ]
