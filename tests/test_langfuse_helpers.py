#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import unittest
from unittest.mock import patch

from pipecat.utils.tracing.langfuse_helpers import mark_trace_public


class _RecordingSpan:
    """Span stand-in that records the attributes set on it."""

    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class TestMarkTracePublic(unittest.TestCase):
    """Tests for the LANGFUSE_TRACES_PUBLIC gate on mark_trace_public()."""

    def _mark(self, env):
        span = _RecordingSpan()
        # clear=True so an inherited flag can't leak into the test.
        with patch.dict(os.environ, env, clear=True):
            mark_trace_public(span)
        return span

    def test_private_by_default(self):
        """Traces stay private when the flag is unset."""
        span = self._mark({})
        self.assertNotIn("langfuse.trace.public", span.attributes)

    def test_private_when_flag_disabled(self):
        """An explicit falsy value keeps the trace private."""
        span = self._mark({"LANGFUSE_TRACES_PUBLIC": "false"})
        self.assertNotIn("langfuse.trace.public", span.attributes)

    def test_public_when_flag_enabled(self):
        """The trace is marked public only when the flag opts in."""
        for value in ("1", "true", "TRUE", "yes"):
            with self.subTest(value=value):
                span = self._mark({"LANGFUSE_TRACES_PUBLIC": value})
                self.assertIs(span.attributes.get("langfuse.trace.public"), True)


if __name__ == "__main__":
    unittest.main()
