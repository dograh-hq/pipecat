#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline execution context variables.

This module provides context variables that are used to track the current
execution context across the pipeline, including workflow run IDs and
turn numbers. These context variables are automatically propagated through
async contexts.
"""

import contextvars
from typing import Optional

# Context variable for tracking the current workflow run ID across the pipeline
run_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("run_id", default=None)

# Context variable for tracking the current turn number in a conversation
turn_var: contextvars.ContextVar[int] = contextvars.ContextVar("turn", default=0)

# Context variable for tracking the current call SID (for telephony calls)
call_sid_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("call_sid", default=None)



def get_current_run_id() -> Optional[str]:
    """Get the current workflow run ID from the context.

    Returns:
        The current workflow run ID or None if not set.
    """
    return run_id_var.get()


def set_current_run_id(run_id: Optional[str | int]) -> None:
    """Set the current workflow run ID in the context.

    Args:
        run_id: The workflow run ID to set (string or int), or None to clear it.
            If an int is provided, it will be converted to string.
    """
    if run_id is not None:
        run_id = str(run_id)
    run_id_var.set(run_id)


def get_current_turn() -> int:
    """Get the current turn number from the context.

    Returns:
        The current turn number, defaulting to 0 if not set.
    """
    return turn_var.get()


def set_current_turn(turn: int) -> None:
    """Set the current turn number in the context.

    Args:
        turn: The turn number to set.
    """
    turn_var.set(turn)


def get_current_call_sid() -> Optional[str]:
    """Get the current call SID from the context.

    Returns:
        The current call SID or None if not set.
    """
    return call_sid_var.get()


def set_current_call_sid(call_sid: Optional[str]) -> None:
    """Set the current call SID in the context.

    Args:
        call_sid: The call SID to set, or None to clear it.
    """
    call_sid_var.set(call_sid)
