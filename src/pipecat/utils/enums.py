from enum import Enum


class EndTaskReason(Enum):
    CALL_DURATION_EXCEEDED = "call_duration_exceeded"
    USER_IDLE = "user_idle"
    USER_CANCELLED = "user_cancelled"
    BOT_CANCELLED = "bot_cancelled"
    SYSTEM_CANCELLED = "system_cancelled"
