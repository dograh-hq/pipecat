from enum import Enum


class EndTaskReason(Enum):
    CALL_DURATION_EXCEEDED = "call_duration_exceeded"
    USER_IDLE_MAX_DURATION_EXCEEDED = "user_idle_max_duration_exceeded"
    USER_HANGUP = "user_hangup"
    USER_QUALIFIED = "user_qualified"
    SYSTEM_CANCELLED = "system_cancelled"
    SYSTEM_CONNECT_ERROR = "system_connect_error"
    UNKNOWN = "unknown"
