"""Speaches self-hosted AI services for Pipecat.

Thin wrappers around existing services with defaults pointing
at a local Speaches server.
"""

from pipecat.services.speaches.llm import SpeachesLLMService, SpeachesLLMSettings
from pipecat.services.speaches.stt import SpeachesSTTService, SpeachesSTTSettings
from pipecat.services.speaches.tts import SpeachesTTSService, SpeachesTTSSettings

__all__ = [
    "SpeachesLLMService",
    "SpeachesLLMSettings",
    "SpeachesSTTService",
    "SpeachesSTTSettings",
    "SpeachesTTSService",
    "SpeachesTTSSettings",
]
