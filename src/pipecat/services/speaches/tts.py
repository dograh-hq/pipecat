"""Speaches TTS Service — uses the OpenAI-compatible /v1/audio/speech endpoint."""

from dataclasses import dataclass
from typing import Optional

from pipecat.services.openai.tts import OpenAITTSService, OpenAITTSSettings


@dataclass
class SpeachesTTSSettings(OpenAITTSSettings):
    """Settings for Speaches TTS service."""

    pass


class SpeachesTTSService(OpenAITTSService):
    """Speaches TTS service using the OpenAI-compatible audio speech endpoint."""

    Settings = SpeachesTTSSettings

    def __init__(
        self,
        *,
        api_key: str = "none",
        base_url: str = "http://localhost:8000/v1",
        sample_rate: int = 24000,
        settings: Optional[SpeachesTTSSettings] = None,
        **kwargs,
    ):
        """Initialize the Speaches TTS service.

        Args:
            api_key: API key for authentication.
            base_url: Base URL of the Speaches-compatible endpoint.
            sample_rate: Audio sample rate in Hz.
            settings: Optional service settings.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            sample_rate=sample_rate,
            settings=settings,
            **kwargs,
        )
