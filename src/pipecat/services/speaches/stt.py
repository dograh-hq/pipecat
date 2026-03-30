"""Speaches STT Service — streams audio over WebSocket to a Speaches server."""

from dataclasses import dataclass
from typing import Optional

from pipecat.services.dograh.stt import DograhSTTService, DograhSTTSettings


@dataclass
class SpeachesSTTSettings(DograhSTTSettings):
    """Settings for Speaches STT service."""

    pass


class SpeachesSTTService(DograhSTTService):
    """Speaches STT service that streams audio over WebSocket to a Speaches server."""

    Settings = SpeachesSTTSettings

    def __init__(
        self,
        *,
        api_key: str = "none",
        base_url: str = "ws://localhost:8000/v1",
        ws_path: str = "/stt/ws",
        settings: Optional[SpeachesSTTSettings] = None,
        **kwargs,
    ):
        """Initialize the Speaches STT service.

        Args:
            api_key: API key for authentication.
            base_url: Base WebSocket URL of the Speaches server.
            ws_path: WebSocket path for STT streaming.
            settings: Optional service settings.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            ws_path=ws_path,
            settings=settings,
            **kwargs,
        )
