#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dograh Flux STT service: Deepgram Flux turn detection proxied via the Dograh MPS."""

from urllib.parse import urlencode

from loguru import logger

from pipecat.frames.frames import StartFrame
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.flux.stt_base import DeepgramFluxSTTSettings
from pipecat.services.dograh.mps_billing import (
    MPS_BILLING_VERSION_KEY,
    MPS_BILLING_VERSION_V2,
    get_correlation_id,
)

__all__ = [
    "DograhFluxSTTService",
    "DeepgramFluxSTTSettings",
]


class DograhFluxSTTService(DeepgramFluxSTTService):
    """Dograh Flux speech-to-text service.

    Uses the upstream Flux connection lifecycle, including background reconnects,
    connection confirmation and partial-turn completion. The MPS endpoint uses
    Bearer authentication, carries billing/correlation in the query string and
    adds proxy errors to the native Flux protocol.

    Event handlers available (in addition to base events):

    - on_start_of_turn(service, transcript): start of speech detected
    - on_end_of_turn(service, transcript): end of turn (EOT) detected
    - on_eager_end_of_turn(service, transcript): end of turn predicted (EagerEOT)
    - on_turn_resumed(service): user resumed speaking after EagerEOT

    Example::

        @stt.event_handler("on_end_of_turn")
        async def on_end_of_turn(service, transcript):
            ...
    """

    Settings = DeepgramFluxSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://services.dograh.com",
        ws_path: str = "/api/v1/stt/flux",
        correlation_id: str | None = None,
        sample_rate: int | None = None,
        flux_encoding: str = "linear16",
        mip_opt_out: bool | None = None,
        tag: list | None = None,
        should_interrupt: bool = False,
        watchdog_min_timeout: float = 0.5,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Dograh Flux STT service.

        Args:
            api_key: Dograh API key for authentication (sent as a Bearer token).
            base_url: WebSocket base URL for the Dograh MPS. Defaults to
                "wss://services.dograh.com".
            ws_path: WebSocket path for the Flux STT endpoint. Defaults to
                "/api/v1/stt/flux".
            correlation_id: Optional server-generated correlation ID for MPS
                billing v2. Pass this explicitly for the initial connection,
                which Pipecat 1.8 opens during setup. StartFrame metadata is
                available only to later reconnects.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            flux_encoding: Audio encoding format required by Flux. Must be
                "linear16" (signed little-endian 16-bit PCM).
            mip_opt_out: Opt out of the Deepgram Model Improvement Program.
            tag: Tags to label requests for identification during usage reporting.
            should_interrupt: Whether the service interrupts the bot when Flux
                detects user speech. Defaults to False so the user-turn
                aggregator owns interruption (via the external turn strategies).
            watchdog_min_timeout: Minimum idle timeout before sending silence to
                prevent dangling turns. Defaults to 0.5.
            settings: Runtime-updatable Flux settings.
            **kwargs: Additional arguments passed to the parent classes.
        """
        default_settings = self.Settings(
            model="flux-general-multi",
            language=None,
            eager_eot_threshold=None,
            eot_threshold=None,
            eot_timeout_ms=None,
            keyterm=[],
            min_confidence=None,
            language_hints=None,
            numerals=None,
            profanity_filter=None,
            redact=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            url=f"{base_url}{ws_path}",
            flux_encoding=flux_encoding,
            mip_opt_out=mip_opt_out,
            tag=tag,
            should_interrupt=should_interrupt,
            watchdog_min_timeout=watchdog_min_timeout,
            settings=default_settings,
            sample_rate=sample_rate,
            **kwargs,
        )
        self._correlation_id = correlation_id
        self._start_metadata = None

    # ------------------------------------------------------------------
    # MPS billing
    # ------------------------------------------------------------------

    def _get_correlation_id(self) -> str | None:
        return get_correlation_id(
            explicit_correlation_id=self._correlation_id,
            start_metadata=self._start_metadata,
        )

    def _build_query_string(self) -> str:
        """Append MPS billing/correlation params to the inherited Flux query."""
        query = super()._build_query_string()
        correlation_id = self._get_correlation_id()
        if correlation_id:
            query += "&" + urlencode({"correlation_id": correlation_id})
            query += "&" + urlencode({MPS_BILLING_VERSION_KEY: MPS_BILLING_VERSION_V2})
        return query

    async def start(self, frame: StartFrame):
        """Capture metadata for correlation on later reconnects."""
        self._start_metadata = frame.metadata
        await super().start(frame)

    async def _connect_websocket(self):
        """Apply current settings and billing metadata on every connection attempt."""
        self._websocket_url = f"{self._url}?{self._build_query_string()}"
        await super()._connect_websocket()

    async def _websocket_connect(self, uri: str, **kwargs):
        """Authenticate with the proxy while retaining the bounded socket transport."""
        headers = dict(kwargs.pop("additional_headers", {}) or {})
        headers["Authorization"] = f"Bearer {self._api_key}"
        return await super()._websocket_connect(uri, additional_headers=headers, **kwargs)

    async def _handle_message(self, data: dict):
        """Handle MPS errors alongside native Flux messages, including confirmation."""
        if isinstance(data, dict) and data.get("type") == "error":
            await self._handle_proxy_error(data)
        else:
            await super()._handle_message(data)

    async def _handle_proxy_error(self, data: dict):
        """Handle a Dograh proxy control error (e.g. quota exceeded)."""
        code = data.get("error")
        detail = data.get("message")
        error_msg = ": ".join(str(part) for part in (code, detail) if part) or "Unknown error"
        is_quota_error = "quota" in error_msg.lower()

        if is_quota_error:
            logger.info(f"STT quota exceeded: {error_msg}")
            await self.push_error(
                error_msg=f"STT service quota exceeded: {error_msg}",
                force_treat_as_permanent=True,
            )
            try:
                if self._websocket:
                    await self._websocket.close(
                        code=1000, reason="Quota exceeded - closing gracefully"
                    )
                    self._websocket = None
            except Exception as close_error:
                logger.debug(f"Error while closing websocket: {close_error}")
            return

        await self.push_error(error_msg=f"STT error: {error_msg}")
