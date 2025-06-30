#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Smart-Turn analyzer that talks to a FastAPI WebSocket endpoint.

This analyzer keeps a persistent WebSocket connection alive so that the TCP/TLS
handshake and HTTP upgrade happen only once per call session.  Each speech
segment is sent as a single binary message containing the NumPy-serialized
float32 array, and a JSON reply is expected in return.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from typing import Any, Dict, Optional

import aiohttp
import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import (
    BaseSmartTurn,
    SmartTurnTimeoutException,
)


class WebSocketSmartTurnAnalyzer(BaseSmartTurn):
    """End-of-turn analyzer that sends audio via a persistent WebSocket."""

    def __init__(
        self,
        *,
        url: str,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        headers: Optional[Dict[str, str]] = None,
        service_context: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._url = url.rstrip("/")  # guard against trailing slash confusion
        self._headers = headers or {}
        self._aiohttp_session = aiohttp_session or aiohttp.ClientSession()
        self._service_context = service_context
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        # Protects _ws so we don't attempt concurrent `receive` on one socket.
        self._ws_lock = asyncio.Lock()

        # ------------------------------------------------------------------
        # Warm-up: establish the WebSocket immediately if an event loop is
        # already running so the first prediction incurs no handshake latency.
        # ------------------------------------------------------------------

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Fire-and-forget task – store it so we can await in tests if needed.
                self._warmup_task = loop.create_task(self._ensure_ws())
        except RuntimeError:
            # No running loop at object creation time (e.g. in sync path). The
            # connection will be opened lazily on first use.
            self._warmup_task = None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _serialize_array(self, audio_array: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        return buffer.getvalue()

    async def _ensure_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Return a connected WebSocket, reconnecting if necessary."""
        if self._ws is None or self._ws.closed:
            logger.debug("Opening new WebSocket connection to Smart-Turn service…")
            # Forward service context as a header if provided.
            extra_headers = dict(self._headers)
            if self._service_context is not None:
                extra_headers["X-Service-Context"] = str(self._service_context)
            try:
                self._ws = await self._aiohttp_session.ws_connect(
                    self._url,
                    headers=extra_headers,
                    heartbeat=30,
                    timeout=self._params.stop_secs,  # same as silence window
                )
            except Exception as exc:
                logger.error(f"Failed to establish WebSocket: {exc}")
                raise
        return self._ws

    # ------------------------------------------------------------------
    # BaseSmartTurn overrides
    # ------------------------------------------------------------------

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Send audio and await JSON response via WebSocket."""
        ws = await self._ensure_ws()
        data_bytes = self._serialize_array(audio_array)

        try:
            async with self._ws_lock:
                send_start = time.perf_counter()
                await ws.send_bytes(data_bytes)
                # Wait for response (must be text) with timeout.
                try:
                    recv_msg = await asyncio.wait_for(ws.receive(), timeout=self._params.stop_secs)
                except asyncio.TimeoutError:
                    logger.error(
                        f"WebSocket request timed out after {self._params.stop_secs} seconds"
                    )
                    raise SmartTurnTimeoutException(
                        f"Request exceeded {self._params.stop_secs} seconds."
                    )

                if recv_msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        return json.loads(recv_msg.data)
                    except json.JSONDecodeError as exc:
                        logger.error(f"Smart turn service returned invalid JSON: {exc}")
                        raise
                elif recv_msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error received from Smart-Turn service")
                    raise recv_msg.data  # re-raise underlying exception if any
                else:
                    logger.error(
                        f"Unexpected WebSocket message type: {recv_msg.type}. Closing socket."
                    )
                    await ws.close()
                    raise Exception("Unexpected WebSocket reply from Smart-Turn service.")

        except SmartTurnTimeoutException:
            raise
        except Exception as exc:
            logger.error(f"Smart turn prediction failed over WebSocket: {exc}")
            # Close so next attempt will reopen.
            if self._ws and not self._ws.closed:
                await self._ws.close()
            # Return default incomplete prediction so pipeline continues.
            return {
                "prediction": 0,
                "probability": 0.0,
                "metrics": {"inference_time": 0.0, "total_time": 0.0},
            }

    async def close(self):
        """Asynchronously close the WebSocket (called from pipeline cleanup)."""
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
