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
        
        # Connection management
        self._connection_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0  # Start with 1 second
        self._max_reconnect_delay = 30.0  # Max 30 seconds
        self._closing = False
        
        # Connection health monitoring
        self._last_successful_request = 0.0
        self._connection_attempts = 0

        # ------------------------------------------------------------------
        # Warm-up: establish the WebSocket immediately if an event loop is
        # already running so the first prediction incurs no handshake latency.
        # ------------------------------------------------------------------

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Start connection manager instead of single warmup
                self._connection_task = loop.create_task(self._connection_manager())
        except RuntimeError:
            # No running loop at object creation time (e.g. in sync path). The
            # connection will be opened lazily on first use.
            logger.error("No running loop at object creation time. The connection will be opened lazily on first use.")
            pass

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _serialize_array(self, audio_array: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        return buffer.getvalue()

    async def _connection_manager(self) -> None:
        """Manages WebSocket connection lifecycle with automatic reconnection."""
        while not self._closing:
            try:
                # Establish connection
                await self._establish_connection()
                
                # Reset reconnect delay on successful connection
                self._reconnect_delay = 1.0
                self._connection_attempts = 0
                
                # Start heartbeat
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                # Wait for connection to close
                if self._ws and not self._ws.closed:
                    await self._ws.wait_closed()
                    
            except Exception as e:
                logger.error(f"Connection manager error: {e}")
                
            finally:
                # Cancel heartbeat if running
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                    self._heartbeat_task = None
                
                # Clean up connection
                if self._ws and not self._ws.closed:
                    try:
                        await self._ws.close()
                    except:
                        pass
                self._ws = None
                
                if not self._closing:
                    # Exponential backoff for reconnection
                    self._connection_attempts += 1
                    delay = min(self._reconnect_delay * (2 ** min(self._connection_attempts - 1, 5)), 
                               self._max_reconnect_delay)
                    logger.info(f"Reconnecting in {delay:.1f} seconds (attempt {self._connection_attempts})")
                    await asyncio.sleep(delay)

    async def _establish_connection(self) -> None:
        """Establish a new WebSocket connection."""
        logger.debug("Establishing new WebSocket connection to Smart-Turn service…")
        
        # Forward service context as a header if provided.
        extra_headers = dict(self._headers)
        if self._service_context is not None:
            extra_headers["X-Service-Context"] = str(self._service_context)
            
        try:
            self._ws = await self._aiohttp_session.ws_connect(
                self._url,
                headers=extra_headers,
                heartbeat=30,
                timeout=aiohttp.ClientTimeout(total=10.0),  # Connection timeout
                autoping=True,  # Enable automatic ping/pong
            )
            logger.info("WebSocket connection established successfully")
        except Exception as exc:
            logger.error(f"Failed to establish WebSocket: {exc}")
            raise

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        try:
            while not self._closing and self._ws and not self._ws.closed:
                await asyncio.sleep(15)  # Ping every 15 seconds
                
                # Check connection health
                if time.time() - self._last_successful_request > 60:
                    logger.debug("Sending heartbeat ping to turn service")
                    try:
                        pong = await self._ws.ping()
                        await pong  # Wait for pong response
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
                        # Force reconnection
                        if self._ws and not self._ws.closed:
                            await self._ws.close()
                        break
        except asyncio.CancelledError:
            pass

    async def _ensure_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Return a connected WebSocket, waiting for connection if necessary."""
        # If connection manager isn't running, start it
        if not self._connection_task or self._connection_task.done():
            self._connection_task = asyncio.create_task(self._connection_manager())
        
        # Wait for connection with timeout
        start_time = time.time()
        while (not self._ws or self._ws.closed) and not self._closing:
            if time.time() - start_time > 10:  # 10 second timeout
                raise Exception("Timeout waiting for WebSocket connection")
            await asyncio.sleep(0.1)
            
        if self._closing:
            raise Exception("Analyzer is closing")
            
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
                        result = json.loads(recv_msg.data)
                        # Mark successful request
                        self._last_successful_request = time.time()
                        return result
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
            # Close so connection manager will reconnect
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
        self._closing = True
        
        # Cancel connection manager
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
                
        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass