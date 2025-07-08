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
import random
import time
from typing import Any, Dict, Optional

import aiohttp
import numpy as np
from aiohttp import WSCloseCode
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
        self._reconnect_delay = 1.0  # Start with 1 second (base before jitter)
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
            logger.error(
                "No running loop at object creation time. The connection will be opened lazily on first use."
            )
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

                # Wait for connection to close by monitoring the closed state
                while self._ws and not self._ws.closed and not self._closing:
                    await asyncio.sleep(1)  # Check connection status periodically

                if self._ws and self._ws.closed:
                    logger.debug("WebSocket connection closed")

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
                    delay = min(
                        self._reconnect_delay * (2 ** min(self._connection_attempts - 1, 5)),
                        self._max_reconnect_delay,
                    )
                    # Add small random jitter (0-500 ms) to avoid stampede
                    delay += random.uniform(0, 0.5)
                    logger.info(
                        f"Reconnecting in {delay:.1f} seconds (attempt {self._connection_attempts})"
                    )
                    await asyncio.sleep(delay)

    async def _establish_connection(self) -> None:
        """Establish a new WebSocket connection with retry logic."""
        logger.debug("Establishing new WebSocket connection to Smart-Turn service…")

        # Forward service context as a header if provided.
        extra_headers = dict(self._headers)
        if self._service_context is not None:
            extra_headers["X-Service-Context"] = str(self._service_context)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Add jitter to prevent thundering herd
                if attempt > 0:
                    jitter = 0.1 * attempt
                    await asyncio.sleep(jitter)

                self._ws = await self._aiohttp_session.ws_connect(
                    self._url,
                    headers=extra_headers,
                    heartbeat=30,
                    timeout=aiohttp.ClientTimeout(
                        total=10.0,  # Total timeout
                        connect=5.0,  # Connection timeout
                        sock_connect=5.0,  # Socket connection timeout
                    ),
                    autoping=True,  # Enable automatic ping/pong
                    autoclose=False,  # We'll handle closing ourselves
                )

                # Verify connection is actually open
                if self._ws.closed:
                    raise Exception("WebSocket connection closed immediately after establishment")

                logger.info("WebSocket connection established successfully")
                return

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    f"Failed to establish WebSocket (attempt {attempt + 1}/{max_attempts}): {exc}"
                )
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        try:
            while not self._closing and self._ws and not self._ws.closed:
                await asyncio.sleep(10)  # Ping every 10 seconds

                # Check connection health
                if time.time() - self._last_successful_request > 60:
                    logger.debug("Sending heartbeat ping to turn service")
                    try:
                        # `ping()` returns a Future that resolves when the pong
                        # is received. Do NOT await the result of `ping()` *and*
                        # then the returned value, otherwise the second await
                        # will raise `TypeError: object NoneType can't be used
                        # in 'await' expression`.
                        pong_waiter = self._ws.ping()
                        await pong_waiter  # Wait for pong response
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
        async with self._ws_lock:  # Prevent concurrent connection attempts
            # If connection manager isn't running, start it
            if not self._connection_task or self._connection_task.done():
                self._connection_task = asyncio.create_task(self._connection_manager())

        # Wait for connection with timeout (outside the lock to avoid deadlock)
        start_time = time.time()
        while not self._closing:
            # Check connection state
            if self._ws and not self._ws.closed:
                # Also check if ws is in closing state
                try:
                    # This is a hack but aiohttp doesn't expose closing state directly
                    if hasattr(self._ws, "_closing") and self._ws._closing:
                        logger.warning("WebSocket is in closing state, waiting for reconnection")
                        await asyncio.sleep(0.1)
                        continue
                except:
                    pass
                return self._ws

            if time.time() - start_time > 10:  # 10 second timeout
                raise Exception("Timeout waiting for WebSocket connection")
            await asyncio.sleep(0.1)

        if self._closing:
            raise Exception("Analyzer is closing")

        raise Exception("Failed to establish WebSocket connection")

    # ------------------------------------------------------------------
    # BaseSmartTurn overrides
    # ------------------------------------------------------------------

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Send audio and await JSON response via WebSocket."""
        ws = await self._ensure_ws()
        data_bytes = self._serialize_array(audio_array)

        try:
            async with self._ws_lock:
                # Check if WebSocket is still open before sending
                if ws.closed:
                    logger.warning("WebSocket is closed, triggering reconnection")
                    # Force reconnection
                    self._ws = None
                    ws = await self._ensure_ws()

                # Additional check for closing state
                if hasattr(ws, "_closing") and ws._closing:
                    logger.warning("WebSocket is in closing state, triggering reconnection")
                    self._ws = None
                    ws = await self._ensure_ws()

                # Send data with specific error handling
                try:
                    await ws.send_bytes(data_bytes)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "closing" in error_msg or "closed" in error_msg:
                        logger.warning(f"WebSocket in closing/closed state: {e}")
                        # Don't try to close again, just mark for reconnection
                        self._ws = None
                        # Return default response instead of raising
                        return {
                            "prediction": 0,
                            "probability": 0.0,
                            "metrics": {"inference_time": 0.0, "total_time": 0.0},
                        }
                    else:
                        logger.error(f"Failed to send data: {e}")
                        # For other errors, attempt to close
                        if self._ws and not self._ws.closed:
                            try:
                                await self._ws.close()
                            except:
                                pass
                        self._ws = None
                        raise

                # Wait for response with timeout, handling ping messages
                start_time = time.time()
                while True:
                    remaining_timeout = self._params.stop_secs - (time.time() - start_time)
                    if remaining_timeout <= 0:
                        logger.error(
                            f"WebSocket request timed out after {self._params.stop_secs} seconds"
                        )
                        raise SmartTurnTimeoutException(
                            f"Request exceeded {self._params.stop_secs} seconds."
                        )

                    try:
                        recv_msg = await asyncio.wait_for(ws.receive(), timeout=remaining_timeout)
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

                            # Handle server ping messages
                            if result.get("type") == "ping":
                                # Send pong response
                                try:
                                    await ws.send_str(
                                        json.dumps({"type": "pong", "timestamp": time.time()})
                                    )
                                    logger.debug("Sent pong response to server ping")
                                    continue  # Wait for actual response
                                except Exception as e:
                                    logger.error(f"Failed to send pong: {e}")
                                    continue

                            # Mark successful request
                            self._last_successful_request = time.time()
                            return result
                        except json.JSONDecodeError as exc:
                            logger.error(f"Smart turn service returned invalid JSON: {exc}")
                            raise
                    elif recv_msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error("WebSocket error received from Smart-Turn service")
                        raise Exception(f"WebSocket error: {recv_msg.data}")
                    elif recv_msg.type == aiohttp.WSMsgType.CLOSE:
                        logger.warning("WebSocket close message received")
                        raise Exception("WebSocket closed by server")
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
                try:
                    await self._ws.close()
                except:
                    pass
            self._ws = None
            # Return default incomplete prediction so pipeline continues.
            return {
                "prediction": 0,
                "probability": 0.0,
                "metrics": {"inference_time": 0.0, "total_time": 0.0},
            }

    async def close(self):
        """Asynchronously close the WebSocket (called from pipeline cleanup)."""
        # Set closing flag first to prevent new operations
        self._closing = True

        # Use a lock to prevent concurrent close operations
        async with self._ws_lock:
            # Cancel connection manager
            if self._connection_task and not self._connection_task.done():
                self._connection_task.cancel()
                try:
                    await self._connection_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"Error canceling connection task: {e}")

            # Cancel heartbeat
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"Error canceling heartbeat task: {e}")

            # Close WebSocket with proper error handling
            if self._ws:
                try:
                    if not self._ws.closed:
                        # Send close frame with normal closure code
                        await self._ws.close(code=WSCloseCode.OK)
                        # Wait a bit for graceful close
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Error during WebSocket close: {e}")
                finally:
                    self._ws = None
