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
import threading

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
        self._aiohttp_session = aiohttp_session
        self._owns_session = aiohttp_session is None
        if self._owns_session:
            self._aiohttp_session = aiohttp.ClientSession()
        self._service_context = service_context
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        # Protects _ws so we don't attempt concurrent `receive` on one socket.
        self._ws_lock = asyncio.Lock()

        # Connection management
        self._connection_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_reader_task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0  # Start with 1 second (base before jitter)
        self._max_reconnect_delay = 30.0  # Max 30 seconds
        self._closing = False
        self._connection_closed_event = asyncio.Event()

        # Connection health monitoring
        self._last_successful_request = 0.0
        self._connection_attempts = 0
        self._last_pong_time = 0.0
        # ------------------------------------------------------------------
        # Ensure only one coroutine calls `ws.receive()` at any time to avoid
        # aiohttp's "Concurrent call to receive()" runtime error.  The
        # background message reader and the prediction logic will both use
        # this lock to serialize receive operations.
        # ------------------------------------------------------------------
        self._recv_lock = asyncio.Lock()
        
        # Message reading coordination
        self._prediction_active = threading.Event()  # Thread-safe flag for prediction state
        self._prediction_lock = asyncio.Lock()  # Prevent concurrent predictions

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
                
                # Start message reader to handle pongs
                if self._message_reader_task:
                    self._message_reader_task.cancel()
                self._message_reader_task = asyncio.create_task(self._read_messages())
                
                # Also monitor the WebSocket state in case it closes unexpectedly
                monitor_task = asyncio.create_task(self._monitor_connection())

                # Wait for connection close event instead of polling
                self._connection_closed_event.clear()
                
                try:
                    await self._connection_closed_event.wait()
                finally:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass

                logger.debug("WebSocket connection closed")

            except Exception as e:
                logger.error(f"Connection manager error: {e}")

            finally:
                # Cancel all tasks
                tasks_to_cancel = []
                
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                    tasks_to_cancel.append(self._heartbeat_task)
                    self._heartbeat_task = None
                    
                if self._message_reader_task and not self._message_reader_task.done():
                    self._message_reader_task.cancel()
                    tasks_to_cancel.append(self._message_reader_task)
                    self._message_reader_task = None
                    
                # Wait for tasks to complete cancellation
                if tasks_to_cancel:
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

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
                # Initialize pong time with current time to avoid false positive on first check
                self._last_pong_time = time.time()
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
        """Send periodic pings to keep connection alive.

        NOTE: Currently using aggressive 2-second interval for stability monitoring.
        TODO: Taper down to 10-15 seconds once services are stable.
        """
        try:
            while not self._closing and self._ws and not self._ws.closed:
                await asyncio.sleep(
                    2
                )  # Aggressive ping every 2 seconds (will taper down once stable)

                try:
                    # Send JSON ping message instead of WebSocket ping frame
                    ping_msg = json.dumps({"type": "ping", "timestamp": time.time()})
                    async with self._ws_lock:
                        if self._ws and not self._ws.closed:
                            await self._ws.send_str(ping_msg)
                            logger.debug(f"Sent ping to smart turn service")
                        else:
                            logger.warning("WebSocket closed when trying to send ping")
                            self._connection_closed_event.set()
                            break

                    # Check if we've received a pong recently (2 missed pings = 4 seconds)
                    # Use 5 seconds to account for initial connection time
                    pong_timeout = 5.0 if time.time() - self._last_pong_time < 10 else 4.0
                    if self._last_pong_time > 0:
                        time_since_pong = time.time() - self._last_pong_time
                        if time_since_pong > pong_timeout:
                            logger.warning(
                                f"No pong received for {time_since_pong:.1f} seconds "
                                f"(>{pong_timeout/2:.0f} missed pings), forcing reconnection"
                            )
                            if self._ws and not self._ws.closed:
                                await self._ws.close()
                            self._connection_closed_event.set()
                            break
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")
                    # Force reconnection
                    if self._ws and not self._ws.closed:
                        try:
                            await self._ws.close()
                        except:
                            pass
                    self._connection_closed_event.set()
                    break
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
            pass
    
    async def _read_messages(self) -> None:
        """Read messages from WebSocket to handle pongs and other control messages."""
        consecutive_errors = 0
        try:
            while not self._closing:
                # Get a local reference to the WebSocket
                ws = self._ws
                if not ws or ws.closed:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Skip reading if prediction is active
                if self._prediction_active.is_set():
                    await asyncio.sleep(0.1)
                    continue
                    
                try:
                    # Read messages with a timeout
                    async with self._recv_lock:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    consecutive_errors = 0  # Reset error counter on success
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            
                            # Handle pong responses
                            if data.get("type") == "pong":
                                old_pong_time = self._last_pong_time
                                self._last_pong_time = time.time()
                                if old_pong_time > 0:
                                    pong_delay = self._last_pong_time - old_pong_time
                                    logger.debug(f"Received pong from smart turn service (delay: {pong_delay:.2f}s)")
                                else:
                                    logger.debug(f"Received first pong from smart turn service")
                                continue
                                
                            # If it's a prediction response, log warning
                            if "prediction" in data:
                                logger.warning(f"Received prediction response in message reader - this should not happen!")
                                continue
                                
                            # Log unexpected messages
                            logger.debug(f"Received unexpected message in reader: {data}")
                            
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON message: {msg.data}")
                            
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        # Log binary messages but don't process them
                        logger.debug(f"Received binary message in reader ({len(msg.data)} bytes) - ignoring")
                        continue
                        
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error in message reader: {msg.data}")
                        self._connection_closed_event.set()
                        break
                        
                    elif msg.type == aiohttp.WSMsgType.CLOSE:
                        logger.info("WebSocket closed by server in message reader")
                        self._connection_closed_event.set()
                        break
                        
                except asyncio.TimeoutError:
                    # Timeout is normal, just check if we should continue
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error reading messages (consecutive errors: {consecutive_errors}): {e}")
                    if consecutive_errors >= 3:
                        logger.error("Too many consecutive errors in message reader, triggering reconnection")
                        self._connection_closed_event.set()
                        break
                    await asyncio.sleep(0.5)  # Brief pause before retry
                    
        except asyncio.CancelledError:
            logger.debug("Message reader cancelled")
            pass
    
    async def _monitor_connection(self) -> None:
        """Monitor WebSocket state and trigger event if closed unexpectedly."""
        try:
            while self._ws and not self._closing:
                if self._ws.closed:
                    logger.debug("WebSocket closed unexpectedly, triggering reconnection")
                    self._connection_closed_event.set()
                    break
                await asyncio.sleep(0.1)  # Check every 100ms
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
        max_wait_time = 10.0  # 10 second timeout
        
        while not self._closing:
            # Check connection state
            if self._ws and not self._ws.closed:
                # Return the WebSocket if it's open
                return self._ws

            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.error(f"Timeout after {elapsed:.1f}s waiting for WebSocket connection")
                raise Exception(f"Timeout waiting for WebSocket connection after {max_wait_time}s")
            
            # Log progress every 2 seconds
            if int(elapsed) % 2 == 0 and elapsed > 0:
                logger.debug(f"Still waiting for WebSocket connection ({elapsed:.1f}s elapsed)")
                
            await asyncio.sleep(0.1)

        if self._closing:
            raise Exception("Analyzer is closing")

        raise Exception("Failed to establish WebSocket connection")

    # ------------------------------------------------------------------
    # BaseSmartTurn overrides
    # ------------------------------------------------------------------

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Send audio and await JSON response via WebSocket."""
        # Prevent concurrent predictions
        async with self._prediction_lock:
            return await self._do_predict(audio_array)
            
    async def _do_predict(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Internal prediction logic with proper cleanup."""
        ws = await self._ensure_ws()
        data_bytes = self._serialize_array(audio_array)
        
        # Set prediction active flag
        self._prediction_active.set()
        
        try:
            # Send request with minimal lock duration
            async with self._ws_lock:
                # Check if WebSocket is still open before sending
                if ws.closed:
                    logger.warning("WebSocket is closed, triggering reconnection")
                    self._ws = None
                    ws = await self._ensure_ws()

                # Send data
                try:
                    await ws.send_bytes(data_bytes)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "closing" in error_msg or "closed" in error_msg:
                        logger.warning(f"WebSocket in closing/closed state: {e}")
                        self._ws = None
                        self._connection_closed_event.set()
                        return {
                            "prediction": 0,
                            "probability": 0.0,
                            "metrics": {"inference_time": 0.0, "total_time": 0.0},
                        }
                    else:
                        logger.error(f"Failed to send data: {e}")
                        if self._ws and not self._ws.closed:
                            try:
                                await self._ws.close()
                            except Exception:
                                pass
                        self._ws = None
                        self._connection_closed_event.set()
                        raise

            # Wait for response outside of lock
            start_time = time.time()
            while True:
                # Check if WebSocket is still connected
                if not ws or ws.closed:
                    logger.error("WebSocket closed during prediction")
                    self._connection_closed_event.set()
                    return {
                        "prediction": 0,
                        "probability": 0.0,
                        "metrics": {"inference_time": 0.0, "total_time": 0.0},
                    }
                    
                remaining_timeout = self._params.stop_secs - (time.time() - start_time)
                if remaining_timeout <= 0:
                    logger.error(
                        f"WebSocket request timed out after {self._params.stop_secs} seconds"
                    )
                    raise SmartTurnTimeoutException(
                        f"Request exceeded {self._params.stop_secs} seconds."
                    )

                try:
                    # Use a shorter timeout to allow checking for other conditions
                    async with self._recv_lock:
                        recv_msg = await asyncio.wait_for(
                            ws.receive(), timeout=min(remaining_timeout, 0.5)
                        )
                except asyncio.TimeoutError:
                    # Check if we should continue waiting
                    if time.time() - start_time >= self._params.stop_secs:
                        logger.error(
                            f"WebSocket request timed out after {self._params.stop_secs} seconds"
                        )
                        raise SmartTurnTimeoutException(
                            f"Request exceeded {self._params.stop_secs} seconds."
                        )
                    continue

                if recv_msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        result = json.loads(recv_msg.data)

                        # This should not happen since server no longer sends pings
                        if result.get("type") == "ping":
                            logger.warning("Unexpected ping from server - ignoring")
                            continue

                        # Handle pong responses (shouldn't arrive here but just in case)
                        if result.get("type") == "pong":
                            self._last_pong_time = time.time()
                            logger.debug("Received pong in prediction flow")
                            continue

                        # Validate that this is a prediction response
                        if "prediction" not in result:
                            if "type" in result:
                                continue  # Wait for actual prediction response
                            else:
                                logger.error("Invalid response format from Smart-Turn service")
                                return {
                                    "prediction": 0,
                                    "probability": 0.0,
                                    "metrics": {"inference_time": 0.0, "total_time": 0.0},
                                }

                        # Mark successful request
                        self._last_successful_request = time.time()
                        return result
                    except json.JSONDecodeError as exc:
                        logger.error(f"Smart turn service returned invalid JSON: {exc}")
                        raise
                elif recv_msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error received from Smart-Turn service")
                    self._connection_closed_event.set()
                    raise Exception(f"WebSocket error: {recv_msg.data}")
                elif recv_msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.warning("WebSocket close message received")
                    self._connection_closed_event.set()
                    raise Exception("WebSocket closed by server")
                else:
                    logger.error(
                        f"Unexpected WebSocket message type: {recv_msg.type}. Closing socket."
                    )
                    await ws.close()
                    self._connection_closed_event.set()
                    raise Exception("Unexpected WebSocket reply from Smart-Turn service.")

        except SmartTurnTimeoutException:
            raise
        except Exception as exc:
            logger.error(f"Smart turn prediction failed over WebSocket: {exc}")
            # Close so connection manager will reconnect
            if self._ws and not self._ws.closed:
                try:
                    await self._ws.close()
                except Exception:
                    pass
            self._ws = None
            self._connection_closed_event.set()
            # Return default incomplete prediction so pipeline continues.
            return {
                "prediction": 0,
                "probability": 0.0,
                "metrics": {"inference_time": 0.0, "total_time": 0.0},
            }
        finally:
            # Resume message reader
            self._prediction_active.clear()

    async def close(self):
        """Asynchronously close the WebSocket (called from pipeline cleanup)."""
        # Set closing flag first to prevent new operations
        self._closing = True
        
        # Trigger connection close event to wake up connection manager
        self._connection_closed_event.set()

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
                    
            # Cancel message reader
            if self._message_reader_task and not self._message_reader_task.done():
                self._message_reader_task.cancel()
                try:
                    await self._message_reader_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"Error canceling message reader task: {e}")

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
                    
            # Close the aiohttp session if we own it
            if self._owns_session and self._aiohttp_session:
                try:
                    await self._aiohttp_session.close()
                except Exception as e:
                    logger.debug(f"Error closing aiohttp session: {e}")
                finally:
                    self._aiohttp_session = None
