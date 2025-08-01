#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server transport implementation for Pipecat.

This module provides WebSocket server transport functionality for real-time
audio and data streaming, including client connection management, session
handling, and frame serialization.
"""

import asyncio
import io
import time
import wave
from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class WebsocketServerParams(TransportParams):
    """Configuration parameters for WebSocket server transport.

    Parameters:
        add_wav_header: Whether to add WAV headers to audio frames.
        serializer: Frame serializer for message encoding/decoding.
        session_timeout: Timeout in seconds for client sessions.
    """

    add_wav_header: bool = False
    serializer: Optional[FrameSerializer] = None
    session_timeout: Optional[int] = None


class WebsocketServerCallbacks(BaseModel):
    """Callback functions for WebSocket server events.

    Parameters:
        on_client_connected: Called when a client connects to the server.
        on_client_disconnected: Called when a client disconnects from the server.
        on_session_timeout: Called when a client session times out.
        on_websocket_ready: Called when the WebSocket server is ready to accept connections.
    """

    on_client_connected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_client_disconnected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_session_timeout: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_websocket_ready: Callable[[], Awaitable[None]]


class WebsocketServerInputTransport(BaseInputTransport):
    """WebSocket server input transport for receiving client data.

    Handles incoming WebSocket connections, message processing, and client
    session management including timeout monitoring and connection lifecycle.
    """

    def __init__(
        self,
        transport: BaseTransport,
        host: str,
        port: int,
        params: WebsocketServerParams,
        callbacks: WebsocketServerCallbacks,
        **kwargs,
    ):
        """Initialize the WebSocket server input transport.

        Args:
            transport: The parent transport instance.
            host: Host address to bind the WebSocket server to.
            port: Port number to bind the WebSocket server to.
            params: WebSocket server configuration parameters.
            callbacks: Callback functions for WebSocket events.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        self._server_task = None

        # This task will monitor the websocket connection periodically.
        self._monitor_task = None

        self._stop_server_event = asyncio.Event()

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the WebSocket server and initialize components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.serializer:
            await self._params.serializer.setup(frame)
        if not self._server_task:
            self._server_task = self.create_task(self._server_task_handler())
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the WebSocket server and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        self._stop_server_event.set()
        if self._monitor_task:
            await self.cancel_task(self._monitor_task)
            self._monitor_task = None
        if self._server_task:
            await self.wait_for_task(self._server_task)
            self._server_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the WebSocket server and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        if self._monitor_task:
            await self.cancel_task(self._monitor_task)
            self._monitor_task = None
        if self._server_task:
            await self.cancel_task(self._server_task)
            self._server_task = None

    async def cleanup(self):
        """Cleanup resources and parent transport."""
        await super().cleanup()
        await self._transport.cleanup()

    async def _server_task_handler(self):
        """Handle WebSocket server startup and client connections."""
        logger.info(f"Starting websocket server on {self._host}:{self._port}")
        async with websockets.serve(self._client_handler, self._host, self._port) as server:
            await self._callbacks.on_websocket_ready()
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket: websockets.WebSocketServerProtocol, path):
        """Handle individual client connections and message processing."""
        logger.info(f"New client connection from {websocket.remote_address}")
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client connected, using new connection")

        self._websocket = websocket

        # Notify
        await self._callbacks.on_client_connected(websocket)

        # Create a task to monitor the websocket connection
        if not self._monitor_task and self._params.session_timeout:
            self._monitor_task = self.create_task(
                self._monitor_websocket(websocket, self._params.session_timeout)
            )

        # Handle incoming messages
        try:
            async for message in websocket:
                if not self._params.serializer:
                    continue

                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await self._websocket.close()
        self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")

    async def _monitor_websocket(
        self, websocket: websockets.WebSocketServerProtocol, session_timeout: int
    ):
        """Monitor WebSocket connection for session timeout."""
        try:
            await asyncio.sleep(session_timeout)
            if not websocket.closed:
                await self._callbacks.on_session_timeout(websocket)
        except asyncio.CancelledError:
            logger.info(f"Monitoring task cancelled for: {websocket.remote_address}")
            raise


class WebsocketServerOutputTransport(BaseOutputTransport):
    """WebSocket server output transport for sending data to clients.

    Handles outgoing frame serialization, audio streaming with timing control,
    and client connection management for WebSocket communication.
    """

    def __init__(self, transport: BaseTransport, params: WebsocketServerParams, **kwargs):
        """Initialize the WebSocket server output transport.

        Args:
            transport: The parent transport instance.
            params: WebSocket server configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._params = params

        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # write_audio_frame() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def set_client_connection(self, websocket: Optional[websockets.WebSocketServerProtocol]):
        """Set the active client WebSocket connection.

        Args:
            websocket: The WebSocket connection to set as active, or None to clear.
        """
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client allowed, using new connection")
        self._websocket = websocket

    async def start(self, frame: StartFrame):
        """Start the output transport and initialize components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.serializer:
            await self._params.serializer.setup(frame)
        self._send_interval = self._params.audio_out_10ms_chunks * 10 / 1000  # ms
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and send final frame.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._write_frame(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and send cancellation frame.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._write_frame(frame)

    async def cleanup(self):
        """Cleanup resources and parent transport."""
        await super().cleanup()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle interruption timing.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        """Send a transport message frame to the client.

        Args:
            frame: The transport message frame to send.
        """
        await self._write_frame(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write an audio frame to the WebSocket client with timing control.

        Args:
            frame: The output audio frame to write.
        """
        if not self._websocket:
            # Simulate audio playback with a sleep.
            await self._write_audio_sleep()
            return

        frame = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                wav_frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                frame = wav_frame

        await self._write_frame(frame)

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

    async def _write_frame(self, frame: Frame):
        """Serialize and send a frame to the WebSocket client."""
        if not self._params.serializer:
            return

        try:
            payload = await self._params.serializer.serialize(frame)
            if payload and self._websocket:
                await self._websocket.send(payload)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

    async def _write_audio_sleep(self):
        """Simulate audio device timing by sleeping between audio chunks."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class WebsocketServerTransport(BaseTransport):
    """WebSocket server transport for bidirectional real-time communication.

    Provides a complete WebSocket server implementation with separate input and
    output transports, client connection management, and event handling for
    real-time audio and data streaming applications.
    """

    def __init__(
        self,
        params: WebsocketServerParams,
        host: str = "localhost",
        port: int = 8765,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the WebSocket server transport.

        Args:
            params: WebSocket server configuration parameters.
            host: Host address to bind the server to. Defaults to "localhost".
            port: Port number to bind the server to. Defaults to 8765.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
            on_websocket_ready=self._on_websocket_ready,
        )
        self._input: Optional[WebsocketServerInputTransport] = None
        self._output: Optional[WebsocketServerOutputTransport] = None
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")
        self._register_event_handler("on_websocket_ready")

    def input(self) -> WebsocketServerInputTransport:
        """Get the input transport for receiving client data.

        Returns:
            The WebSocket server input transport instance.
        """
        if not self._input:
            self._input = WebsocketServerInputTransport(
                self, self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> WebsocketServerOutputTransport:
        """Get the output transport for sending data to clients.

        Returns:
            The WebSocket server output transport instance.
        """
        if not self._output:
            self._output = WebsocketServerOutputTransport(
                self, self._params, name=self._output_name
            )
        return self._output

    async def _on_client_connected(self, websocket):
        """Handle client connection events."""
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnection events."""
        if self._output:
            await self._output.set_client_connection(None)
            await self._call_event_handler("on_client_disconnected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_session_timeout(self, websocket):
        """Handle client session timeout events."""
        await self._call_event_handler("on_session_timeout", websocket)

    async def _on_websocket_ready(self):
        """Handle WebSocket server ready events."""
        await self._call_event_handler("on_websocket_ready")
