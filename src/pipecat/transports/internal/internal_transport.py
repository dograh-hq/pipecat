#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Dict, Optional, Tuple

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
    OutputImageRawFrame,
    StartFrame,
    StopFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.internal import InternalFrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams


class InternalInputTransport(BaseInputTransport):
    """Input side of internal transport for agent-to-agent communication."""

    def __init__(self, transport: Optional["InternalTransport"], params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._partner: Optional["InternalOutputTransport"] = None
        self._running = False
        self._connected = False
        self._serializer = InternalFrameSerializer()

    def set_partner(self, partner: "InternalOutputTransport"):
        """Connect this input transport to an output transport."""
        self._partner = partner

    async def receive_data(self, data: bytes):
        """Receive serialized data from the partner output transport."""
        await self._queue.put(data)

    async def start(self, frame: StartFrame):
        """Start the input transport."""
        self._running = True
        await super().start(frame)
        await self._serializer.setup(frame)

        # Trigger on_client_connected event for InternalTransport (only once)
        if hasattr(self, "_transport") and self._transport and not self._connected:
            self._connected = True
            await self._transport._call_event_handler("on_client_connected", self._transport)

        asyncio.create_task(self._run())

    async def stop(self, frame: EndFrame | StopFrame | None = None):
        """Stop the input transport."""
        self._running = False
        await super().stop(frame)

        # Trigger on_client_disconnected event for InternalTransport
        if hasattr(self, "_transport") and self._transport:
            await self._transport._call_event_handler("on_client_disconnected", self._transport)

    async def _run(self):
        """Main loop to process incoming data."""
        while self._running:
            try:
                data = await asyncio.wait_for(self._queue.get(), timeout=0.1)

                # Deserialize the data
                frame = await self._serializer.deserialize(data)
                if frame:
                    await self.push_frame(frame, FrameDirection.DOWNSTREAM)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in internal input transport: {e}")


class InternalOutputTransport(BaseOutputTransport):
    """Output side of internal transport for agent-to-agent communication."""

    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._partner: Optional[InternalInputTransport] = None
        self._serializer = InternalFrameSerializer()

    def set_partner(self, partner: InternalInputTransport):
        """Connect this output transport to an input transport."""
        self._partner = partner

    async def start(self, frame: StartFrame):
        """Start the output transport."""
        await super().start(frame)
        await self._serializer.setup(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write audio frame to partner through serializer."""
        data = await self._serializer.serialize(frame)
        if data and self._partner:
            await self._partner.receive_data(data)

    async def write_video_frame(self, frame: OutputImageRawFrame):
        """Internal transport doesn't support video."""
        pass

    async def write_dtmf(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Internal transport doesn't support DTMF."""
        pass


class InternalTransport(BaseTransport):
    """Internal transport for in-memory agent-to-agent communication."""

    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)
        self._params = params

        # Create input and output transports
        self._input = InternalInputTransport(
            self, params, name=self._input_name or f"{self.name}#input"
        )
        self._output = InternalOutputTransport(
            params, name=self._output_name or f"{self.name}#output"
        )

        # Register supported event handlers
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self) -> InternalInputTransport:
        """Get the input transport."""
        return self._input

    def output(self) -> InternalOutputTransport:
        """Get the output transport."""
        return self._output

    def connect_partner(self, partner: "InternalTransport"):
        """Connect this transport to another internal transport."""
        # Connect output of this transport to input of partner
        self._output.set_partner(partner._input)
        # Connect output of partner to input of this transport
        partner._output.set_partner(self._input)


class InternalTransportManager:
    """Manages multiple internal transport pairs for load testing."""

    def __init__(self):
        self._transport_pairs: Dict[str, Tuple[InternalTransport, InternalTransport]] = {}

    def create_transport_pair(
        self, test_session_id: str, actor_params: TransportParams, adversary_params: TransportParams
    ) -> Tuple[InternalTransport, InternalTransport]:
        """Create a connected pair of internal transports."""

        # Create actor transport
        actor_transport = InternalTransport(params=actor_params, name=f"actor-{test_session_id}")

        # Create adversary transport
        adversary_transport = InternalTransport(
            params=adversary_params, name=f"adversary-{test_session_id}"
        )

        # Connect them
        actor_transport.connect_partner(adversary_transport)

        # Store the pair
        self._transport_pairs[test_session_id] = (actor_transport, adversary_transport)

        logger.info(f"Created internal transport pair for test session: {test_session_id}")

        return actor_transport, adversary_transport

    def get_transport_pair(
        self, test_session_id: str
    ) -> Optional[Tuple[InternalTransport, InternalTransport]]:
        """Get an existing transport pair."""
        return self._transport_pairs.get(test_session_id)

    def remove_transport_pair(self, test_session_id: str):
        """Remove a transport pair."""
        if test_session_id in self._transport_pairs:
            del self._transport_pairs[test_session_id]
            logger.info(f"Removed internal transport pair for test session: {test_session_id}")

    def get_active_test_count(self) -> int:
        """Get the number of active test sessions."""
        return len(self._transport_pairs)
