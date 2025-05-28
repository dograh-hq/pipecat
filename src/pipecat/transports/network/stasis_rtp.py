# transports/ari_external_media.py  (new file)

import asyncio
import time
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
from pipecat.transports.network.stasis_rtp_client import StasisRTPClient
from pipecat.transports.network.stasis_rtp_connection import StasisRTPConnection


class StasisRTPTransportParams(TransportParams):
    serializer: FrameSerializer


class StasisRTPCallbacks(BaseModel):
    on_client_connected: Callable[[str], Awaitable[None]]
    on_client_disconnected: Callable[[str], Awaitable[None]]
    on_client_closed: Callable[[str], Awaitable[None]]


# ------------------------------------------------ Input Transport -------------------------


class StasisRTPInputTransport(BaseInputTransport):
    def __init__(
        self,
        transport: BaseTransport,
        client: StasisRTPClient,
        params: StasisRTPTransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._params = params

        self._receive_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        await super().start(frame)

        await self._client.setup(frame)
        await self._params.serializer.setup(frame)

        # Ensure underlying connection is established and socket ready.
        await self._client.connect()

        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_audio())

        await self.set_transport_ready(frame)

    async def _stop_tasks(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def _receive_audio(self):
        try:
            async for payload in self._client.receive():
                frame = await self._params.serializer.deserialize(payload)
                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as exc:
            logger.error(f"StasisRTPInputTransport receive error: {exc}")

    # No app-messages in RTP path, but keep compatibility
    async def push_app_message(self, message):
        logger.debug("StasisRTPInputTransport received app message ignored (RTP only)")


# ------------------------------------------------ Output Transport ------------------------


class StasisRTPOutputTransport(BaseOutputTransport):
    def __init__(
        self,
        transport: BaseTransport,
        client: StasisRTPClient,
        params: StasisRTPTransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client
        self._params = params

        # Pace outgoing audio so we don't dump buffers instantly (simulate 10-ms chunks)
        self._send_interval: float = 0
        self._next_send_time: float = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)

        await self._client.setup(frame)
        await self._params.serializer.setup(frame)

        # Compute pacing interval (same logic as FastAPI transport)
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2

        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._write_frame(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._write_frame(frame)
        await self._client.disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        # RTP path has no generic message channel; ignore.
        pass

    async def write_raw_audio_frames(self, frames: bytes, destination: Optional[str] = None):
        if self._client.is_closing:
            return

        if not self._client.is_connected:
            # If not connected yet, just simulate playback delay.
            await self._write_audio_sleep()
            return

        frame = OutputAudioRawFrame(
            audio=frames,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        payload = await self._params.serializer.serialize(frame)
        if payload:
            await self._client.send(payload)

        await self._write_audio_sleep()

    # ------------------------------------------------ helpers

    async def _write_frame(self, frame: Frame):
        try:
            payload = await self._params.serializer.serialize(frame)
            if payload:
                await self._client.send(payload)
        except Exception as exc:
            logger.error(f"StasisRTPOutputTransport send error: {exc}")

    async def _write_audio_sleep(self):
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class StasisRTPTransport(BaseTransport):
    def __init__(
        self,
        stasis_connection: StasisRTPConnection,
        params: StasisRTPTransportParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = StasisRTPCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_client_closed=self._on_client_closed,
        )

        self._client = StasisRTPClient(stasis_connection, self._callbacks)

        self._input = StasisRTPInputTransport(
            self, self._client, self._params, name=self._input_name
        )

        self._output = StasisRTPOutputTransport(
            self, self._client, self._params, name=self._output_name
        )

        # expose handlers
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_client_closed")

    def input(self) -> StasisRTPInputTransport:
        return self._input

    def output(self) -> StasisRTPOutputTransport:
        return self._output

    # ------------------------------------------------ event adapters ----------
    async def _on_client_connected(self, chan_id: str):
        await self._call_event_handler("on_client_connected", chan_id)

    async def _on_client_disconnected(self, chan_id: str):
        await self._call_event_handler("on_client_disconnected", chan_id)

    async def _on_client_closed(self, chan_id: str):
        await self._call_event_handler("on_client_closed", chan_id)
