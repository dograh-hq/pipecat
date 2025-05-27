# transports/ari_external_media.py  (new file)

import asyncio
import socket
import struct
import time
import uuid
from typing import AsyncIterator, Awaitable, Callable, Deque, Optional, Tuple

import asyncari
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
from pipecat.serializers.asterisk import AsteriskFrameSerializer
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

ULAW_PT = 0  # PCMU per RFC 3551
RTP_HEADER = struct.Struct("!BBHII")  # same as your working demo


class ARIExternalMediaParams(TransportParams):
    ari_url: str = "http://localhost:8088"
    ari_user: str = "dograhapp"
    ari_pass: str = "supersecretari"
    app_name: str = "pybridge"
    media_ip: str = "127.0.0.1"
    media_port: int = 4000  # This is the port on which Asterisk will try to connect
    codec: str = "ulaw"  # works with μ-law reader/writer demo
    direction: str = "both"  # send+receive
    serializer: FrameSerializer = AsteriskFrameSerializer()


class ARIExternalMediaCallbacks(BaseModel):
    on_client_connected: Callable[[str], Awaitable[None]]
    on_client_disconnected: Callable[[str], Awaitable[None]]
    on_session_timeout: Callable[[str], Awaitable[None]]


class ARIExternalMediaClient:
    """
    * Creates a bridge + ExternalMedia channel when told to `setup()`
    * Starts two background tasks (reader / writer) that push raw payloads
      through asyncio Queues.
    * Exposes:
        - `receive()`  -> AsyncIterator[bytes]
        - `send(bytes)` (μ-law payload *only*, no RTP header)
        - `disconnect()`
    """

    def __init__(self, params: ARIExternalMediaParams, callbacks: ARIExternalMediaCallbacks):
        self._params = params
        self._cb = callbacks
        self._audio_in_queue = asyncio.Queue()  # payloads → Input transport
        self._audio_out_queue = asyncio.Queue()  # Output transport → writer

        self._closing = False
        self._tasks = []  # asyncio tasks
        self._ari_channel_id = None  # ARI channel id (str)

        self._know_destination_future = asyncio.get_running_loop().create_future()
        self._destination_socket = None  # tell writer

    async def setup(self, _: StartFrame):
        if self._tasks:
            return  # already running
        task = asyncio.create_task(self._spawn_ari_and_rtp())
        self._tasks.append(task)

    def receive(self) -> AsyncIterator[bytes]:
        async def _gen():
            while True:
                payload = await self._audio_in_queue.get()
                yield payload

        return _gen()

    async def send(self, data: bytes):
        if self._closing:
            return
        await self._audio_out_queue.put(data)

    @property
    def is_connected(self) -> bool:
        return not self._closing

    @property
    def is_closing(self) -> bool:
        return self._closing

    async def disconnect(self):
        if self._closing:
            return
        self._closing = True
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = []

        await self._cb.on_client_disconnected(self._ari_channel_id or "<unknown>")

    # ------------------------------------------------------------ private impl
    async def _spawn_ari_and_rtp(self):
        """Runs as a task; spins reader & writer once ExternalMedia
        is inside the bridge.
        """
        try:
            async with asyncari.connect(
                self._params.ari_url,
                self._params.app_name,
                self._params.ari_user,
                self._params.ari_pass,
            ) as ari_client:
                logger.debug(f"Connected to ARI client")

                # Wait for regular caller leg
                async with ari_client.on_channel_event("StasisStart") as listener:
                    async for objs, _ in listener:
                        chan = objs["channel"]

                        # Ignore the helper leg
                        if chan.name.startswith("UnicastRTP/"):
                            continue

                        logger.debug(f"StasisStart: channel name: {chan.name}")

                        self._ari_channel_id = chan.id
                        await self._cb.on_client_connected(chan.id)

                        # handle call
                        await chan.answer()
                        bridge = await ari_client.bridges.create(type="mixing")
                        await bridge.addChannel(channel=chan.id)

                        # Now ExternalMedia leg
                        external_media_channel_id = str(uuid.uuid4())
                        external_media = await ari_client.channels.externalMedia(
                            channelId=external_media_channel_id,
                            app=self._params.app_name,
                            external_host=f"{self._params.media_ip}:{self._params.media_port}",
                            format=self._params.codec,
                            direction=self._params.direction,
                        )
                        # Wait for its StasisStart before opening sockets
                        await self._wait_external_media_start(ari_client, external_media.id)
                        await bridge.addChannel(channel=external_media.id)

                        # launch RTP reader / writer now that we know peer
                        reader_task = asyncio.create_task(self._rtp_reader())
                        writer_task = asyncio.create_task(self._rtp_writer())
                        self._tasks.extend([reader_task, writer_task])
                        return  # we break out, reader/writer live on
        except asyncio.CancelledError:
            # Handle task cancellation
            raise
        except Exception as e:
            logger.error(f"Error in ARI external media: {e}")
            raise

    # -- helpers --------------------------------------------------------------
    async def _wait_external_media_start(self, ari, em_id):
        async with ari.on_channel_event("StasisStart") as l:
            async for objs, _ in l:
                if objs["channel"].id == em_id:
                    return

    async def _rtp_reader(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self._params.media_ip, self._params.media_port))
        sock.setblocking(False)

        first = True
        loop = asyncio.get_running_loop()
        while True:
            data, addr = await loop.sock_recvfrom(sock, 2048)
            if first:
                self._destination_socket = addr  # symmetric RTP
                self._know_destination_future.set_result(True)
                first = False
            await self._audio_in_queue.put(data[12:])  # strip RTP header

    async def _rtp_writer(self):
        # wait for us to know the destination after we learn it from connected socket
        await self._know_destination_future

        dst = self._destination_socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        seq, ts = 0, 0
        while True:
            payload = await self._audio_out_queue.get()
            header = RTP_HEADER.pack(0x80, 0x00, seq & 0xFFFF, ts, 0x12345678)
            sock.sendto(header + payload, dst)
            seq += 1
            ts += 160  # 20 ms @ 8 kHz


class ARIInputTransport(BaseInputTransport):
    def __init__(
        self,
        transport: BaseTransport,
        client: ARIExternalMediaClient,
        params: ARIExternalMediaParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._params = params
        self._receive_task = None
        self._serializer = self._params.serializer

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.setup(frame)
        await self._serializer.setup(frame)
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_audio())
        await self.set_transport_ready(frame)

    async def _receive_audio(self):
        try:
            async for ulaw in self._client.receive():
                frame = await self._serializer.deserialize(ulaw)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self}: reader error {e}")

    async def stop(self, f: EndFrame):
        await super().stop(f)
        await self._client.disconnect()

    async def cancel(self, f: CancelFrame):
        await super().cancel(f)
        await self._client.disconnect()


class ARIOutputTransport(BaseOutputTransport):
    def __init__(
        self,
        transport: "ARIExternalMediaTransport",
        client: ARIExternalMediaClient,
        params: ARIExternalMediaParams,
        **kw,
    ):
        super().__init__(params, **kw)
        self._t = transport
        self._c = client
        self._params = params
        self._serializer = self._params.serializer
        self._send_interval = 0
        self._next_send = 0

    async def start(self, f: StartFrame):
        await super().start(f)
        await self._c.setup(f)
        await self._serializer.setup(f)
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
        await self.set_transport_ready(f)

    async def write_raw_audio_frames(self, frames: bytes, destination=None):
        if self._c.is_closing:
            return

        # Convert PCM → μ-law before sending.
        out_frame = OutputAudioRawFrame(
            audio=frames,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        payload = await self._serializer.serialize(out_frame)

        if payload is not None:
            await self._c.send(payload)

        await self._simulate_clock()

    async def _simulate_clock(self):
        now = time.monotonic()
        wait = max(0, self._next_send - now)
        await asyncio.sleep(wait)
        if wait == 0:
            self._next_send = time.monotonic() + self._send_interval
        else:
            self._next_send += self._send_interval

    # boiler-plate stop / cancel identical to WS version
    async def stop(self, f: EndFrame):
        await super().stop(f)
        await self._c.disconnect()

    async def cancel(self, f: CancelFrame):
        await super().cancel(f)
        await self._c.disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartInterruptionFrame):
            self._next_send = 0


###############################################################################
# 4   High-level transport wrapper (mirrors FastAPIWebsocketTransport API)
###############################################################################


class ARIExternalMediaTransport(BaseTransport):
    def __init__(
        self,
        params: ARIExternalMediaParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)

        self._p = params
        self._cb = ARIExternalMediaCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )
        self._cli = ARIExternalMediaClient(self._p, self._cb)

        self._in = ARIInputTransport(self, self._cli, self._p, name=self._input_name)
        self._out = ARIOutputTransport(self, self._cli, self._p, name=self._output_name)

        # expose handlers
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")

    def input(self) -> ARIInputTransport:
        return self._in

    def output(self) -> ARIOutputTransport:
        return self._out

    # ------------------------------------------------ event adapters ----------
    async def _on_client_connected(self, chan_id: str):
        await self._call_event_handler("on_client_connected", chan_id)

    async def _on_client_disconnected(self, chan_id: str):
        await self._call_event_handler("on_client_disconnected", chan_id)

    async def _on_session_timeout(self, chan_id: str):
        await self._call_event_handler("on_session_timeout", chan_id)
