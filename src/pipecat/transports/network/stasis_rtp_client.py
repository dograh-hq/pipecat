"""
stasis_rtp_client.py
~~~~~~~~~~~~~~~~~~~~
Low-level RTP transport for Asterisk *externalMedia* sessions.

* Sends and receives **proper RTP/UDP** (PT 0 PCMU/μ-law).
* Uses 20 ms frames (160 bytes payload) by default; automatically
  chunks or concatenates data so timestamps stay correct.
* Verifies the RTP header on the receive path (SSRC and PT).
"""

import asyncio
import secrets
import socket
import struct
from typing import TYPE_CHECKING, AsyncIterator, Optional

from loguru import logger

from pipecat.utils.enums import EndTaskReason

if TYPE_CHECKING:
    from pipecat.transports.network.stasis_rtp import StasisRTPCallbacks
    from pipecat.transports.network.stasis_rtp_connection import StasisRTPConnection

# ─────────────────────────────────────────────────────────────────── helpers


_RTP_HDR = struct.Struct("!BBHII")  # v/p/x/cc, m/pt, seq, ts, ssrc
_PT_PCMU = 0  # static payload type for μ-law


class _RTPEncoder:
    """Builds PCMU RTP headers for the packets we SEND to Asterisk."""

    def __init__(self):
        self.ssrc = secrets.randbits(32)
        self.seq = secrets.randbits(16)
        self.ts = 0  # incremented by #payload bytes

    def pack(self, payload: bytes, mark=False) -> bytes:
        b0 = 0x80  # V=2
        b1 = (0x80 if mark else 0x00) | _PT_PCMU
        hdr = _RTP_HDR.pack(b0, b1, self.seq, self.ts, self.ssrc)
        self.seq = (self.seq + 1) & 0xFFFF
        self.ts += len(payload)  # 1 sample/byte @ 8 kHz
        return hdr + payload


class _RTPDecoder:
    """
    Very forgiving: latches on the first valid packet and then insists
    that SSRC & PT match afterwards.  Returns *None* if the packet
    should be ignored.
    """

    def __init__(self):
        self.peer_ssrc: int | None = None  # learned from first packet

    def unpack(self, packet: bytes) -> bytes | None:
        if len(packet) < _RTP_HDR.size:
            return None
        b0, b1, seq, ts, ssrc = _RTP_HDR.unpack_from(packet)
        if (b0 & 0xC0) != 0x80:  # RTP v2?
            return None
        if (b1 & 0x7F) != _PT_PCMU:  # payload-type 0?
            return None
        if self.peer_ssrc is None:
            self.peer_ssrc = ssrc  # latch on first good packet
        elif ssrc != self.peer_ssrc:
            return None  # stray stream – drop
        return packet[_RTP_HDR.size :]


# ──────────────────────────────────────────────────────────────── client


class StasisRTPClient:
    """
    Low-level wrapper around StasisRTPConnection.

    Public API
    ──────────
    • await setup(start_frame)       kept for parity (does nothing)
    • await connect()
    • async for payload in receive():  # μ-law bytes (20 ms each)
          …
    • await send(data)               # any length; will be chunked
    • await disconnect()
    """

    _FRAME_SIZE = 160  # 20 ms @ 8 kHz PCMU

    def __init__(
        self,
        connection: "StasisRTPConnection",
        callbacks: "StasisRTPCallbacks",
    ):
        from typing import Any

        self._connection = connection
        self._callbacks = callbacks
        self._encoder = _RTPEncoder()
        self._decoder = _RTPDecoder()

        self._recv_sock: Optional[socket.socket] = None
        self._send_sock: Optional[socket.socket] = None
        self._closing = False
        self._recv_sock_ready = asyncio.Event()  # Signal when recv socket is ready
        self._leave_counter = 0  # Track input/output transport usage
        self._fallback_disconnect_timer: Optional[asyncio.Task] = (
            None  # Safety timer for disconnect
        )

        # ── wire event handlers to the connection ────────────────
        @self._connection.event_handler("connected")
        async def _on_connected(_: Any):
            await self._setup_sockets()
            await self._callbacks.on_client_connected(self._connection.caller_channel.id)

        @self._connection.event_handler("disconnected")
        async def _on_disconnected(_: Any, reason: str):
            # Cancel the safety timer if it exists
            if self._fallback_disconnect_timer and not self._fallback_disconnect_timer.done():
                self._fallback_disconnect_timer.cancel()
                self._fallback_disconnect_timer = None
            await self._callbacks.on_client_disconnected(self._connection.caller_channel.id, reason)

    # ─── public helpers ──────────────────────────────────────────

    async def setup(self, _):
        self._leave_counter += 1

    async def connect(self):
        if self._connection.is_connected():
            return
        await self._connection.connect()

    async def disconnect(
        self, reason: str = EndTaskReason.UNKNOWN.value, call_transfer_context: dict = {}
    ):
        # Decrement leave counter when disconnect is called
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self._closing:
            logger.warning(
                "In StasisRTPClient disconnect, but already closing. Figure out "
                "where is this double disconnect getting called from"
            )
            return
        self._closing = True

        # Close local sockets first - this will cause receive() to break
        await self._close_sockets()

        # Create a safety timer that will call on_client_disconnected if we don't hear from the connection
        async def _fallback_disconnect_timeout():
            await asyncio.sleep(5.0)
            logger.warning(
                "Disconnect event not received within 5 seconds, calling on_client_disconnected as fallback"
            )
            await self._callbacks.on_client_disconnected(self._connection.caller_channel.id)

        self._fallback_disconnect_timer = asyncio.create_task(_fallback_disconnect_timeout())

        # Decide whether to call connection's transfer or disconnect based on the reason
        try:
            if reason == EndTaskReason.USER_QUALIFIED.value:
                # User qualified - call transfer method
                await asyncio.wait_for(
                    self._connection.transfer(call_transfer_context), timeout=5.0
                )
            else:
                # Other reasons - call disconnect method
                await asyncio.wait_for(self._connection.disconnect(reason), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("RTP connection disconnect/transfer timed out")
        except Exception as exc:
            logger.error(f"Failed to disconnect/transfer RTP connection: {exc}")

    # ─── socket management ──────────────────────────────────────

    async def _setup_sockets(self):
        if self._recv_sock and self._send_sock:
            return

        logger.debug(
            f"Final addresses in _setup_sockets - local {self._connection.local_addr}, remote: {self._connection.remote_addr}"
        )

        # receive socket – bind to local address provided by connection
        if not self._recv_sock:
            rs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rs.setblocking(False)
            rs.bind(self._connection.local_addr)
            self._recv_sock = rs
            self._recv_sock_ready.set()  # Signal that recv socket is ready

        # send socket – connect to remote (Asterisk) address
        if not self._send_sock:
            ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ss.setblocking(False)
            ss.connect(self._connection.remote_addr)
            self._send_sock = ss

    async def _close_sockets(self):
        """Safely close sockets with proper error handling."""
        for sock_name, sock in [("recv", self._recv_sock), ("send", self._send_sock)]:
            if sock:
                try:
                    # Shutdown the socket first to break any pending operations
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    # Socket might already be closed or in a bad state
                    pass
                try:
                    sock.close()
                except Exception as exc:
                    logger.debug(f"Error closing {sock_name} socket: {exc}")

        self._recv_sock = None
        self._send_sock = None
        self._recv_sock_ready.clear()  # Reset the event for potential reconnection

        logger.debug("Closed sockets in StasisRTPClient")

    # ─── receive path ────────────────────────────────────────────

    async def receive(self) -> AsyncIterator[bytes]:
        """
        Async generator yielding μ-law frames (exactly 160 bytes each).
        Silently drops any packet whose RTP header does not match our SSRC/PT.
        """
        loop = asyncio.get_running_loop()

        # Wait for recv socket to be created
        try:
            await self._recv_sock_ready.wait()
        except asyncio.CancelledError:
            return

        logger.debug("Going to receive from the socket now")

        while not self._closing:
            try:
                data = await loop.sock_recv(self._recv_sock, 2048)
                # each loop gets 172 bytes UDP packet, which is 160 bytes of
                # audio data (Asterisk sends 20ms audio chunks with 8k sample rate)
                # and 12 bytes of RTP header
            except (asyncio.CancelledError, OSError, socket.error) as exc:
                logger.warning(f"RTP receive failed (socket closed): {exc}")
                break
            except Exception as exc:
                logger.debug(f"Unexpected error in receive: {exc}")
                break

            payload = self._decoder.unpack(data)
            if payload is None:
                continue  # header failed validation

            # In practice Asterisk sends 20 ms frames – assert just in case.
            if len(payload) != self._FRAME_SIZE:
                logger.warning(f"Dropping non-20 ms packet len={len(payload)}")
                continue
            yield payload

    # ─── send path ───────────────────────────────────────────────

    async def send(self, data: bytes):
        """
        Send μ-law data of arbitrary length.
        Splits/aggregates into 160-byte chunks before RTP-wrapping.
        """
        if self._closing or not self._send_sock:
            return
        loop = asyncio.get_running_loop()

        # chunk/concat to 160-byte frames
        chunks = self._chunk_ulaw(data, self._FRAME_SIZE)
        for i, chunk in enumerate(chunks):
            mark = i == 0  # set marker on the first packet of talk-spurt
            packet = self._encoder.pack(chunk, mark=mark)
            try:
                await loop.sock_sendall(self._send_sock, packet)
            except (OSError, socket.error) as exc:
                logger.warning(f"RTP send failed (socket closed): {exc}")
                break
            except Exception as exc:
                logger.error(f"RTP send failed: {exc}")
                break

    def _chunk_ulaw(self, buf: bytes, size: int) -> list[bytes]:
        """
        Split / aggregate μ-law bytes to exact *size* multiples.

        • If buf length is not a multiple of *size*, pad the last chunk with 0xFF
        (silence).  That keeps timestamps monotonic.
        """
        if not buf:
            return []
        if len(buf) % size:
            pad = size - (len(buf) % size)
            buf += b"\xff" * pad
        return [buf[i : i + size] for i in range(0, len(buf), size)]

    # ─── properties ──────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connection.is_connected() and not self._closing

    @property
    def is_closing(self) -> bool:
        return self._closing
