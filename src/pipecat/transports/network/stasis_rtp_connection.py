"""Stasis RTP connection management for Asterisk External Media."""

import asyncio
import json
import os
import uuid
from typing import Optional

import asyncari
from asyncari.model import Channel
from loguru import logger

from pipecat.utils.base_object import BaseObject
from pipecat.utils.enums import EndTaskReason

# NOTE: Currently we always use μ-law ("ulaw") with 20 ms frame size just like
# `SmallWebRTCConnection`.  If in the future we want to support other codecs or
# ptime we can expose those knobs here.


class StasisRTPConnection(BaseObject):
    """Represents the server-side view of a single RTP call from Asterisk.

    Uses Asterisk *externalMedia* for RTP communication.

    Lifecycle
    =========
    1. The **caller** (a PJSIP channel inside Asterisk) enters the Stasis
       application we receive the *StasisStart* ARI event.
    2. We create a dedicated UDP port and spawn an *externalMedia* channel so
       that Asterisk starts sending/receiving raw RTP to that port.
    3. A mixing bridge is created and both legs (caller + externalMedia) are
       added to it.  From this moment on the call is *connected*.
    4. When either channel is destroyed we trigger the *disconnected* / *closed*
       events and perform cleanup.

    The class purposefully mirrors the behaviour and public API of
    :class:`SmallWebRTCConnection` so that higher-level components can treat
    WebRTC and Stasis (RTP) calls uniformly.
    """

    _SUPPORTED_EVENTS = [
        # Connection states.  Keep the same wording as SmallWebRTCConnection to
        # allow code reuse.
        "connecting",
        "connected",
        "disconnected",
        "closed",
        "failed",
        "new",
    ]

    def __init__(
        self,
        ari_client: asyncari.Client,
        caller_channel: Channel,
        host: str,
        port: int,
    ) -> None:
        """Initialize Stasis RTP connection.

        Args:
            ari_client: ARI client for Asterisk communication.
            caller_channel: The caller's channel object.
            host: Host address for RTP transport.
            port: Port number for RTP transport.
        """
        super().__init__()

        # External dependencies.
        self._ari: asyncari.Client = ari_client

        # Channels/bridge involved in this call.
        self.caller_channel: Channel = caller_channel
        self.em_channel: Optional[Channel] = None  # externalMedia leg
        self._bridge: Optional[asyncari.model.Bridge] = None

        # RTP addressing information (useful for users wanting to create their
        # own RTP transports).
        self.local_addr = ("0.0.0.0", port)
        self.remote_addr = None

        # Internal state.
        self._closed: bool = False  # Means that the channels are hung up remotely
        self._connect_invoked: bool = False  # Mirrors SmallWebRTCConnection
        self._is_connected: bool = False  # True once bridge created & channels bridged

        # Register supported event handler names so that client code can
        # register callbacks via `add_event_handler` / `@event_handler`.
        for evt in self._SUPPORTED_EVENTS:
            self._register_event_handler(evt)

        # Kick-off asynchronous initialisation (bridge creation, etc.).
        asyncio.create_task(self._setup_call(host, port))

    # ---------------------------------------------------------------------
    # Public helpers – similar surface as SmallWebRTCConnection
    # ---------------------------------------------------------------------

    async def disconnect(self, reason: str):
        """Instruct Asterisk to hang-up the call and perform cleanup."""
        # If self._closed is set, it means there has been a remote hangup
        if self._closed:
            return

        # Hangup the caller channel
        try:
            if self.caller_channel:
                logger.debug(
                    f"Hanging up caller channel {self.caller_channel.id} due to reason: {reason}"
                )
                await self.caller_channel.hangup()
        except Exception:
            logger.exception("Failed to hangup caller channel")

    async def transfer(self, call_transfer_context: dict):
        """Transfer the call by continuing in dialplan with extracted variables."""
        # If self._closed is set, it means there has been a remote hangup
        if self._closed:
            return

        # Continue in dialplan with extracted variables
        try:
            if self.caller_channel:
                logger.debug(
                    f"User qualified, continuing in dialplan for channel {self.caller_channel.id} REMOTE_DISPO_CALL_VARIABLES: {json.dumps(call_transfer_context)}"
                )
                # Set variable REMOTE_DISPO_CALL_VARIABLES before continuing in dialplan
                await self.caller_channel.setChannelVar(
                    variable="REMOTE_DISPO_CALL_VARIABLES",
                    value=json.dumps(call_transfer_context),
                )
                await self.caller_channel.continueInDialplan()
        except Exception:
            logger.exception("Failed to transfer caller channel")

    async def connect(self):
        """Signal that the user is ready to start the call.

        For API parity with `SmallWebRTCConnection` this method **must** be
        called after the application has added its event handlers.  Only after
        this invocation will the *connected* event be dispatched (either
        immediately if the underlying bridge has already been formed, or later
        once `_setup_call` completes).
        """
        self._connect_invoked = True

        # If the call has already reached the *connected* state we can dispatch
        # the event right away.
        if self.is_connected():
            await self._call_event_handler("connected")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Check if the connection is established.

        Return *True* once the call is established **and** `connect()` has been
        invoked by the user (same semantics as `SmallWebRTCConnection`).
        """
        return self._connect_invoked and self._is_connected and not self._closed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _setup_call(self, host: str, port: int):
        """Create externalMedia + bridge and notify that the call is connected."""
        try:
            em_channel_id = str(uuid.uuid4())

            logger.debug(f"Creating externalMedia channel {em_channel_id} on {host}:{port}")

            self.em_channel = await self._ari.channels.externalMedia(
                app=self._ari.app,
                channelId=em_channel_id,
                external_host=f"{host}:{port}",
                format="ulaw",
                direction="both",
            )

            # Create a mixing bridge and add both legs.
            self._bridge = await self._ari.bridges.create(type="mixing")

            # Add both legs to the mixing bridge.
            await self._bridge.addChannel(channel=[self.caller_channel.id, self.em_channel.id])

            ip = await self.em_channel.getChannelVar(variable="UNICASTRTP_LOCAL_ADDRESS")
            port = await self.em_channel.getChannelVar(variable="UNICASTRTP_LOCAL_PORT")

            # self.remote_addr = (ip["value"], int(port["value"]))
            self.remote_addr = (os.environ.get("ASTERISK_REMOTE_IP"), int(port["value"]))

            logger.debug(
                f"StasisRTPConnection {self} connection resources ready (bridge {self._bridge.id}) and remote address: {self.remote_addr}"
            )

            # Mark internal state.
            self._is_connected = True

            # Only dispatch *connected* if the application has already invoked
            # `connect()`.
            if self._connect_invoked:
                await self._call_event_handler("connected")

            # Start background listener for ChannelDestroyed so that we know when the
            # call ends.
            asyncio.create_task(self._watch_channel_termination())

        except Exception as exc:
            logger.exception(f"Error setting up StasisRTPConnection: {exc}")
            await self._handle_disconnect(EndTaskReason.SYSTEM_CONNECT_ERROR.value)

    async def _watch_channel_termination(self):
        """Listen for ARI StasisEnd events for either leg."""
        try:
            async with self._ari.on_channel_event("StasisEnd") as listener:
                async for channel, _ in listener:
                    logger.debug(f"Got StasisEnd event with channel_id: {channel.id}")
                    if channel.id in (self.caller_channel.id, getattr(self.em_channel, "id", "")):
                        logger.debug(
                            f"Channel {channel.id} destroyed closing StasisRTPConnection {self}"
                        )

                        # Determine the disconnect reason based on which channel ended
                        if channel.id == self.caller_channel.id:
                            disconnect_reason = EndTaskReason.USER_HANGUP.value
                        else:
                            # externalMedia channel ended
                            disconnect_reason = EndTaskReason.SYSTEM_CANCELLED.value

                        await self._handle_disconnect(disconnect_reason)
                        break  # Exit listener
        except Exception as exc:
            # Listener finished unexpectedly – make sure we clean up.
            logger.exception(f"Channel watchdog stopped with error: {exc}")
            await self._handle_disconnect(EndTaskReason.UNKNOWN.value)

    async def _handle_disconnect(self, reason: str = EndTaskReason.UNKNOWN.value):
        """Common logic for both remote and local hang-up."""
        if self._closed:
            return
        self._closed = True

        # Emit disconnected **only** if the call had actually reached the
        # connected state. This will propagate the disconnected to the
        # RTPClient and to the Transport, where pipeline can be ended
        if self._is_connected:
            await self._call_event_handler("disconnected", reason)

        # Set _is_connected to False (fix the bug where is_connected was being set instead of _is_connected)
        self._is_connected = False

        # Hang up the externalMedia channel
        try:
            if self.em_channel:
                await self.em_channel.hangup()
        except Exception:
            logger.warning(f"Failed to hang-up externalMedia channel: {self.em_channel.id}")

        # Cleanup bridge if still present.
        try:
            if self._bridge:
                await self._bridge.destroy()
        except Exception:
            logger.warning(f"Failed to destroy bridge during disconnect: {self._bridge.id}")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        """Return string representation of connection."""
        return (
            f"<StasisRTPConnection id={self.id} caller={self.caller_channel.id} "
            f"em={getattr(self.em_channel, 'id', None)} state={'closed' if self._closed else 'open'}>"
        )
