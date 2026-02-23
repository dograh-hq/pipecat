"""Asterisk ARI WebSocket serializer for Pipecat.

Handles G.711 mu-law (ulaw) audio at 8kHz sent by Asterisk's
chan_websocket / externalMedia over binary WebSocket frames.
"""

import json
import time
from typing import Optional

from loguru import logger

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.enums import EndTaskReason


class AsteriskFrameSerializer(FrameSerializer):
    """Serializer for Asterisk ARI WebSocket audio streaming.

    Asterisk's chan_websocket sends raw G.711 mu-law (ulaw) audio at 8kHz
    as binary WebSocket frames. Unlike Twilio, there is no JSON wrapper
    or base64 encoding — audio bytes are sent directly as binary frames.

    On EndFrame/CancelFrame, the serializer will hang up the channel
    via ARI REST API (DELETE /ari/channels/{channel_id}).
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for AsteriskFrameSerializer.

        Parameters:
            asterisk_sample_rate: Sample rate used by Asterisk, defaults to 8000 Hz (ulaw).
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate channel on EndFrame.
        """

        asterisk_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(
        self,
        channel_id: str,
        ari_endpoint: str,
        app_name: str,
        app_password: str,
        params: Optional[InputParams] = None,
    ):
        """Initialize the AsteriskFrameSerializer.

        Args:
            channel_id: The Asterisk channel ID.
            ari_endpoint: ARI REST endpoint URL (e.g. http://localhost:8088).
            app_name: ARI application name for authentication.
            app_password: ARI application password for authentication.
            params: Configuration parameters.
        """
        super().__init__(params or AsteriskFrameSerializer.InputParams())

        self._channel_id = channel_id
        self._ari_endpoint = ari_endpoint
        self._app_name = app_name
        self._app_password = app_password

        self._asterisk_sample_rate = self._params.asterisk_sample_rate
        self._sample_rate = 0  # Pipeline input rate, set in setup()

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False
        self._transfer_attempted = False

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Asterisk WebSocket format.

        Converts PCM audio to G.711 mu-law and sends as raw binary bytes.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as bytes (ulaw audio) or None if the frame isn't handled.
        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            frame_reason = getattr(frame, "reason", None)
            logger.debug(f"Processing {type(frame).__name__} with reason: {frame_reason}")
            
            if frame_reason == EndTaskReason.TRANSFER_CALL.value and not self._transfer_attempted:
                self._transfer_attempted = True
                await self._transfer_call()
                return None
            elif (
                self._params.auto_hang_up
                and not self._hangup_attempted
                and frame_reason != "transfer_call"
            ):
                self._hangup_attempted = True
                await self._hang_up_call()
                return None
        elif isinstance(frame, InterruptionFrame):
            # Asterisk doesn't have a buffer clear command over the audio websocket.
            # Returning None; the transport will stop sending audio.
            return None
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz mu-law for Asterisk
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._asterisk_sample_rate, self._output_resampler
            )
            if serialized_data is None or len(serialized_data) == 0:
                return None

            # Asterisk expects raw binary ulaw bytes (no JSON wrapper, no base64)
            return serialized_data

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Asterisk WebSocket data to Pipecat frames.

        Binary messages contain raw G.711 mu-law audio bytes.
        Text messages contain JSON control events (if any).

        Args:
            data: The raw WebSocket data from Asterisk.

        Returns:
            A Pipecat frame corresponding to the data, or None if unhandled.
        """
        if isinstance(data, bytes):
            # Binary message = raw ulaw audio bytes
            deserialized_data = await ulaw_to_pcm(
                data,
                self._asterisk_sample_rate,
                self._sample_rate,
                self._input_resampler,
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=1,  # Asterisk sends mono audio
                sample_rate=self._sample_rate,
            )
            return audio_frame
        else:
            # Text message = JSON control event
            try:
                message = json.loads(data)
                event = message.get("type") or message.get("event")
                logger.debug(f"Asterisk WebSocket event: {event} - {message}")
                return None
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON message from Asterisk: {data}")
                return None

    async def _hang_up_call(self):
        """Hang up the Asterisk channel via ARI REST API."""
        try:
            import aiohttp
            from aiohttp import BasicAuth

            if not self._channel_id or not self._ari_endpoint:
                logger.warning(
                    "Cannot hang up Asterisk channel: missing channel_id or ari_endpoint"
                )
                return

            endpoint = f"{self._ari_endpoint}/ari/channels/{self._channel_id}"
            auth = BasicAuth(self._app_name, self._app_password)

            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, auth=auth) as response:
                    if response.status in (200, 204):
                        logger.info(f"Successfully terminated Asterisk channel {self._channel_id}")
                    elif response.status == 404:
                        logger.debug(f"Asterisk channel {self._channel_id} was already terminated")
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to terminate Asterisk channel {self._channel_id}: "
                            f"Status {response.status}, Response: {error_text}"
                        )

        except Exception as e:
            logger.exception(f"Failed to hang up Asterisk channel: {e}")

    async def _transfer_call(self):
        """Execute call transfer by performing bridge swap operations."""
        try:
            import aiohttp
            import redis.asyncio as aioredis
            from aiohttp import BasicAuth

            if not self._channel_id or not self._ari_endpoint:
                logger.warning(
                    "Cannot execute transfer: missing channel_id or ari_endpoint"
                )
                return

            logger.info(f"[ARI Transfer] Executing bridge swap for channel {self._channel_id}")

            # Import here to avoid circular dependencies
            from api.services.telephony.call_transfer_manager import get_call_transfer_manager
            from api.db import db_client
            from api.constants import REDIS_URL

            auth = BasicAuth(self._app_name, self._app_password)
            call_transfer_manager = await get_call_transfer_manager()
            
            # 1. Find active transfer context for this caller channel
            transfer_context = await self._find_transfer_context_for_caller(self._channel_id)
            if not transfer_context:
                logger.error(f"[ARI Transfer] No active transfer context found for caller {self._channel_id}")
                return

            logger.info(
                f"[ARI Transfer] Found transfer context: {transfer_context.transfer_id}, "
                f"destination: {transfer_context.call_sid}"
            )

            # 2. Get workflow run to find current bridge and external media channel
            redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            workflow_run_id = await redis.get(f"ari:channel:{self._channel_id}")
            if not workflow_run_id:
                logger.error(f"[ARI Transfer] No workflow run found for caller {self._channel_id}")
                return
                
            workflow_run = await db_client.get_workflow_run_by_id(int(workflow_run_id))
            if not workflow_run or not workflow_run.gathered_context:
                logger.error(f"[ARI Transfer] No workflow context found for run {workflow_run_id}")
                return

            ctx = workflow_run.gathered_context
            bridge_id = ctx.get("bridge_id")
            ext_channel_id = ctx.get("ext_channel_id")
            
            if not bridge_id or not ext_channel_id:
                logger.error(f"[ARI Transfer] Missing bridge/external channel info: {ctx}")
                return
                
            destination_channel_id = transfer_context.call_sid
            if not destination_channel_id:
                logger.error(f"[ARI Transfer] No destination channel in transfer context")
                return

            logger.info(
                f"[ARI Transfer] Bridge swap: bridge={bridge_id}, caller={self._channel_id}, "
                f"destination={destination_channel_id}, ext_media={ext_channel_id}"
            )

            # 2.5. Set transfer state to prevent StasisEnd auto-teardown
            workflow_run.gathered_context["transfer_state"] = "in-progress"
            await db_client.update_workflow_run(
                run_id=int(workflow_run_id), gathered_context=workflow_run.gathered_context
            )
            logger.debug(f"[ARI Transfer] Set transfer_state=in-progress for workflow {workflow_run_id}")

            # 3. Execute bridge swap operations via ARI REST API
            async with aiohttp.ClientSession() as session:
                # Add destination channel to existing bridge
                add_url = f"{self._ari_endpoint}/ari/bridges/{bridge_id}/addChannel"
                async with session.post(
                    add_url, 
                    auth=auth, 
                    params={"channel": destination_channel_id}
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"[ARI Transfer] Added destination {destination_channel_id} to bridge {bridge_id}")
                    else:
                        error_text = await response.text()
                        logger.error(f"[ARI Transfer] Failed to add destination to bridge: {response.status} {error_text}")
                        return

                # Remove external media channel from bridge
                remove_url = f"{self._ari_endpoint}/ari/bridges/{bridge_id}/removeChannel"
                async with session.post(
                    remove_url, 
                    auth=auth, 
                    params={"channel": ext_channel_id}
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"[ARI Transfer] Removed external media {ext_channel_id} from bridge {bridge_id}")
                    else:
                        error_text = await response.text()
                        logger.error(f"[ARI Transfer] Failed to remove external media from bridge: {response.status} {error_text}")
                        # Continue anyway - the main transfer connection is established

                # Hang up the external media channel
                hangup_url = f"{self._ari_endpoint}/ari/channels/{ext_channel_id}"
                async with session.delete(hangup_url, auth=auth) as response:
                    if response.status in (200, 204):
                        logger.info(f"[ARI Transfer] Hung up external media channel {ext_channel_id}")
                    elif response.status == 404:
                        logger.debug(f"[ARI Transfer] External media channel {ext_channel_id} already gone")
                    else:
                        error_text = await response.text()
                        logger.warning(f"[ARI Transfer] Failed to hang up external media: {response.status} {error_text}")

            # 4. Publish completion event
            from api.services.telephony.transfer_event_protocol import TransferEvent, TransferEventType
            completion_event = TransferEvent(
                type=TransferEventType.TRANSFER_COMPLETED,
                transfer_id=transfer_context.transfer_id,
                original_call_sid=self._channel_id,
                transfer_call_sid=destination_channel_id,
                conference_name=transfer_context.conference_name,
                message="Bridge swap completed successfully",
                status="success",
                action="transfer_success",
                end_call=True,
                timestamp=time.time()
            )
            await call_transfer_manager.publish_transfer_event(completion_event)

            # Note: Transfer context cleanup is now handled by background job in pipecat_engine_custom_tools.py
            # to avoid timing conflicts with pipeline processing

            logger.info(
                f"[ARI Transfer] Bridge swap completed successfully for transfer {transfer_context.transfer_id}"
            )
            
        except Exception as e:
            logger.exception(f"Failed to execute ARI transfer: {e}")

    async def _find_transfer_context_for_caller(self, caller_channel_id: str):
        """Find the active transfer context for this caller channel."""
        try:
            from api.services.telephony.call_transfer_manager import get_call_transfer_manager
            from api.services.telephony.transfer_event_protocol import TransferContext
            import redis.asyncio as aioredis
            from api.constants import REDIS_URL

            # Search Redis for transfer contexts where original_call_sid matches this caller
            redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            transfer_keys = await redis.keys("transfer:context:*")
            
            for key in transfer_keys:
                try:
                    context_data = await redis.get(key)
                    if context_data:
                        context = TransferContext.from_json(context_data)
                        if context.original_call_sid == caller_channel_id:
                            return context
                except Exception:
                    continue  # Skip malformed contexts
            
            return None
            
        except Exception as e:
            logger.error(f"[ARI Transfer] Error finding transfer context: {e}")
            return None
