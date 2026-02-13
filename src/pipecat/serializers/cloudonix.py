"""Cloudonix Media Streams WebSocket protocol serializer for Pipecat."""

from typing import Optional

from loguru import logger

from pipecat.serializers.twilio import TwilioFrameSerializer


class CloudonixFrameSerializer(TwilioFrameSerializer):
    """Serializer for Cloudonix Media Streams WebSocket protocol.

    This serializer extends TwilioFrameSerializer for Cloudonix compatibility,
    providing Cloudonix-specific call termination functionality while reusing
    Twilio's audio handling and frame processing capabilities.
    """

    def __init__(
        self,
        call_id: str,
        stream_sid: str,
        domain_id: Optional[str] = None,
        bearer_token: Optional[str] = None,
        region: Optional[str] = None,
        edge: Optional[str] = None,
        params: Optional[TwilioFrameSerializer.InputParams] = None,
    ):
        """Initialize the CloudonixFrameSerializer.

        Args:
            call_id: The associated Cloudonix Call ID (required for auto hang-up)
            stream_sid: The associated Cloudonix Stream SID (required for streaming audio)
            domain_id: Cloudonix domain ID (required for auto hang-up).
            bearer_token: Cloudonix bearer token (required for auto hang-up).
            region: Optional region parameter (legacy compatibility).
            edge: Optional edge parameter (legacy compatibility).
            params: Configuration parameters.
        """
        self._call_id = call_id
        self._domain_id = domain_id
        self._bearer_token = bearer_token

        super().__init__(
            stream_sid=stream_sid,
            call_sid=call_id,
            account_sid=domain_id,
            auth_token=bearer_token,
            region=region,
            edge=edge,
            params=params,
        )

        logger.info(f"Cloudonix serializer initialized with call_id: {self._call_id}")

    async def _hang_up_call(self):
        """Terminate the Cloudonix call by issuing a DELETE request to the session endpoint."""
        logger.debug(f"Attempting hangup for call {self._call_id}")

        # If call_id is not available, fall back to WebSocket close behavior
        if not self._call_id:
            logger.warning(f"No call_id available for call. Relying on WebSocket close for hangup.")
            return

        # Validate required parameters for API call
        if not self._domain_id or not self._bearer_token:
            logger.warning(
                f"Missing domain_id or bearer_token for call {self._call_id}. "
                f"Cannot perform explicit hangup via API."
            )
            return

        try:
            import aiohttp

            # Construct the DELETE session endpoint
            # Using "self" as customer-id as per Cloudonix documentation
            base_url = "https://api.cloudonix.io"
            endpoint = (
                f"{base_url}/customers/self/domains/{self._domain_id}/sessions/{self._call_id}"
            )

            # Prepare headers with Bearer token authentication
            headers = {
                "Authorization": f"Bearer {self._bearer_token}",
                "Content-Type": "application/json",
            }

            logger.info(f"Terminating Cloudonix call {self._call_id} via DELETE {endpoint}")

            # Make the DELETE request to terminate the session
            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, headers=headers) as response:
                    status = response.status
                    response_text = await response.text()

                    if status in (200, 204, 404):
                        # 200/204: Success
                        # 404: Session already terminated (acceptable)
                        logger.info(
                            f"Successfully terminated Cloudonix session {self._call_id} "
                            f"(HTTP {status}), Response: {response_text}"
                        )
                    else:
                        logger.warning(
                            f"Unexpected response terminating Cloudonix session {self._call_id}: "
                            f"HTTP {status}, Response: {response_text}"
                        )

        except Exception as e:
            logger.error(f"Error terminating Cloudonix call {self._call_id}: {e}", exc_info=True)
