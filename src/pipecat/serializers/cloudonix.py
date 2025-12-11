from typing import Optional
from pipecat.serializers.twilio import TwilioFrameSerializer
from pydantic import BaseModel
from pipecat.audio.utils import create_stream_resampler
from loguru import logger

class CloudonixFrameSerializer(TwilioFrameSerializer):

    class InputParams(BaseModel):
        """Configuration parameters for CloudonixFrameSerializer.

        Parameters:
            cloudonix_sample_rate: Sample rate same as used by Twilio, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        cloudonix_sample_rate: int = 8000
        sample_rate: Optional[int] = None
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_sid: str,
        call_sid: Optional[str] = None,
        domain_id: Optional[str] = None,
        bearer_token: Optional[str] = None,
        region: Optional[str] = None,
        edge: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the CloudonixFrameSerializer.

        Args:
            stream_sid: The WebSocket Stream SID (Twilio-compatible).
            call_sid: The associated Cloudonix Call SID (optional, but required for auto hang-up).
            domain_id: Cloudonix domain ID (required for auto hang-up).
            bearer_token: Cloudonix bearer token (required for auto hang-up).
            region: Optional region parameter (legacy compatibility).
            edge: Optional edge parameter (legacy compatibility).
            params: Configuration parameters.
        """
        self._params = params or CloudonixFrameSerializer.InputParams()

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            # Validate required credentials
            missing_credentials = []
            if not call_sid:
                missing_credentials.append("call_sid")
            if not domain_id:
                missing_credentials.append("domain_id")
            if not bearer_token:
                missing_credentials.append("bearer_token")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )

            # Validate region and edge are both provided if either is specified (legacy compatibility)
            if (region and not edge) or (edge and not region):
                raise ValueError(
                    "Both edge and region parameters are required if one is set (legacy compatibility). "
                    f"Got: region='{region}', edge='{edge}'"
                )

        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._domain_id = domain_id
        self._bearer_token = bearer_token
        self._region = region
        self._edge = edge

        self.cloudonix_sample_rate = self._params.cloudonix_sample_rate
        self._twilio_sample_rate = self._params.cloudonix_sample_rate  # For parent class compatibility
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False
    
    async def _hang_up_call(self):
      """Cloudonix may auto-hang-up when WebSocket closes."""
      logger.info(f"Pipeline ended for Cloudonix call {self._call_sid} - relying on WebSocket close for hangup")
      return  # No-op - let WebSocket close handle it