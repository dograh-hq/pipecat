from __future__ import annotations

# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
"""Asterisk External Media frame serializer.

This serializer converts between Pipecat frames and the raw μ-law RTP payload
stream expected by an Asterisk *External Media* channel.

Unlike Twilio's WebSocket media protocol, Asterisk delivers/consumes **bare
μ-law RTP payload bytes** there is no JSON envelope or base-64 encoding.
Consequently the serializer simply encodes/decodes audio while leaving control
frames untouched (they are handled at a higher level by the transport).

Typical usage with :class:`~pipecat.transports.network.ari_external_media.ARIExternalMediaTransport`::

    serializer = AsteriskFrameSerializer()
    transport = ARIExternalMediaTransport(params=ARIExternalMediaParams(), ...)
    transport.input().params.serializer = serializer  # or inject via DI
    transport.output().params.serializer = serializer

The serializer:

* Down-samples PCM to 8-kHz μ-law for **outgoing** audio (:class:`AudioRawFrame`).
* Up-samples μ-law to the pipeline's native rate for **incoming** audio.
* Ignores control frames – they are not represented on the wire.
"""

from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class AsteriskFrameSerializer(FrameSerializer):
    """Serializer for Asterisk External Media streams (raw μ-law)."""

    class InputParams(BaseModel):
        """Configuration parameters.

        Attributes
        ----------
        asterisk_sample_rate : int, default 8000
            The sample-rate used by Asterisk when sending μ-law (PCMU).
        sample_rate : Optional[int]
            Override for the pipeline's *input* sample-rate.  When omitted the
            value from the :class:`StartFrame` is used.
        """

        asterisk_sample_rate: int = 8000
        sample_rate: Optional[int] = None

    def __init__(self, params: Optional["AsteriskFrameSerializer.InputParams"] = None):
        self._params = params or AsteriskFrameSerializer.InputParams()

        # Wire / pipeline rates
        self._asterisk_sample_rate = self._params.asterisk_sample_rate
        self._sample_rate = 0  # pipeline rate, filled in *setup*

        # Resampler shared between encode / decode paths
        self._resampler = create_default_resampler()

    # ---------------------------------------------------------------------
    # FrameSerializer interface
    # ---------------------------------------------------------------------

    @property
    def type(self) -> FrameSerializerType:
        """Asterisk uses raw bytes → BINARY."""

        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        """Remember pipeline configuration."""

        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    # ------------------------------------------------------------------
    # Encoding (Pipecat → Asterisk)
    # ------------------------------------------------------------------

    async def serialize(self, frame: Frame) -> bytes | str | None:  # noqa: D401
        """Convert a Pipecat frame to a wire payload.

        Only :class:`AudioRawFrame` instances are translated – all other frame
        types are silently ignored, allowing higher-level transports to deal
        with them as needed.
        """

        if isinstance(frame, AudioRawFrame):
            try:
                # Pipeline PCM → 8-kHz μ-law
                encoded = await pcm_to_ulaw(
                    frame.audio,
                    frame.sample_rate,
                    self._asterisk_sample_rate,
                    self._resampler,
                )
                return encoded  # raw bytes
            except Exception as exc:  # pragma: no cover – robustness
                logger.error(f"AsteriskFrameSerializer.serialize: encode failed: {exc}")
                return None

        # Non-audio frames are not transmitted on the media path
        return None

    # ------------------------------------------------------------------
    # Decoding (Asterisk → Pipecat)
    # ------------------------------------------------------------------

    async def deserialize(self, data: bytes | str) -> Frame | None:  # noqa: D401
        """Convert wire payloads to Pipecat frames.

        The Asterisk media socket delivers bare μ-law bytes, therefore *data*
        must be *bytes*.  Any *str* is ignored.
        """

        if not isinstance(data, (bytes, bytearray)):
            # Unexpected payload type – just ignore
            return None

        try:
            pcm = await ulaw_to_pcm(
                bytes(data),
                self._asterisk_sample_rate,
                self._sample_rate,
                self._resampler,
            )
            return InputAudioRawFrame(
                audio=pcm,
                sample_rate=self._sample_rate,
                num_channels=1,
            )
        except Exception as exc:  # pragma: no cover
            logger.error(f"AsteriskFrameSerializer.deserialize: decode failed: {exc}")
            return None
