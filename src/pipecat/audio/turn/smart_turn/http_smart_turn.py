#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP-based smart turn analyzer for remote ML inference.

This module provides a smart turn analyzer that sends audio data to remote
HTTP endpoints for ML-based end-of-turn detection.
"""

import asyncio
import io
from typing import Any, Dict, Optional

import aiohttp
import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnTimeoutException


class HttpSmartTurnAnalyzer(BaseSmartTurn):
    """Smart turn analyzer using HTTP-based ML inference.

    Sends audio data to remote HTTP endpoints for ML-based end-of-turn
    prediction. Handles serialization, HTTP communication, and error recovery.
    """

    def __init__(
        self,
        *,
        url: str,
        aiohttp_session: aiohttp.ClientSession,
        headers: Optional[Dict[str, str]] = None,
        service_context: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize the HTTP smart turn analyzer.

        Args:
            url: HTTP endpoint URL for the smart turn ML service.
            aiohttp_session: HTTP client session for making requests.
            headers: Optional HTTP headers to include in requests.
            service_context: Optional service context for request tracking.
            **kwargs: Additional arguments passed to BaseSmartTurn.
        """
        super().__init__(**kwargs)
        self._url = url
        self._headers = headers or {}
        self._aiohttp_session = aiohttp_session
        self._service_context = service_context

    def _serialize_array(self, audio_array: np.ndarray) -> bytes:
        """Serialize NumPy audio array to bytes for HTTP transmission."""
        logger.trace("Serializing NumPy array to bytes...")
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        serialized_bytes = buffer.getvalue()
        logger.trace(f"Serialized size: {len(serialized_bytes)} bytes")
        return serialized_bytes

    async def _send_raw_request(self, data_bytes: bytes) -> Dict[str, Any]:
        """Send raw audio data to the HTTP endpoint for prediction."""
        headers = {"Content-Type": "application/octet-stream"}
        headers.update(self._headers)

        # Add service context as header if available
        if hasattr(self, "_service_context") and self._service_context is not None:
            headers["X-Service-Context"] = str(self._service_context)

        try:
            timeout = aiohttp.ClientTimeout(total=self._params.stop_secs)

            async with self._aiohttp_session.post(
                self._url, data=data_bytes, headers=headers, timeout=timeout
            ) as response:
                logger.trace("\n--- Response ---")
                logger.trace(f"Status Code: {response.status}")

                # Check if successful
                if response.status != 200:
                    error_text = await response.text()
                    logger.trace("Response Content (Error):")
                    logger.trace(error_text)

                    if response.status == 500:
                        logger.warning(f"Smart turn service returned 500 error: {error_text}")
                        raise Exception(f"Server returned HTTP 500: {error_text}")
                    else:
                        response.raise_for_status()

                # Process successful response
                try:
                    json_data = await response.json()
                    logger.trace("Response JSON:")
                    logger.trace(json_data)
                    return json_data
                except aiohttp.ContentTypeError:
                    # Non-JSON response
                    text = await response.text()
                    logger.trace("Response Content (non-JSON):")
                    logger.trace(text)
                    raise Exception(f"Non-JSON response: {text}")

        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self._params.stop_secs} seconds")
            raise SmartTurnTimeoutException(f"Request exceeded {self._params.stop_secs} seconds.")
        except aiohttp.ClientError as e:
            logger.error(f"Failed to send raw request to Daily Smart Turn: {e}")
            raise Exception("Failed to send raw request to Daily Smart Turn.")

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Predict end-of-turn using remote HTTP ML service."""
        try:
            serialized_array = self._serialize_array(audio_array)
            return await self._send_raw_request(serialized_array)
        except Exception as e:
            logger.error(f"Smart turn prediction failed: {str(e)}")
            # Return an incomplete prediction when a failure occurs
            return {
                "prediction": 0,
                "probability": 0.0,
                "metrics": {"inference_time": 0.0, "total_time": 0.0},
            }
