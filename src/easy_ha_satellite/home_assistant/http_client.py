import asyncio
from typing import Any
from urllib.parse import urljoin

import aiohttp

from easy_ha_satellite.config import get_logger

from .schemas import HomeAssistantConfig

logger = get_logger("http_client")



class HASSHttpClient:
    def __init__(self, config: HomeAssistantConfig, api_token: str):
        self._cfg = config
        self._token = api_token
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session and not self._session.closed:
            logger.warning("HTTP client session already started.")
            return

        headers = {"Authorization": f"Bearer {self._token}"}
        self._session = aiohttp.ClientSession(headers=headers)
        logger.info("HASS HTTP Client started.")

    async def stop(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("HASS HTTP Client stopped.")
        self._session = None

    def _get_url(self, path: str) -> str:
        """Constructs a full API URL from a relative path."""
        # Joins base URL (e.g., 'http://host/api/') with a path (e.g., 'tts_proxy/...')
        # lstrip('/') prevents urljoin from treating the path as absolute
        return urljoin(self._cfg.http_api_root, path.lstrip("/"))

    async def download_media(self, path: str, chunk_size: int = 8192) -> bytes:
        """
        Downloads media (like TTS audio) from a given Home Assistant path.

        Args:
            path: The relative path to the media resource (e.g., 'tts_proxy/...')
            chunk_size: The size of chunks to download in bytes.

        Returns:
            The complete media content as bytes.

        Raises:
            RuntimeError: If the client session is not started.
            aiohttp.ClientError: For connection or HTTP status errors.
        """
        if not self._session:
            raise RuntimeError("HTTP Client session not started. Call start() first.")

        url = self._get_url(path)
        timeout = aiohttp.ClientTimeout(total=20, sock_read=15)
        buffer = bytearray()

        logger.info(f"Downloading media from: {url}")
        try:
            async with self._session.get(url, timeout=timeout) as resp:
                # Raise an exception for non-2xx status codes
                resp.raise_for_status()

                async for chunk in resp.content.iter_chunked(chunk_size):
                    if chunk:
                        buffer.extend(chunk)

            logger.debug(f"Successfully downloaded {len(buffer)} bytes.")
            return bytes(buffer)

        except asyncio.CancelledError:
            logger.warning("Media download was cancelled.")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Media download failed: {e}")
            raise

    async def post(self, path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        """
        Sends a POST request with a JSON payload to a Home Assistant path.

        Args:
            path: The relative path for the API endpoint.
            json_data: A dictionary to be sent as the JSON body.

        Returns:
            The JSON response from the server as a dictionary.
        """
        if not self._session:
            raise RuntimeError("HTTP Client session not started. Call start() first.")

        url = self._get_url(path)
        timeout = aiohttp.ClientTimeout(total=10)

        logger.info(f"POSTing to {url}")
        try:
            async with self._session.post(url, json=json_data, timeout=timeout) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"POST request failed: {e}")
            raise

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()
