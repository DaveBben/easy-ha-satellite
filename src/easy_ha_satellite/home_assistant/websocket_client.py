import asyncio
import json
from collections.abc import Callable
from itertools import count
from typing import Any

import websockets
from websockets import ClientConnection, ConnectionClosed

from easy_ha_satellite.config import get_logger

from .schemas import AuthMessage, HomeAssistantConfig

logger = get_logger("socket_client")


class HASSocketClient:
    def __init__(
        self,
        config: HomeAssistantConfig,
        api_token: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        self._cfg = config
        self._ws: ClientConnection | None = None
        self._token = api_token
        self._task: asyncio.Task | None = None
        self._closed = asyncio.Event()
        self._on_event = on_event
        self._auth_ok: bool = False
        self._id_counter: int = count(1)

    async def start(self) -> None:
        if self._task and not self._task.done():
            logger.warning("Client already running")
            return
        await self._connect()
        self._task = asyncio.get_running_loop().create_task(self._recv_loop())

    async def stop(self) -> None:
        self._closed.set()
        if self._task:
            self._task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send(self, payload: Any) -> None:
        """Send JSON-able payload or bytes."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        if isinstance(payload, bytes | bytearray):
            await self._ws.send(payload)
        else:
            if hasattr(payload, "model_dump_json"):
                # pydantic model
                msg = payload.model_dump_json()
            else:
                msg = json.dumps(payload)
            await self._ws.send(msg)

    async def _connect(self) -> None:
        backoff = 1
        for _ in range(5):  # try 5 times
            try:
                self._ws = await websockets.connect(self._cfg.ws_api_url, ping_interval=20)
                logger.info("Connected to %s", self._cfg.ws_api_url)
                return
            except ConnectionRefusedError:
                logger.error("Server offline, retrying...")
            except Exception:
                logger.exception("Unexpected error connecting to HA WS")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        raise ConnectionError("Could not connect to Home Assistant websocket")

    async def _recv_loop(self) -> None:
        if self._ws is None:
            raise RuntimeError("Websocket is not initialized")
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON message: %r", raw)
                    continue
                await self._handle_message(data)
        except ConnectionClosed:
            logger.info("WebSocket closed by server")
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception:  # noqa: BLE001
            logger.exception("Fatal error in receive loop")
        finally:
            self._closed.set()

    async def send_auth(self) -> None:
        await self.send(AuthMessage(access_token=self._token))

    def get_command_id(self) -> int:
        return next(self._id_counter)

    async def _handle_message(self, data: dict[str, Any]) -> None:
        match data:
            case {"type": "auth_required"}:
                await self.send_auth()
            case {"type": "auth_ok"}:
                logger.info("Successfully authenticated to Home Assistant")
                self._auth_ok = True
            case {"type": "auth_invalid"}:
                logger.info("Authentication Failed")
                self._auth_ok = False
            case _:
                if self._on_event:
                    try:
                        self._on_event(data)
                    except Exception:
                        logger.exception("on_event callback failed")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    @property
    def on_event(self) -> Callable[[dict[str, Any]], None] | None:
        return self._on_event

    @property
    def authenticated(self) -> bool:
        return self._auth_ok

    @on_event.setter
    def on_event(self, fn: Callable[[dict[str, Any]], None] | None) -> None:
        self._on_event = fn
