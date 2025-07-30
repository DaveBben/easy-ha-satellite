import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from easy_ha_satellite.config import get_logger

from .schemas import AssistPipelineRun
from .websocket_client import HASSocketClient

logger = get_logger("pipeline")


class PipelineEventType(Enum):
    RUN_START = "RUN_START"
    RUN_END = "RUN_END"
    STT_START = "STT_START"
    STT_END = "STT_END"
    STT_VAD_START = "STT_VAD_START"
    STT_VAD_END = "STT_VAD_END"
    INTENT_START = "INTENT_START"
    INTENT_END = "INTENT_END"
    TTS_START = "TTS_START"
    TTS_END = "TTS_END"
    ERROR = "ERROR"
    WAKE_WORD_START = "WAKE_WORD_START"
    WAKE_WORD_END = "WAKE_WORD_END"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    type: PipelineEventType
    data: Any | None = None


class Pipeline:
    def __init__(
        self,
        client: HASSocketClient,
        sample_rate: int,
        timeout_secs: int = 30,
        start_stage: str = "stt",
        end_stage: str = "tts",
    ):
        self._client = client
        self._id = client.get_command_id()
        self._stt_binary_handler_id = 0
        self._timeout = timeout_secs
        self._sample_rt = sample_rate
        self._media_url = None
        self._stt_output = None
        self._start_stage = start_stage
        self._end_stage = end_stage
        self._done = False
        self._events: asyncio.Queue[PipelineEvent] = asyncio.Queue()
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {
            "run-start": self._on_run_start,
            "run-end": self._on_run_end,
            "stt-start": self._on_stt_start,
            "stt-end": self._on_stt_end,
            "tts-start": self._on_tts_start,
            "tts-end": self._on_tts_end,
            "stt-vad-start": self.on_stt_vad_start,
            "wake_word-start": self._on_wake_word_start,
            "wake_word-end": self._on_wake_word_end,
            "stt-vad-end": self._on_stt_vad_end,
            "error": self._on_error,
        }
        self._messages: list[dict[str, Any]] = []

    async def send_audio(self, chunk: bytes) -> None:
        handler_id = bytes([self._stt_binary_handler_id])
        await self._client.send(handler_id + chunk)

    async def get_events(self, timeout: float | None = None) -> PipelineEvent:
        if timeout is None:
            return await self._events.get()
        return await asyncio.wait_for(self._events.get(), timeout)

    async def start(self) -> None:
        logger.debug("Sending Pipeline Run Command")
        await self._client.send(
            AssistPipelineRun(
                id=self._id,
                start_stage=self._start_stage,
                end_stage=self._end_stage,
                timeout=self._timeout,
                input={"sample_rate": self._sample_rt},
            )
        )

    async def _on_wake_word_start(self, msg: dict[str, Any]) -> None:
        logger.debug("Received Wake Word Start Event")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.WAKE_WORD_START))

    async def _on_wake_word_end(self, msg: dict[str, Any]) -> None:
        logger.debug("Received Wake Word End Event")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.WAKE_WORD_END))

    async def _on_tts_start(self, msg: dict[str, Any]) -> None:
        logger.debug("Received TTS Start Event")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.TTS_START))

    async def on_stt_vad_start(self, msg: dict[str, Any]) -> None:
        logger.debug("Received STT VAD Start Event")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.STT_VAD_START))

    async def _on_stt_vad_end(self, msg: dict[str, Any]) -> None:
        logger.debug("Received STT VAD End Event")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.STT_VAD_END))

    async def _on_stt_start(self, msg: dict[str, Any]) -> None:
        logger.debug("Received STT Start Event")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.STT_START))

    async def _on_run_start(self, msg: dict[str, Any]) -> None:
        logger.debug("Received Run Start Event")
        try:
            self._stt_binary_handler_id = msg["data"]["runner_data"]["stt_binary_handler_id"]
        except KeyError:
            logger.warning("Could not find stt_binary_handler_id")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.RUN_START))

    async def _on_run_end(self, msg: dict[str, Any]) -> None:
        logger.debug("Received Run End Event")
        self._done = True
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.RUN_END))

    async def _on_stt_end(self, msg: dict[str, Any]) -> None:
        logger.debug("Received STT End Event")
        try:
            self._stt_output = msg["data"]["stt_output"]["text"]
            logger.info(f"Phrase: {self._stt_output}")
        except KeyError:
            logger.warning("STT complete, but no phrase transcribed")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.STT_END))

    async def _on_tts_end(self, msg: dict[str, Any]) -> None:
        logger.debug("Received TTS End Event")
        try:
            self._media_url = msg["data"]["tts_output"]["url"]
        except KeyError:
            logger.warning("Could not retrieve output media url")
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.TTS_END))

    async def _on_error(self, msg: dict[str, Any]) -> None:
        logger.debug("Received Error Event")
        self._done = True
        await self._events.put(PipelineEvent(data=msg, type=PipelineEventType.ERROR))

    def __aiter__(self):
        return self

    async def __anext__(self) -> PipelineEvent:
        if not self._done:
            item = await self._events.get()
            if item is None:
                raise StopAsyncIteration
            return item
        raise StopAsyncIteration

    def handle_event(self, msg: dict[str, Any]) -> None:
        self._messages.append(msg)
        match msg:
            case {"type": "result", "success": False}:
                logger.warning(f"Something went wrong: {msg}")
                # TODO: Raise pipeline error
                self._done = True
            case {"type": "event"}:
                evt = msg.get("event")
                etype = evt.get("type")
                handler = self._handlers.get(etype)
                if handler:
                    asyncio.get_running_loop().create_task(handler(evt))
                else:
                    logger.debug("No handler for message type %s", etype)
            case _:
                logger.debug(msg)

    @property
    def running(self) -> bool:
        return not self._done

    @property
    def stt_output(self) -> str:
        return self._stt_output

    @property
    def media_url(self) -> str | None:
        return self._media_url

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages
