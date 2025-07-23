"""
This program acts as a home assistant satellite. It runs a local wakeword model. When
the wakeword is detected, it will forward audio to Home Assistant for Speech to text
and to handle intents.
"""

import asyncio
import io
import json
import logging
import multiprocessing as mp
import os
import queue
import threading
from dataclasses import dataclass
from enum import Enum
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openwakeword
import requests
import sounddevice as sd
import soundfile as sf
import uvloop
import websockets
from openwakeword.model import Model
from websockets.exceptions import ConnectionClosed

# Environment Variables
HASS_TOKEN = os.environ.get("HASS_TOKEN")
BASE_URL = os.environ.get("BASE_URL")
SECURE_WEBSOCKET = os.environ.get("SECURE_WEBSOCKET", "true")
INPUT_DEVICE_NAME = os.environ.get("INPUT_AUDIO_DEVICE", None)
OUTPUT_DEVICE_NAME = os.environ.get("OUTPUT_AUDIO_DEVICE", None)
WAKEWORD_MODEL = os.environ.get("WAKEWORD_MODEL_NAME", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(os.environ.get("WW_THRESHOLD", "0.5"))
INFERENCE_FRAMEWORK = os.environ.get("INFERENCE_FRAMEWORK", "onnx")

# Audio Config
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "dtype": "int16",
    "chunk_ms": 10,
}


# Sound files
PROJECT_ROOT = Path(__file__).resolve().parent
SOUND_DIR = PROJECT_ROOT / "sounds"
AUDIO_CLIPS = {
    "listen_start": sf.read(SOUND_DIR / "listen_start.wav", dtype="int16"),
    "task_complete": sf.read(SOUND_DIR / "task_complete.wav", dtype="int16"),
    "connected": sf.read(SOUND_DIR / "connected.wav", dtype="int16"),
    "error": sf.read(SOUND_DIR / "failed.wav", dtype="int16"),
}

# Server
WS_SERVER_URL = f"{BASE_URL}/api/websocket"
RECONNECT_DELAY_SECONDS = 5  # Time to wait before trying to reconnect
WS_CONFIG = {"ping_interval": 20, "ping_timeout": 20}
PIPELINE_ID = 1
PIPELINE_TIMEOUT_SECS = 60

# WakeWord
REQUIRED_WAKEWORD_MS = 80
COOLDOWN_MS = 3000
WAKEWORD_EVENT = "WAKEWORD_DETECTED"

# Output Audio
OUTPUT_SR = 48_000
OUTPUT_CH = 2
_play_lock = threading.Lock()
_output_stream = sd.OutputStream(
    samplerate=OUTPUT_SR,
    channels=OUTPUT_CH,
    device=OUTPUT_DEVICE_NAME,
    dtype="int16",
    latency=0.01,
)
_output_stream.start()


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CLIENT")


# Application events
class ServerEventType(Enum):
    STT_STARTED = "stt_started"
    STT_COMPLETED = "stt_completed"
    ACTION_FAILED = "action_failed"
    TTS_COMPLETED = "tts_completed"
    ACTION_SUCCESS = "action_success"


@dataclass
class ServerEvent:
    type: ServerEventType
    data: Optional[Any] = None


def wakeword_process_worker(
    in_audio_q: mp.Queue,
    app_event_q: mp.Queue,
    model_name: str,
    inference_framework: str,
    shutdown_event: Event,
    wake_word_run_event: Event,
):
    try:
        oww_model = Model(
            wakeword_models=[model_name], inference_framework=inference_framework
        )
    except Exception:
        logging.critical("Failed to load model in subprocess.", exc_info=True)
        return
    logging.info(f"Listening for {model_name}...")
    while not shutdown_event.is_set():
        try:
            wake_word_run_event.wait()
            while wake_word_run_event.is_set():
                chunk = in_audio_q.get(timeout=AUDIO_CONFIG["chunk_ms"] / 1000)
                if chunk is None:  # Sentinel value to stop the process
                    break
                # Wakeword Prediction
                prediction = oww_model.predict(np.frombuffer(chunk, dtype=np.int16))
                if prediction[model_name] > WAKE_WORD_THRESHOLD:
                    logger.info(
                        f"Wakeword detected! (Score: {prediction[model_name]:.2f})"
                    )
                    app_event_q.put(WAKEWORD_EVENT)
                    oww_model.reset()

        except queue.Empty:
            # This is expected when no audio is coming in, just continue
            continue
        except KeyboardInterrupt:
            break
        except Exception:
            logger.error("Error in Keyword detection process", exc_info=True)


def _to_output_format(data: np.ndarray, sr: int) -> np.ndarray:
    # Resample if needed
    if sr != OUTPUT_SR:
        ratio = OUTPUT_SR / sr
        idx = (np.arange(int(len(data) * ratio)) / ratio).astype(np.int32)
        idx = np.clip(idx, 0, len(data) - 1)
        data = data[idx]

    # Mono -> stereo
    if data.ndim == 1 or data.shape[1] == 1:
        data = np.repeat(data.reshape(-1, 1), OUTPUT_CH, axis=1)

    return data.astype("int16", copy=False)


async def play_sound_task(out_audio_q: asyncio.Queue):
    loop = asyncio.get_running_loop()
    logger.info("Streaming Out task started")
    try:
        while True:
            audio, sr = await out_audio_q.get()
            pcm = _to_output_format(audio, sr)
            if not _output_stream.active:
                _output_stream.start()
            try:
                await loop.run_in_executor(None, _output_stream.write, pcm)
            except sd.PortAudioError as e:
                # Expected because we abort
                logger.debug(e)
    except asyncio.CancelledError:
        logger.info("play sound task stopped")
    except Exception:
        logger.exception("Play sound task encountered a fatal error.")
        raise


async def handle_server_events_task(
    event_q: asyncio.Queue, out_audio_q: asyncio.Queue, stream_audio: asyncio.Event
):
    logger.info("Handle Events Task Started")
    try:
        while True:
            event: ServerEvent = await event_q.get()
            match event.type:
                case ServerEventType.STT_STARTED:
                    await out_audio_q.put(AUDIO_CLIPS["listen_start"])
                    # Set flag that audio stream is ready
                    stream_audio.set()
                case ServerEventType.STT_COMPLETED:
                    # Stop Audio Stream
                    stream_audio.clear()
                    logger.info(f"Phrase: {event.data}")
                case ServerEventType.ACTION_SUCCESS:
                    await out_audio_q.put(AUDIO_CLIPS["task_complete"])
                case ServerEventType.ACTION_FAILED:
                    await out_audio_q.put(AUDIO_CLIPS["error"])
                    logger.warning(event.data)
                    stream_audio.clear()
                    # Stop Processing Events
                    break
                case ServerEventType.TTS_COMPLETED:
                    media_url = event.data
                    audio_mp3 = download_tts_bytes(media_url)
                    pcm, sr = sf.read(io.BytesIO(audio_mp3), dtype="int16")
                    await out_audio_q.put((pcm, sr))
                    # Stop Processing Events
                    break
                case _:
                    pass
    except asyncio.CancelledError:
        logger.info("Handle Events task stopped")
    except Exception:
        logger.exception("Handle Events task encountered a fatal error.")
        raise


def download_tts_bytes(path: str, chunk_size: int = 8192) -> bytes:
    url = f"http://{BASE_URL}{path}"
    logger.debug(f"Grabbing audio from: {url}")
    headers = {"Authorization": f"Bearer {HASS_TOKEN}"}
    buf = bytearray()
    with requests.get(url, headers=headers, stream=True, timeout=15) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            buf.extend(chunk)
    return bytes(buf)


async def handle_server_msg_task(
    websocket: websockets.ClientConnection, event_q: asyncio.Queue
) -> None:
    """Handle incoming messages from the server and play sounds."""
    logger.info("Message handler task started.")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                match data:
                    case {"type": "auth_required"}:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "auth",
                                    "access_token": HASS_TOKEN,
                                }
                            )
                        )
                    case {"type": "auth_ok"}:
                        logger.info("Successfully authenticated with Home Assistant")
                        await websocket.send(
                            json.dumps(
                                {
                                    "id": PIPELINE_ID,
                                    "type": "assist_pipeline/run",
                                    "start_stage": "stt",
                                    "timeout": PIPELINE_TIMEOUT_SECS,
                                    "end_stage": "tts",
                                    "input": {
                                        "sample_rate": AUDIO_CONFIG["sample_rate"]
                                    },
                                }
                            )
                        )
                    # Speech to text started
                    case {"event": {"type": "stt-start"}}:
                        await event_q.put(ServerEvent(type=ServerEventType.STT_STARTED))

                    # Speech to Text Complete
                    case {"event": {"type": "stt-end"}}:
                        try:
                            phrase = data["event"]["data"]["stt_output"]["text"]
                            await event_q.put(
                                ServerEvent(
                                    type=ServerEventType.STT_COMPLETED, data=phrase
                                )
                            )
                        except KeyError:
                            logger.warning("STT complete, but no phrase transcribed")

                    # Pipeline Error
                    case {"event": {"type": "error"}}:
                        try:
                            err_msg = data["event"]["data"]["message"]
                            await event_q.put(
                                ServerEvent(
                                    type=ServerEventType.ACTION_FAILED, data=err_msg
                                )
                            )
                        except KeyError:
                            logger.warning(
                                "Pipeline Error but could not retrieve response"
                            )
                        break

                    # Audio Response Available
                    case {"event": {"type": "tts-end"}}:
                        try:
                            media_url = data["event"]["data"]["tts_output"]["url"]
                            await event_q.put(
                                ServerEvent(
                                    type=ServerEventType.TTS_COMPLETED, data=media_url
                                )
                            )
                        except Exception as ex:
                            logger.warning(f"Failed to play output media: {ex}")
                        finally:
                            break

                    # Action Successful
                    case {
                        "event": {
                            "data": {
                                "intent_output": {
                                    "response": {"response_type": "action_done"}
                                }
                            }
                        }
                    }:
                        await event_q.put(
                            ServerEvent(type=ServerEventType.ACTION_SUCCESS)
                        )

                    # Action Failed
                    case {
                        "event": {
                            "data": {
                                "intent_output": {
                                    "response": {"response_type": "error"}
                                }
                            }
                        }
                    }:
                        await event_q.put(
                            ServerEvent(type=ServerEventType.ACTION_FAILED)
                        )
                    case _:
                        logger.debug(f"Server message: {data}")

            except json.JSONDecodeError:
                logger.info(f"Received non-JSON server response: {message}")
            except Exception as e:
                logger.error(f"Error processing server message: {e}")
                break

    except ConnectionClosed:
        logger.info("Connection closed")
        raise
    except asyncio.CancelledError:
        logger.info("handle_server_message tasks stopped")
    except Exception:
        logger.exception("Message handler encountered a fatal error.")
        raise


async def stream_audio_task(
    websocket: websockets.ClientConnection,
    in_audio_q: mp.Queue,
    stream_audio: asyncio.Event,
):
    loop = asyncio.get_running_loop()
    try:
        await stream_audio.wait()
        logger.info("🎤 Microphone stream is now active.")
        while stream_audio.is_set():
            chunk = await loop.run_in_executor(None, in_audio_q.get)
            if chunk is None:
                break
            # Need to send handler ID as part of audio
            handler_id = bytes([PIPELINE_ID])
            await websocket.send(handler_id + chunk)
    except ConnectionClosed:
        logger.info("Stopped streaming audio to server.")
    except asyncio.CancelledError:
        logger.info("stream audio tasks stopped")
    except Exception:
        logger.exception("Stream Audio task encountered a fatal error.")
        raise
    finally:
        logger.info("🛑 Microphone stream is now stopped")


async def handle_app_events(
    in_audio_q: mp.Queue,
    out_audio_q: asyncio.Queue,
    app_event_queue: mp.Queue,
    shutdown_event: Event,
    wake_word_run_event: Event,
) -> None:
    """Waits for a wakeword detection event and then manages the server interaction."""
    loop = asyncio.get_running_loop()

    # Start Wakeword Loop
    wake_word_run_event.set()

    stream_audio = asyncio.Event()
    server_event_q = asyncio.Queue()

    # Create Server URL
    server_url = f"wss://{WS_SERVER_URL}"
    if str(SECURE_WEBSOCKET.lower()) == "false":
        server_url = f"ws://{WS_SERVER_URL}"
    logger.info(f"Using HomeAssistant URl {server_url}")

    # Main Loop
    while not shutdown_event.is_set():
        event: ServerEvent = await loop.run_in_executor(None, app_event_queue.get)
        if event is None:
            break

        if event == WAKEWORD_EVENT:
            # Stop any playing audio and clear queue
            _output_stream.abort()
            while not out_audio_q.empty():
                out_audio_q.empty()
            # Stop WakeWord Loop
            wake_word_run_event.clear()
            try:
                # Connect to server
                async with websockets.connect(server_url, **WS_CONFIG) as websocket:
                    try:
                        logger.info("✅ WebSocket connection established.")
                        stream_audio.set()
                        async with asyncio.TaskGroup() as tg:
                            tg.create_task(
                                handle_server_msg_task(websocket, server_event_q)
                            )
                            tg.create_task(
                                handle_server_events_task(
                                    server_event_q, out_audio_q, stream_audio
                                )
                            )
                            tg.create_task(
                                stream_audio_task(websocket, in_audio_q, stream_audio)
                            )
                    except* Exception as eg:
                        for exc in eg.exceptions:
                            if not isinstance(
                                exc,
                                (
                                    ConnectionClosed,
                                    asyncio.CancelledError,
                                    TimeoutError,
                                ),
                            ):
                                logger.error(
                                    f"Unhandled exception in task group: {exc!r}"
                                )

            except ConnectionRefusedError:
                logger.warning("Server is offline. Could not connect.")
                await out_audio_q.put(AUDIO_CLIPS["error"])
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
            except Exception:
                logger.exception("An error occurred during WebSocket communication.")
                await out_audio_q.put(AUDIO_CLIPS["error"])
            finally:
                while not in_audio_q.empty():
                    in_audio_q.get_nowait()
                logger.info("Ready to listen for wakeword")
                wake_word_run_event.set()


async def run_client(
    in_audio_q: mp.Queue, app_event_queue: mp.Queue, shutdown_event, wake_word_run_event
) -> None:
    """Main client function with a persistent reconnection loop."""

    def audio_callback(indata, frames, time, status):
        """This is called from a separate thread for each audio block."""
        if status:
            logger.warning(status)
        try:
            in_audio_q.put_nowait(bytes(indata))
        except queue.Full:
            logger.warning("Queue is full. Not taking items off fast enough")

    while True:
        try:
            with sd.RawInputStream(
                samplerate=AUDIO_CONFIG["sample_rate"],
                channels=AUDIO_CONFIG["channels"],
                dtype=AUDIO_CONFIG["dtype"],
                callback=audio_callback,
                device=INPUT_DEVICE_NAME,
                blocksize=int(
                    AUDIO_CONFIG["sample_rate"] * (AUDIO_CONFIG["chunk_ms"] / 1000)
                ),
            ):
                try:
                    out_audio_q = asyncio.Queue()
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(
                            handle_app_events(
                                in_audio_q=in_audio_q,
                                out_audio_q=out_audio_q,
                                app_event_queue=app_event_queue,
                                shutdown_event=shutdown_event,
                                wake_word_run_event=wake_word_run_event,
                            )
                        )
                        tg.create_task(play_sound_task(out_audio_q=out_audio_q))
                except* Exception as eg:
                    for exc in eg.exceptions:
                        if not isinstance(
                            exc,
                            (
                                ConnectionClosed,
                                asyncio.CancelledError,
                                TimeoutError,
                            ),
                        ):
                            logger.error(f"Unhandled exception in task group: {exc!r}")

        except sd.PortAudioError as e:
            logger.critical(
                f"Audio device error: {e}. Please check your microphone and restart."
            )
            raise
        except Exception:
            logger.exception(
                f"An unexpected error occurred. Retrying in {RECONNECT_DELAY_SECONDS}s..."
            )


def main() -> None:
    if not HASS_TOKEN:
        raise ValueError("Home Assistant Long Lived token is not set!")
    if not BASE_URL:
        raise ValueError("Home Assistant Base URL is not set!")
    try:
        print("\n" + "-" * 50)
        print("Available audio input devices:")
        print(sd.query_devices())
        print("-" * 50)
        if INPUT_DEVICE_NAME:
            logger.info(
                f"Attempting to use specified input device: {INPUT_DEVICE_NAME}"
            )
        if OUTPUT_DEVICE_NAME:
            logger.info(
                f"Attempting to use specified output device: {OUTPUT_DEVICE_NAME}"
            )
        else:
            logger.info("Using default audio device.")

        # Download Wakeword Model
        openwakeword.utils.download_models(model_names=[WAKEWORD_MODEL])

        # Create Mulitprocessing Queue
        in_audio_q = mp.Queue()
        app_event_queue = mp.Queue()
        # Shutdown tasks and threads
        shutdown_event = mp.Event()
        # Signal whether we are streaming
        wake_word_run_event = mp.Event()

        # Start Dedicated Wakeword Process
        wakeword_process = mp.Process(
            target=wakeword_process_worker,
            args=(
                in_audio_q,
                app_event_queue,
                WAKEWORD_MODEL,
                INFERENCE_FRAMEWORK,
                shutdown_event,
                wake_word_run_event,
            ),
            daemon=True,  # auto-terminate with the main process
        )
        wakeword_process.start()

        loop = uvloop.new_event_loop()
        asyncio.set_event_loop(loop)
        main_task = loop.create_task(
            run_client(in_audio_q, app_event_queue, shutdown_event, wake_word_run_event)
        )

        try:
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            logger.info("Client stopping due to user request...")
        finally:
            logger.info("Shutting down...")

            shutdown_event.set()
            # Gracefully cancel all running asyncio tasks
            for task in asyncio.all_tasks(loop=loop):
                task.cancel()

            # Wait for tasks to finish cancelling
            try:
                # gather cancelled tasks to allow them to finish
                loop.run_until_complete(
                    asyncio.gather(
                        *asyncio.all_tasks(loop=loop), return_exceptions=True
                    )
                )
            except asyncio.CancelledError:
                pass  # This is expected

            # Now close the loop
            loop.close()

            # join the background process
            wakeword_process.join(timeout=2)
            if wakeword_process.is_alive():
                wakeword_process.terminate()

    except Exception:
        logger.critical("A fatal error occurred in main.", exc_info=True)
        raise


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
