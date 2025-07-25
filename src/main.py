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
from urllib.parse import urljoin

import aiohttp
import numpy as np
import openwakeword
import sounddevice as sd
import soundfile as sf
import uvloop
import websockets
from openwakeword.model import Model
from websockets.exceptions import ConnectionClosed

# Environment Variables
HASS_TOKEN = os.environ.get("HASS_TOKEN")
HASS_HOST = os.environ.get("HASS_HOST")
HASS_PORT = os.environ.get("HASS_PORT")

SECURE_WEBSOCKET = os.environ.get("SECURE_WEBSOCKET", "true").lower() == "true"
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
if not SECURE_WEBSOCKET:
    BASE_URL = f"http://{HASS_HOST}:{HASS_PORT}"
    WS_SERVER_URL = f"ws://{HASS_HOST}:{HASS_PORT}/api/websocket"
else:
    BASE_URL = f"https://{HASS_HOST}:{HASS_PORT}"
    WS_SERVER_URL = f"wss://{HASS_HOST}:{HASS_PORT}/api/websocket"

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
    model_name: str,
    inference_framework: str,
    shutdown_event: Event,
    wake_event: Event,
):
    try:
        oww_model = Model(
            wakeword_models=[model_name], inference_framework=inference_framework
        )
    except Exception:
        logging.critical("Failed to load model in subprocess.", exc_info=True)
        return
    logging.info(f"Listening for {model_name}...")
    try:
        while not shutdown_event.is_set():
            try:
                while not wake_event.is_set():
                    chunk = in_audio_q.get(timeout=AUDIO_CONFIG['chunk_ms'] / 1000)
                    if chunk is None:
                        break
                    # Wakeword Prediction
                    prediction = oww_model.predict(np.frombuffer(chunk, dtype=np.int16))
                    if prediction[model_name] > WAKE_WORD_THRESHOLD:
                        logger.info(
                            f"Wakeword detected! (Score: {prediction[model_name]:.2f})"
                        )
                        wake_event.set()
                        oww_model.reset()

            except queue.Empty:
                # This is expected when no audio is coming in, just continue
                continue
            except KeyboardInterrupt:
                break
            except Exception:
                logger.error("Error in Keyword detection process", exc_info=True)
    except KeyboardInterrupt:
        pass


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


def play_sound_thread(out_audio_q: queue.Queue, shutdown_event: Event, abort_playback: Event):
    chunk_frames = int(OUTPUT_SR * 0.02)
    try:
        with sd.OutputStream(
            samplerate=OUTPUT_SR,
            channels=OUTPUT_CH,
            device=OUTPUT_DEVICE_NAME,
            dtype="int16",
            latency=0.01,
        ) as stream:
            while not shutdown_event.is_set():
                try:
                    audio, sr = out_audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                # In case flag got set while we were waiting
                if abort_playback.is_set():
                    try:
                        stream.abort()
                    except sd.PortAudioError: 
                        pass
                    abort_playback.clear()
                stream.start()

                pcm = _to_output_format(audio, sr)
                # Write in small slices so we can abort if needed
                for start in range(0, len(pcm), chunk_frames):
                    if abort_playback.is_set() or shutdown_event.is_set():
                        try:
                            stream.abort()
                        except sd.PortAudioError: 
                            pass
                        while not out_audio_q.empty():
                            out_audio_q.get_nowait()
                        abort_playback.clear()
                        break

                    slice_ = pcm[start:start + chunk_frames]
                    stream.write(slice_)
    except Exception:
        logger.exception("Play sound thread crashed")


async def handle_server_events_task(
    event_q: asyncio.Queue, out_audio_q: queue.Queue, stream_audio: asyncio.Event
):
    loop = asyncio.get_running_loop()
    logger.info("Handle Events Task Started")
    try:
        while True:
            event: ServerEvent = await event_q.get()
            match event.type:
                case ServerEventType.STT_STARTED:
                    loop.run_in_executor(None, out_audio_q.put, AUDIO_CLIPS["listen_start"])
                    # Set flag that audio stream is ready
                    stream_audio.set()
                case ServerEventType.STT_COMPLETED:
                    # Stop Audio Stream
                    stream_audio.clear()
                    logger.info(f"Phrase: {event.data}")
                case ServerEventType.ACTION_SUCCESS:
                    loop.run_in_executor(None, out_audio_q.put, AUDIO_CLIPS["task_complete"])
                case ServerEventType.ACTION_FAILED:
                    loop.run_in_executor(None, out_audio_q.put, AUDIO_CLIPS["error"])
                    logger.warning(event.data)
                    stream_audio.clear()
                    # Stop Processing Events
                    break
                case ServerEventType.TTS_COMPLETED:
                    media_url = event.data
                    audio_mp3 = await download_tts_bytes(media_url)
                    pcm, sr = sf.read(io.BytesIO(audio_mp3), dtype="int16")
                    loop.run_in_executor(None, out_audio_q.put, (pcm, sr))
                    # Stop Processing Events
                    break
                case _:
                    pass
    except asyncio.CancelledError:
        logger.info("Handle Events task stopped")
    except Exception:
        logger.exception("Handle Events task encountered a fatal error.")
        raise


async def download_tts_bytes(
    path: str,
    chunk_size: int = 8192,
) -> bytes:
    url = urljoin(BASE_URL.rstrip("/") + "/", path.lstrip("/"))
    timeout = aiohttp.ClientTimeout(total=15, sock_read=10)
    buf = bytearray()

    async with aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {HASS_TOKEN}"},
        timeout=timeout,
    ) as session:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(chunk_size):
                    if not chunk:
                        continue
                    buf.extend(chunk)
            return bytes(buf)

        except asyncio.CancelledError:
            raise
        except aiohttp.ClientError as e:
            logger.error(f"TTS download failed: {e}")
            raise


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


def clear_queue(q: mp.Queue):
    while not q.empty():
        q.get_nowait()


async def handle_app_events(
    in_audio_q: mp.Queue,
    out_audio_q: queue.Queue,
    shutdown_event: Event,
    wake_event: Event,
    abort_playback: Event
) -> None:
    """Waits for a wakeword detection event and then manages the server interaction."""

    stream_audio = asyncio.Event()
    server_event_q = asyncio.Queue()

    # Create Server URL
    logger.info(f"Using HomeAssistant URl {WS_SERVER_URL}")

    # Main Loop
    while not shutdown_event.is_set():
        if wake_event.is_set():
            # Stop any playing audio and clear queue
            abort_playback.set()
            try:
                # Connect to server
                async with websockets.connect(WS_SERVER_URL, **WS_CONFIG) as websocket:
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
                asyncio.get_event_loop().run_in_executor(None, out_audio_q.put, AUDIO_CLIPS["error"])
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
            except Exception:
                logger.exception("An error occurred during WebSocket communication.")
                asyncio.get_event_loop().run_in_executor(None, out_audio_q.put, AUDIO_CLIPS["error"])
                break
            finally:
                clear_queue(in_audio_q)
                wake_event.clear()
                logger.info("Ready to listen for wakeword")


async def run_client(in_audio_q: mp.Queue,out_audio_q: queue.Queue, shutdown_event: Event, wake_event: Event, abort_playback_event: Event) -> None:
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
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(
                            handle_app_events(
                                in_audio_q=in_audio_q,
                                out_audio_q=out_audio_q,
                                shutdown_event=shutdown_event,
                                wake_event=wake_event,
                                abort_playback=abort_playback_event
                            )
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
    if not HASS_PORT or not HASS_HOST:
        raise ValueError("Home Assistant post and host needs to be set!")
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
        out_audio_q = queue.Queue()
        # Shutdown tasks and threads
        shutdown_event = mp.Event()
        # Signal whether we are streaming
        wake_event = mp.Event()
        # Signal to abort audio playback
        abort_playback = mp.Event()

        # Start Dedicated Wakeword Process
        wakeword_process = mp.Process(
            target=wakeword_process_worker,
            args=(
                in_audio_q,
                WAKEWORD_MODEL,
                INFERENCE_FRAMEWORK,
                shutdown_event,
                wake_event,
            ),
            daemon=True,  # auto-terminate with the main process
        )
        wakeword_process.start()

        # Play sound
        player_thr = threading.Thread(
            target=play_sound_thread,
            args=(out_audio_q, shutdown_event, abort_playback),   # pass your asyncio.Queue and the event loop
            daemon=True,
        )
        player_thr.start()

        loop = uvloop.new_event_loop()
        asyncio.set_event_loop(loop)
        main_task = loop.create_task(run_client(in_audio_q, out_audio_q, shutdown_event, wake_event, abort_playback))

        try:
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            logger.info("Client stopping due to user request...")
        finally:
            logger.info("Shutting down...")

            shutdown_event.set()
            main_task.cancel()
            loop.close()

            # join the background process
            wakeword_process.join(timeout=2)
            player_thr.join(timeout=2)
            if wakeword_process.is_alive():
                wakeword_process.terminate()

    except Exception:
        logger.critical("A fatal error occurred in main.", exc_info=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
