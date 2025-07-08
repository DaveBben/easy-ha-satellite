import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import openwakeword
import sounddevice as sd
import soundfile as sf
import uvloop
import websockets
from openwakeword.model import Model
from websockets.exceptions import ConnectionClosed

# Envrionment Variables
BASE_URL = os.environ.get("BASE_URL", "192.168.1.16:9292")
INPUT_DEVICE_NAME = os.environ.get("INPUT_AUDIO_DEVICE", None)
WAKEWORD_MODEL = os.environ.get("WAKEWORD_MODEL_NAME", "hey_jarvis")
WAKE_WORD_THRESHOLD = os.environ.get("WW_THRESHOLD", "0.5")
INFERENCE_FRAMEWORK = os.environ.get("INFERENCE_FRAMEWORK", "onnx")
INFERENCE_THREADS = os.environ.get("INFERENCE_THREADS", os.cpu_count())

# Sound files
PROJECT_ROOT = Path(__file__).resolve().parent
SOUND_DIR = PROJECT_ROOT / "sounds"
AUDIO_CLIPS = {
    "listen_start": sf.read(SOUND_DIR / "listen_start.wav", dtype="int32"),
    "listen_end": sf.read(SOUND_DIR / "listen_end.wav", dtype="int32"),
    "connected": sf.read(SOUND_DIR / "connected.wav", dtype="int32"),
    "error": sf.read(SOUND_DIR / "failed.wav", dtype="int32"),
}

# Server
WS_SERVER_URL = f"ws://{BASE_URL}/stream"
RECONNECT_DELAY_SECONDS = 5  # Time to wait before trying to reconnect

# WakeWord
openwakeword.utils.download_models(model_names=[WAKEWORD_MODEL])
oww_model = Model(
    wakeword_models=[WAKEWORD_MODEL], inference_framework=INFERENCE_FRAMEWORK
)

# Buffer for AUdio
_MAX_MS = 160  # longest window we will grab at once
_MAX_FRAMES = 16_000 * _MAX_MS // 1000  # 16 kHz mono
_buffer = np.empty(_MAX_FRAMES, dtype=np.int16)  # ≈ 320 kB

# Preprocessing



AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "dtype": "int16",
    "chunk_ms": 10,
}
WS_CONFIG = {"ping_interval": 20, "ping_timeout": 20}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CLIENT")


def play_sound(sound_data: tuple[np.ndarray[Any, np.dtype[Any]]]):
    """A synchronous function that plays a sound file (this will block)."""
    try:
        sd.play(sound_data[0], sound_data[1], blocking=True)
    except Exception:
        logger.error("Could not play sound")


async def handle_server_messages(websocket: websockets.ClientConnection) -> None:
    """Handle incoming messages from the server and play sounds."""
    logger.info("Message handler task started.")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                event_type = data.get("event")

                # TODO: Use Structured Events and Match Case
                if event_type == "listen_started":
                    logger.info("Server started listening")
                    play_sound(AUDIO_CLIPS["listen_start"])
                elif event_type == "listen_stopped":
                    logger.info("Server stopped listening")
                    play_sound(AUDIO_CLIPS["listen_end"])
                elif event_type == "action_failed":
                    logger.info("Server Failed to complete action")
                    # TODO: Won't play because listen_stopped plays first
                    play_sound(AUDIO_CLIPS["error"])
                else:
                    logger.info(f"Server message: {data}")

            except json.JSONDecodeError:
                logger.info(f"Received non-JSON server response: {message}")
            except Exception as e:
                logger.error(f"Error processing server message: {e}")

    except ConnectionClosed:
        logger.info("Connection closed")
        raise
    except asyncio.CancelledError:
        logger.info("handle_server_message tasks stopped")
    except Exception:
        logger.exception("Message handler encountered a fatal error.")
        raise


async def stream_audio(
    websocket: websockets.ClientConnection, audio_queue: asyncio.Queue
):
    while True:
        try:
            frames = await audio_queue.get()
            await websocket.send(frames)
        except ConnectionClosed:
            logger.info("Connection closed")
            raise
        except asyncio.CancelledError:
            logger.info("stream audio tasks stopped")
        except Exception:
            logger.exception("Message handler encountered a fatal error.")
            raise


def clear_queue(q: asyncio.Queue):
    while True:
        try:
            q.get_nowait()
        except asyncio.queues.QueueEmpty:
            break



async def get_audio_frames(
    audio_queue: asyncio.Queue[bytes],
    chunk_ms: int,
    duration_ms: int,
    normalize: bool = False,  # kept for API compatibility
) -> np.ndarray:
    """Return *duration_ms* of int16 PCM as a SLICE view into a
    pre-allocated workspace.  **No allocations, no list building.**"""
    n_chunks = duration_ms // chunk_ms
    if n_chunks == 0:
        return _buffer[:0]  # empty view; zero alloc

    timeout = chunk_ms * 1.5 / 1000.0
    write = 0

    try:
        for _ in range(n_chunks):
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=timeout)

            view = memoryview(chunk).cast("h")  # bytes → int16  (no copy)
            ln = view.shape[0]
            _buffer[write : write + ln] = view  # memcpy once
            write += ln

    except TimeoutError:
        if write == 0:
            return _buffer[:0]

    return _buffer[:write]


async def stream_audio_from_queue(audio_queue: asyncio.Queue) -> None:
    """Pulls audio from a thread-safe queue and sends it over the WebSocket."""
    logger.info("Audio streaming task started.")
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                # Wakeword requires at least 80ms
                frames_view = await get_audio_frames(
                    audio_queue=audio_queue,
                    chunk_ms=AUDIO_CONFIG['chunk_ms'],
                    duration_ms=80,
                    normalize=True,
                )
                if frames_view.size > 0:
                    frames = frames_view.copy()
                    prediction = await loop.run_in_executor(
                        None, oww_model.predict, frames
                    )
                    if prediction[WAKEWORD_MODEL] > float(WAKE_WORD_THRESHOLD):
                        logger.info(
                            f"Wake word detected! (Score: {prediction[WAKEWORD_MODEL]:.2f})"
                        )
                        oww_model.reset()
                        async with websockets.connect(
                            WS_SERVER_URL, **WS_CONFIG
                        ) as websocket:
                            logger.info("✅ WebSocket connection established.")
                            logger.info("🎤 Microphone stream is now active.")

                            try:
                                async with asyncio.TaskGroup() as tg:
                                    tg.create_task(
                                        handle_server_messages(websocket),
                                        name="message_handler",
                                    )
                                    tg.create_task(
                                        stream_audio(websocket, audio_queue),
                                        name="mic_streamer",
                                    )
                            except* Exception as eg:
                                for exc in eg.exceptions:
                                    if not isinstance(
                                        exc, (ConnectionClosed, asyncio.CancelledError)
                                    ):
                                        logger.error(
                                            f"Unhandled exception in task group: {exc!r}",
                                            exc_info=exc,
                                        )
                            finally:
                                play_sound(AUDIO_CLIPS["listen_end"])
                                logger.info("No longer streaming audio")

            except ConnectionRefusedError:
                logger.warning("Server is offline")
                play_sound(AUDIO_CLIPS["error"])
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
            except Exception as e:
                logger.error(f"Connection failed: {e}.")
            finally:
                clear_queue(audio_queue)

    except asyncio.CancelledError:
        logger.info("Audio streaming task cancelled.")
    except Exception:
        logger.exception("Audio streaming task failed critically.")
        raise


async def run_client() -> None:
    """Main client function with a persistent reconnection loop."""
    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def audio_callback(indata, frames, time, status):
        """This is called from a separate thread for each audio block."""
        if status:
            logger.warning(status)
        try:
            loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))
        except asyncio.QueueFull:
            pass

    while True:
        try:
            with sd.InputStream(
                samplerate=AUDIO_CONFIG["sample_rate"],
                channels=AUDIO_CONFIG["channels"],
                dtype=AUDIO_CONFIG["dtype"],
                callback=audio_callback,
                device=INPUT_DEVICE_NAME,
                blocksize=int(
                    AUDIO_CONFIG["sample_rate"] * AUDIO_CONFIG["chunk_ms"] / 1000
                ),
            ):
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(
                        stream_audio_from_queue(audio_queue),
                        name="audio_streamer",
                    )

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
    logger.info("Starting WebSocket audio client...")
    try:
        print("\n" + "-" * 50)
        print("Available audio input devices:")
        print(sd.query_devices())
        print("-" * 50)
        if INPUT_DEVICE_NAME:
            logger.info(
                f"Attempting to use specified audio device: {INPUT_DEVICE_NAME}"
            )
        else:
            logger.info("Using default audio device.")
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(run_client())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception:
        logger.critical("A fatal error occurred in main.", exc_info=True)
        raise


if __name__ == "__main__":
    main()
