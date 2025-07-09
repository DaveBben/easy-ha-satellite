import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
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

pool = ThreadPoolExecutor(max_workers=1)

# Envrionment Variables
BASE_URL = os.environ.get("BASE_URL", "192.168.1.16:9292")
INPUT_DEVICE_NAME = os.environ.get("INPUT_AUDIO_DEVICE", None)
WAKEWORD_MODEL = os.environ.get("WAKEWORD_MODEL_NAME", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(os.environ.get("WW_THRESHOLD", "0.5"))
INFERENCE_FRAMEWORK = os.environ.get("INFERENCE_FRAMEWORK", "onnx")
INFERENCE_THREADS = os.environ.get("INFERENCE_THREADS", os.cpu_count())

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
    "listen_start": sf.read(SOUND_DIR / "listen_start.wav", dtype="int32"),
    "listen_end": sf.read(SOUND_DIR / "listen_end.wav", dtype="int32"),
    "connected": sf.read(SOUND_DIR / "connected.wav", dtype="int32"),
    "error": sf.read(SOUND_DIR / "failed.wav", dtype="int32"),
}

# Server
WS_SERVER_URL = f"ws://{BASE_URL}/stream"
RECONNECT_DELAY_SECONDS = 5  # Time to wait before trying to reconnect\
WS_CONFIG = {"ping_interval": 20, "ping_timeout": 20}

# WakeWord
REQUIRED_WAKEWORD_MS = 80
COOLDOWN_MS = 3000


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CLIENT")


def wakeword_process_worker(
    audio_q: mp.Queue,
    event_q: mp.Queue,
    model_name: str,
    inference_framework: str,
    shutdown_event: Any,
    streaming_event: Any,
):
    logging.info("Wakeword process started.")
    try:
        oww_model = Model(
            wakeword_models=[model_name], inference_framework=inference_framework
        )
    except Exception:
        logging.critical("Failed to load model in subprocess.", exc_info=True)
        return
    while not shutdown_event.is_set():
        try:
            if not streaming_event.is_set():
                chunk = audio_q.get(timeout=AUDIO_CONFIG["chunk_ms"] / 1000)
                if chunk is None:  # Sentinel value to stop the process
                    break
                # Wakeword Prediction
                prediction = oww_model.predict(np.frombuffer(chunk, dtype=np.int16))
                if prediction[model_name] > WAKE_WORD_THRESHOLD:
                    logger.info(
                        f"Wakeword detected! (Score: {prediction[model_name]:.2f})"
                    )
                    event_q.put("WAKEWORD_DETECTED")
                    oww_model.reset()
            else:
                time.sleep(1)

        except queue.Empty:
            # This is expected when no audio is coming in, just continue
            continue
        except KeyboardInterrupt:
            break
        except Exception:
            logger.error("Error in wakeword process", exc_info=True)


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


async def stream_audio(websocket: websockets.ClientConnection, audio_q: mp.Queue):
    loop = asyncio.get_running_loop()
    try:
        while True:
            chunk = await loop.run_in_executor(None, audio_q.get)
            if chunk is None:
                break
            await websocket.send(chunk)
    except ConnectionClosed:
        logger.info("Stopped streaming audio to server.")
    except Exception:
        logger.exception("Message handler encountered a fatal error.")
        raise


async def listen_for_wakeword(
    audio_queue: mp.Queue,
    event_queue: mp.Queue,
    shutdown_event: Any,
    streaming_event: Any,
) -> None:
    """Waits for a wakeword detection event and then manages the server interaction."""
    loop = asyncio.get_running_loop()
    while not shutdown_event.is_set():
        event = await loop.run_in_executor(None, event_queue.get)
        if event is None:
            break

        if event == "WAKEWORD_DETECTED":
            streaming_event.set()
            try:
                # When wakeword is detected, connect to WebSocket and stream
                async with websockets.connect(WS_SERVER_URL, **WS_CONFIG) as websocket:
                    logger.info("✅ WebSocket connection established.")
                    logger.info("🎤 Microphone stream is now active.")

                    try:
                        async with asyncio.TaskGroup() as tg:
                            tg.create_task(handle_server_messages(websocket))
                            tg.create_task(stream_audio(websocket, audio_queue))
                    except* Exception as eg:
                        for exc in eg.exceptions:
                            if not isinstance(
                                exc, (ConnectionClosed, asyncio.CancelledError)
                            ):
                                logger.error(
                                    f"Unhandled exception in task group: {exc!r}"
                                )
                    finally:
                        play_sound(AUDIO_CLIPS["listen_end"])

            except ConnectionRefusedError:
                logger.warning("Server is offline. Could not connect.")
                play_sound(AUDIO_CLIPS["error"])
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
            except Exception:
                logger.exception("An error occurred during WebSocket communication.")
            finally:
                streaming_event.clear()


async def run_client(
    audio_queue: mp.Queue, event_queue: mp.Queue, shutdown_event, streaming_event
) -> None:
    """Main client function with a persistent reconnection loop."""

    def audio_callback(indata, frames, time, status):
        """This is called from a separate thread for each audio block."""
        if status:
            logger.warning(status)
        try:
            audio_queue.put_nowait(bytes(indata))
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
                await listen_for_wakeword(
                    audio_queue, event_queue, shutdown_event, streaming_event
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

        # Download Wakeword Model
        openwakeword.utils.download_models(model_names=[WAKEWORD_MODEL])

        # Create Mulitprocessing Queue
        audio_queue = mp.Queue()
        event_queue = mp.Queue()
        # Shutdown tasks and threads
        shutdown_event = mp.Event()
        # Signal whether we are streaming
        streaming_event = mp.Event()

        # Start Dedicated Wakeword Process
        wakeword_process = mp.Process(
            target=wakeword_process_worker,
            args=(
                audio_queue,
                event_queue,
                WAKEWORD_MODEL,
                INFERENCE_FRAMEWORK,
                shutdown_event,
                streaming_event,
            ),
            daemon=True,  # auto-terminate with the main process
        )
        wakeword_process.start()

        loop = uvloop.new_event_loop()
        asyncio.set_event_loop(loop)
        main_task = loop.create_task(
            run_client(audio_queue, event_queue, shutdown_event, streaming_event)
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
