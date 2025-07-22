"""
This program acts as a home assistant satellite. It runs a local wakeword model. When
the wakeword is detected, it will forward audio to Home Assistant for Speech to text
and to handle intents.
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue
from multiprocessing.synchronize import Event
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


_output_stream = sd.OutputStream(
    samplerate=48000,
    channels=2,
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


def wakeword_process_worker(
    audio_q: mp.Queue,
    event_q: mp.Queue,
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
                chunk = audio_q.get(timeout=AUDIO_CONFIG["chunk_ms"] / 1000)
                if chunk is None:  # Sentinel value to stop the process
                    break
                # Wakeword Prediction
                prediction = oww_model.predict(np.frombuffer(chunk, dtype=np.int16))
                if prediction[model_name] > WAKE_WORD_THRESHOLD:
                    logger.info(
                        f"Wakeword detected! (Score: {prediction[model_name]:.2f})"
                    )
                    event_q.put(WAKEWORD_EVENT)
                    oww_model.reset()

        except queue.Empty:
            # This is expected when no audio is coming in, just continue
            continue
        except KeyboardInterrupt:
            break
        except Exception:
            logger.error("Error in Keyword detection process", exc_info=True)


def play_sound(sound_data: tuple[np.ndarray[Any, np.dtype[Any]]]):
    try:
        _output_stream.write(sound_data[0])
    except Exception:
        logger.error("Could not play sound")


async def handle_server_messages(
    websocket: websockets.ClientConnection, stt_active: asyncio.Event
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
                                    "end_stage": "intent",
                                    "input": {
                                        "sample_rate": AUDIO_CONFIG["sample_rate"]
                                    },
                                }
                            )
                        )
                    # Speech to text started
                    case {"event": {"type": "stt-start"}}:
                        play_sound(AUDIO_CLIPS["listen_start"])
                        stt_active.set()
                    # Speech to Text Complete
                    case {"event": {"type": "stt-end"}}:
                        try:
                            phrase = data["event"]["data"]["stt_output"]["text"]
                            logger.info(f"Command: {phrase}")
                        except KeyError:
                            logger.warning("STT complete, but no phrase transcribed")
                        stt_active.clear()
                    # Error
                    case {"event": {"type": "error"}}:
                        try:
                            err_msg = data["event"]["data"]["message"]
                            logger.warning(err_msg)
                        except KeyError:
                            logger.warning(
                                "Pipeline Error but could not retrieve reponse"
                            )
                        stt_active.clear()
                        play_sound(AUDIO_CLIPS["error"])

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
                        play_sound(AUDIO_CLIPS["task_complete"])
                        break

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
                        play_sound(AUDIO_CLIPS["error"])
                        break

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


async def stream_audio(
    websocket: websockets.ClientConnection,
    audio_q: mp.Queue,
    stt_active: asyncio.Event,
):
    loop = asyncio.get_running_loop()
    try:
        await stt_active.wait()
        logger.info("🎤 Microphone stream is now active.")
        while stt_active.is_set():
            chunk = await loop.run_in_executor(None, audio_q.get)
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
        logger.exception("Message handler encountered a fatal error.")
        raise
    finally:
        logger.info("🛑 Microphone stream is now stopped")


async def listen_for_wakeword(
    audio_queue: mp.Queue,
    event_queue: mp.Queue,
    shutdown_event: Event,
    wake_word_run_event: Event,
) -> None:
    """Waits for a wakeword detection event and then manages the server interaction."""
    loop = asyncio.get_running_loop()
    stt_active = asyncio.Event()
    wake_word_run_event.set()

    # Create Server URL
    server_url = f"wss://{WS_SERVER_URL}"
    if str(SECURE_WEBSOCKET.lower()) == "false":
        server_url = f"ws://{WS_SERVER_URL}"
    logger.info(f"Using HomeAssistant URl {server_url}")

    # Main Loop
    while not shutdown_event.is_set():
        event = await loop.run_in_executor(None, event_queue.get)
        if event is None:
            break
        if event == WAKEWORD_EVENT:
            wake_word_run_event.clear()
            try:
                # When wakeword is detected, connect to WebSocket and stream
                async with websockets.connect(server_url, **WS_CONFIG) as websocket:
                    try:
                        logger.info("✅ WebSocket connection established.")
                        async with asyncio.TaskGroup() as tg:
                            tg.create_task(
                                handle_server_messages(websocket, stt_active)
                            )
                            tg.create_task(
                                stream_audio(websocket, audio_queue, stt_active)
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
                play_sound(AUDIO_CLIPS["error"])
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
            except Exception:
                logger.exception("An error occurred during WebSocket communication.")
                play_sound(AUDIO_CLIPS["error"])
            finally:
                while not audio_queue.empty():
                    audio_queue.get_nowait()
                wake_word_run_event.set()


async def run_client(
    audio_queue: mp.Queue, event_queue: mp.Queue, shutdown_event, wake_word_run_event
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
                    audio_queue, event_queue, shutdown_event, wake_word_run_event
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
        audio_queue = mp.Queue()
        event_queue = mp.Queue()
        # Shutdown tasks and threads
        shutdown_event = mp.Event()
        # Signal whether we are streaming
        wake_word_run_event = mp.Event()

        # Start Dedicated Wakeword Process
        wakeword_process = mp.Process(
            target=wakeword_process_worker,
            args=(
                audio_queue,
                event_queue,
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
            run_client(audio_queue, event_queue, shutdown_event, wake_word_run_event)
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
