import asyncio
import json
import logging
import os
from pathlib import Path
import numpy as np

import openwakeword
import sounddevice as sd
import websockets
from openwakeword.model import Model
from pydub import AudioSegment
from pydub.playback import play
from websockets.exceptions import ConnectionClosed, InvalidStatus

# Envrionment Variables
BASE_URL = os.environ.get("BASE_URL", "192.168.1.16:9292")
DEVICE_ID = os.environ.get("AUDIO_DEVICE", None)
WAKEWORD_MODEL = os.environ.get("WAKEWORD_MODEL_NAME", "hey_jarvis")
WAKE_WORD_THRESHOLD = os.environ.get("WW_THRESHOLD", "0.5")
INFERENCE_FRAMEWORK = os.environ.get("INFERENCE_FRAMEWORK", "onnx")

# Sound files
PROJECT_ROOT = Path(__file__).resolve().parent
SOUND_DIR = PROJECT_ROOT / "sounds"
LISTEN_START = SOUND_DIR / "listen_start.mp3"
LISTEN_END = SOUND_DIR / "listen_end.mp3"
CONNECTED = SOUND_DIR / "connected.mp3"
ERROR = SOUND_DIR / "failed.mp3"

# Server
WS_SERVER_URL = f"ws://{BASE_URL}/stream"
RECONNECT_DELAY_SECONDS = 5  # Time to wait before trying to reconnect

# State
IS_LISTENING = False

# WakeWord
openwakeword.utils.download_models(model_names=[WAKEWORD_MODEL])
oww_model = Model(wakeword_models=[WAKEWORD_MODEL], inference_framework=INFERENCE_FRAMEWORK)

AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "dtype": "int16",
    "chunk_ms": 20,
}
WS_CONFIG = {
    "ping_interval": 20,
    "ping_timeout": 20,
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CLIENT")


# --- Sound Playback (wrapped for async) ---
def play_sound_blocking(sound_path: str):
    """A synchronous function that plays a sound file (this will block)."""
    try:
        # 1. Load the audio file using pydub
        sound = AudioSegment.from_file(sound_path)

        # 2. Play the sound. pydub will automatically use simpleaudio.
        play(sound)

    except Exception as e:
        logger.error(f"Could not play sound {sound_path}: {e}")


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
                    # Run the blocking playsound() in a separate thread
                    await asyncio.to_thread(play_sound_blocking, LISTEN_START)
                elif event_type == "listen_stopped":
                    logger.info("Server stopped listening")
                    await asyncio.to_thread(play_sound_blocking, LISTEN_END)
                elif event_type == "action_failed":
                    logger.info("Server Failed to complete action")
                    await asyncio.to_thread(play_sound_blocking, ERROR)
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
            frames = await get_audio_frames(audio_queue=audio_queue, chunk_ms=20, duration_ms=20, normalize=True)
            await websocket.send(frames)
        except ConnectionClosed:
            logger.info("Connection closed")
            raise
        except asyncio.CancelledError:
            logger.info("stream audio tasks stopped")
        except Exception:
            logger.exception("Message handler encountered a fatal error.")
            raise

def normalize_audio(audio_data: list) -> np.ndarray:
    # Normalize to prevent clipping
    audio_array = np.concatenate(audio_data).astype(np.int16)
    # Convert to float32 and normalize to [-1.0, 1.0] range
    audio_float = audio_array.astype(np.float32) / 32768.0

    # Apply gain reduction if needed to prevent clipping
    max_val = np.max(np.abs(audio_float))
    if max_val > 0.7:  # If signal is too loud
        audio_float = audio_float * (0.7 / max_val)

    # Convert back to 16-bit integer array
    audio_int16 = (audio_float * 32768.0).astype(np.int16)
    return audio_int16

async def get_audio_frames(
    audio_queue: asyncio.Queue, chunk_ms: int,  duration_ms: int, normalize: bool = True
) -> np.ndarray:
    num_chunks = duration_ms // chunk_ms
    if num_chunks == 0:
        return np.array([], dtype=np.int16)

    frames = []
    try:
        timeout_seconds = (chunk_ms * 1.5) / 1000.0
        for _ in range(num_chunks):
            chunk = await asyncio.wait_for(
                audio_queue.get(), timeout=timeout_seconds
            )
            frames.append(np.frombuffer(chunk, dtype=np.int16))
        if normalize:
            return normalize_audio(frames)

    except TimeoutError:
        logger.debug("Did not receive expected audio chunk from client in time.")
        if not frames:
            return np.array([], dtype=np.int16)
        return np.concatenate(frames)


async def stream_audio_from_queue(audio_queue: asyncio.Queue) -> None:
    """Pulls audio from a thread-safe queue and sends it over the WebSocket."""
    global IS_LISTENING
    logger.info("Audio streaming task started.")
    try:
        while True:
            try:
                if not IS_LISTENING:
                    # Wakeword requires at least 80ms
                    frames = await get_audio_frames(audio_queue=audio_queue, chunk_ms=20, duration_ms=80, normalize=True)
                    if frames.size > 0:
                        prediction = oww_model.predict(frames)
                        if prediction[WAKEWORD_MODEL] > float(WAKE_WORD_THRESHOLD):
                            print(
                                f"Wake word detected! (Score: {prediction[WAKEWORD_MODEL]:.2f})"
                            )
                            IS_LISTENING = True
                else:
                    async with websockets.connect(
                        WS_SERVER_URL, **WS_CONFIG
                    ) as websocket:
                        logger.info("✅ WebSocket connection established.")
                        logger.info("🎤 Microphone stream is now active.")
                        await asyncio.to_thread(play_sound_blocking, CONNECTED)
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
                            IS_LISTENING = False
                            logger.info("No longer streaming audio")

            except Exception as e:
                logger.error(f"Connection failed: {e}.")
                # await asyncio.to_thread(
                #     play_sound_blocking, ERROR
                # )  # No this will be annoying
                IS_LISTENING = False
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)

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
                device=DEVICE_ID,
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
        if DEVICE_ID:
            logger.info(f"Attempting to use specified audio device: {DEVICE_ID}")
        else:
            logger.info("Using default audio device.")
        asyncio.run(run_client())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception:
        logger.critical("A fatal error occurred in main.", exc_info=True)
        raise


if __name__ == "__main__":
    main()
