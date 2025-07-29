"""
This program acts as a home assistant satellite. It runs a local wakeword model. When
the wakeword is detected, it will forward audio to Home Assistant for Speech to text
and to handle intents.
"""

import asyncio
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory

# Background worker for WakeWord Detection
import os
import queue
import signal
import time
import wave
from asyncio import TaskGroup
from contextlib import AsyncExitStack
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event

import numpy as np
import uvloop

from easy_ha_satellite.audio_io import (
    Alert,
    AudioPlayback,
    InputAudioConfig,
    OutputAudioConfig,
    play_alert,
)
from easy_ha_satellite.config import get_root_logger
from easy_ha_satellite.home_assistant import (
    HASSHttpClient,
    HASSocketClient,
    HomeAssistantConfig,
    Pipeline,
    PipelineEventType,
)
from easy_ha_satellite.wake_word_consumer import WakeEvent, WakeEventType

logger = get_root_logger()


async def run_pipeline(
    mic_audio: np.ndarray,
    write_index: Synchronized,
    mic_cfg: InputAudioConfig,
    ws_client: HASSocketClient,
    http_client: HASSHttpClient,
    speaker: AudioPlayback,
) -> None:
    pipe = Pipeline(client=ws_client, sample_rate=mic_cfg.sample_rate)
    ws_client.on_event = pipe.handle_event

    need_audio = asyncio.Event()
    stop_all = asyncio.Event()
    need_audio.set()

    async def audio_pump():
        # A list to hold all audio chunks for the current pumping session
        recorded_chunks = []
        try:
            while not stop_all.is_set():
                # 1. Wait for the signal to start pumping audio
                await need_audio.wait()

                # Sync up to the most recent audio data in the buffer
                local_read_index = write_index.value - 1

                # Continuously pump audio as long as the signal is active
                while need_audio.is_set() and not stop_all.is_set():
                    # Wait for the producer to write the next chunk
                    while local_read_index >= write_index.value:
                        await asyncio.sleep(0.01)
                    chunk: np.ndarray = mic_audio[local_read_index % mic_cfg.buffer_slots]
                    recorded_chunks.append(chunk)
                    await pipe.send_audio(chunk.tobytes())

                    # Move to the next chunk
                    local_read_index += 1

        except asyncio.CancelledError:
            pass
        finally:
            if not recorded_chunks:
                logger.warning("No audio was pumped, nothing to save.")
        if os.getenv("RECORD_INPUT"):
            logger.info("Writing recorded audio to file...")
            try:
                # 1. Combine all the small chunks into one large NumPy array
                final_audio_array = np.concatenate(recorded_chunks)

                # 2. Generate a unique filename using a timestamp
                timestamp = int(time.time())
                filename = f"audio_pump_dump_{timestamp}.wav"

                # 3. Use the wave module to write a standard WAV file
                with wave.open(filename, "wb") as wav_file:
                    wav_file.setnchannels(mic_cfg.channels)
                    wav_file.setsampwidth(np.dtype(mic_cfg.dtype).itemsize)
                    wav_file.setframerate(mic_cfg.sample_rate)
                    wav_file.writeframes(final_audio_array.tobytes())

                logger.info(f"Successfully saved recorded audio to {filename}")

            except Exception as e:
                logger.exception(f"Failed to save audio file: {e}")

    async def event_pump():
        async for evt in pipe:
            match evt.type:
                case PipelineEventType.STT_START:
                    need_audio.set()
                    logger.info("🎤 Listening")
                    await play_alert(Alert.LISTEN_START, speaker)
                case PipelineEventType.STT_END:
                    need_audio.clear()
                    stop_all.set()
                    logger.info("🛑 No longer listening")
                    await play_alert(Alert.LISTEN_COMPLETE, speaker)
                case PipelineEventType.ERROR:
                    need_audio.clear()
                    stop_all.set()
                    await play_alert(Alert.ERROR, speaker)
                case _:
                    pass

    try:
        async with TaskGroup() as tg:
            tg.create_task(audio_pump())
            tg.create_task(event_pump())
            tg.create_task(pipe.start())
    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        stop_all.set()
    if pipe.media_url:
        data = await http_client.download_media(pipe.media_url)
        await speaker.play(data)
    logger.info("Pipeline Complete")


async def main(
    in_audio_cfg: InputAudioConfig,
    out_audio_cfg: OutputAudioConfig,
    ha_cfg: HomeAssistantConfig,
    stop_event: Event,
    events_q: mp.Queue,
    audio_buffer: np.ndarray,
    write_index: Synchronized,
) -> None:
    try:
        async with AsyncExitStack() as stack:
            speaker = await stack.enter_async_context(
                AudioPlayback(out_audio_cfg, os.getenv("OUTPUT_AUDIO_DEVICE"))
            )
            ha_http_client = await stack.enter_async_context(
                HASSHttpClient(ha_cfg, os.environ["HA_TOKEN"])
            )
            ha_ws_client = await stack.enter_async_context(
                HASSocketClient(ha_cfg, os.environ["HA_TOKEN"])
            )
            await play_alert(Alert.CONNECTED, speaker)
            while not stop_event.is_set():
                # Wait for detector to signal wake
                try:
                    event: WakeEvent = await asyncio.to_thread(events_q.get)
                except queue.Empty:
                    continue

                if event.type == WakeEventType.DETECTED:
                    logger.info("Keyword Detected")
                    asyncio.create_task(
                        run_pipeline(
                            mic_audio=audio_buffer,
                            write_index=write_index,
                            mic_cfg=in_audio_cfg,
                            ws_client=ha_ws_client,
                            http_client=ha_http_client,
                            speaker=speaker,
                        )
                    )

    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        pass


def test_record():
    pass


def voice_pipeline_consumer(
    stop_event: Event,
    events_q: mp.Queue,
    shm_name: str,
    write_index: Synchronized,
    mic_cfg: InputAudioConfig,
    speaker_cfg: OutputAudioConfig,
    ha_cfg: HomeAssistantConfig,
):
    try:
        # Connect to shared memory
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        samples_per_chunk = mic_cfg.chunk_samples * mic_cfg.channels
        buffer = np.ndarray(
            (mic_cfg.buffer_slots, samples_per_chunk),
            dtype=mic_cfg.dtype,
            buffer=existing_shm.buf,
        )
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.info("Voice Pipeline process starting.")
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(
                main(
                    in_audio_cfg=mic_cfg,
                    out_audio_cfg=speaker_cfg,
                    ha_cfg=ha_cfg,
                    stop_event=stop_event,
                    audio_buffer=buffer,
                    write_index=write_index,
                    events_q=events_q,
                )
            )
    except Exception:
        logger.exception("An unrecoverable error occurred in the microphone producer.")
    finally:
        if "existing_shm" in locals():
            existing_shm.close()
        logger.info("Wake word consumer stopped.")
