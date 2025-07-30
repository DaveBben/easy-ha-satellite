"""
This program acts as a home assistant satellite. It runs a local wakeword model. When
the wakeword is detected, it will forward audio to Home Assistant for Speech to text
and to handle intents.
"""

import asyncio
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory
import os
import signal
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
    preload_alerts,
)
from easy_ha_satellite.config import AppConfig, get_root_logger
from easy_ha_satellite.home_assistant import (
    HASSHttpClient,
    HASSocketClient,
    HomeAssistantConfig,
    Pipeline,
    PipelineEventType,
)

logger = get_root_logger()


async def _save_captured_audio(captured_chunks: list[bytes], mic_cfg: InputAudioConfig) -> None:
    """
    Save captured pipeline audio to a WAV file for analysis.
    This runs in a thread to avoid blocking the main pipeline.
    """
    import datetime
    import wave

    def save_audio():
        try:
            # Generate timestamp-based filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_audio_webrtc_{timestamp}.wav"

            # Combine all chunks
            audio_data = b"".join(captured_chunks)

            # Save to WAV file
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(mic_cfg.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(mic_cfg.sample_rate)
                wf.writeframes(audio_data)

            # Calculate statistics
            audio_array = np.frombuffer(audio_data, dtype=mic_cfg.dtype)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            peak = np.max(np.abs(audio_array))
            duration_ms = len(captured_chunks) * mic_cfg.chunk_ms

            logger.info(f"Saved pipeline audio: {filename}")
            logger.info(
                f"Duration: {duration_ms}ms, RMS: {20 * np.log10(rms / 32768):.1f}dBFS, "
                f"Peak: {20 * np.log10(peak / 32768):.1f}dBFS"
            )

        except Exception as e:
            logger.error(f"Failed to save captured audio: {e}")

    # Save in thread to avoid blocking pipeline
    await asyncio.to_thread(save_audio)


async def _save_captured_audio(captured_chunks: list[bytes], mic_cfg: InputAudioConfig) -> None:
    """
    Save captured pipeline audio to a WAV file for analysis.
    This runs in a thread to avoid blocking the main pipeline.
    """
    import datetime
    import wave

    def save_audio():
        try:
            # Generate timestamp-based filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_audio_webrtc_{timestamp}.wav"

            # Combine all chunks
            audio_data = b"".join(captured_chunks)

            # Save to WAV file
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(mic_cfg.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(mic_cfg.sample_rate)
                wf.writeframes(audio_data)

            # Calculate statistics
            audio_array = np.frombuffer(audio_data, dtype=mic_cfg.dtype)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            peak = np.max(np.abs(audio_array))
            duration_ms = len(captured_chunks) * mic_cfg.chunk_ms

            logger.info(f"Saved pipeline audio: {filename}")
            logger.info(
                f"Duration: {duration_ms}ms, RMS: {20 * np.log10(rms / 32768):.1f}dBFS, "
                f"Peak: {20 * np.log10(peak / 32768):.1f}dBFS"
            )

        except Exception as e:
            logger.error(f"Failed to save captured audio: {e}")

    # Save in thread to avoid blocking pipeline
    await asyncio.to_thread(save_audio)


async def run_pipeline(
    mic_audio: np.ndarray,
    write_index: Synchronized,
    mic_cfg: InputAudioConfig,
    ws_client: HASSocketClient,
    http_client: HASSHttpClient,
    speaker: AudioPlayback,
    app_cfg: AppConfig,
) -> None:
    end_stage = "tts"
    logger.debug("Inside of run pipeline")
    if not app_cfg.enable_tts:
        logger.debug("TTS is disabled")
        end_stage = "intent"
    pipe = Pipeline(client=ws_client, sample_rate=mic_cfg.sample_rate, end_stage=end_stage)
    ws_client.on_event = pipe.handle_event

    need_audio = asyncio.Event()
    need_audio.set()

    # Audio capture for testing/debugging
    capture_audio = os.getenv("CAPTURE_PIPELINE_AUDIO", "false").lower() == "true"
    captured_chunks = [] if capture_audio else None

    async def audio_pump():
        try:
            logger.debug("audio_pump starting...")
            # Wait for the signal to start pumping audio
            await need_audio.wait()

            # Sync up to the most recent audio data in the buffer
            local_read_index = write_index.value - 1

            # Continuously pump audio as long as the signal is active
            while need_audio.is_set():
                # Wait for the producer to write the next chunk
                while local_read_index >= write_index.value:
                    await asyncio.sleep(0)  # Yield to other tasks without delay
                chunk: np.ndarray = mic_audio[local_read_index % mic_cfg.buffer_slots]
                chunk_bytes = chunk.tobytes()

                # Capture audio for debugging if enabled
                if capture_audio:
                    captured_chunks.append(chunk_bytes)

                try:
                    await pipe.send_audio(chunk_bytes)
                except Exception as e:
                    logger.error(f"Failed to send audio chunk: {e}")
                    raise

                # Move to the next chunk
                local_read_index += 1
            logger.debug("Send audio pump cleanly stopped")

        except asyncio.CancelledError:
            logger.debug("audio_pump cancelled")
            pass
        except Exception as e:
            logger.error(f"audio_pump failed: {e}")
            raise

    async def event_pump():
        try:
            logger.debug("event_pump starting...")
            async for evt in pipe:
                match evt.type:
                    case PipelineEventType.STT_START:
                        need_audio.set()
                        logger.info("🎤 Listening")
                        logger.debug("About to play listen start alert...")
                        play_alert(Alert.LISTEN_START, speaker)
                    case PipelineEventType.STT_END:
                        need_audio.clear()
                        logger.info("🛑 No longer listening")
                        logger.debug("About to play listen complete alert...")
                        play_alert(Alert.LISTEN_COMPLETE, speaker),
                    case PipelineEventType.ERROR:
                        need_audio.clear()
                        play_alert(Alert.ERROR, speaker)
                    case _:
                        pass
            logger.debug("Event pump cleanll cancelled")
        except asyncio.CancelledError:
            logger.debug("event_pump cancelled")
            pass
        except Exception as e:
            logger.error(f"event_pump failed: {type(e).__name__}: {e}")
            raise

    try:
        logger.debug("Starting TaskGroup with 3 tasks...")

        async def pipe_start_wrapper():
            try:
                logger.debug("pipe.start() starting...")
                await pipe.start()
                logger.debug("pipe.start() completed successfully")
            except Exception as e:
                logger.error(f"pipe.start() failed: {type(e).__name__}: {e}")
                raise

        async with TaskGroup() as tg:
            logger.debug("Creating audio_pump task...")
            tg.create_task(audio_pump())
            logger.debug("Creating event_pump task...")
            tg.create_task(event_pump())
            logger.debug("Creating pipe.start task...")
            tg.create_task(pipe_start_wrapper())
            logger.debug("All tasks created, waiting for TaskGroup...")
        logger.debug("TaskGroup completed successfully")
    except KeyboardInterrupt:
        logger.debug("TaskGroup interrupted by KeyboardInterrupt")
        pass
    except asyncio.CancelledError:
        logger.debug("TaskGroup cancelled")
        pass
    except Exception as e:
        # Handle ExceptionGroup from TaskGroup
        if hasattr(e, "exceptions"):
            logger.error(f"TaskGroup failed with {len(e.exceptions)} sub-exceptions:")
            for i, exc in enumerate(e.exceptions):
                logger.error(f"  Sub-exception {i + 1}: {type(exc).__name__}: {exc}")
        else:
            logger.error(f"TaskGroup failed with single exception: {type(e).__name__}: {e}")

    # Save captured audio if enabled
    if capture_audio and captured_chunks:
        await _save_captured_audio(captured_chunks, mic_cfg)

    if pipe.media_url:
        data = await http_client.download_media(pipe.media_url)
        await speaker.play(data)
    logger.info("Pipeline Complete")


async def main(
    in_audio_cfg: InputAudioConfig,
    out_audio_cfg: OutputAudioConfig,
    ha_cfg: HomeAssistantConfig,
    stop_event: Event,
    wake_counter: Synchronized,
    wake_model_name: mp.Array,
    audio_buffer: np.ndarray,
    write_index: Synchronized,
    app_cfg: AppConfig,
) -> None:
    logger.debug("Voice pipeline main() starting...")
    try:
        last_wake_count = 0
        poll_count = 0
        speaker, ha_http_client, ha_ws_client = None, None, None
        logger.debug("Etnering AsyncExitStack")
        async with AsyncExitStack() as stack:
            logger.debug("Creating HASS HTTP client...")
            ha_http_client = await stack.enter_async_context(
                HASSHttpClient(ha_cfg, os.environ["HA_TOKEN"])
            )
            logger.debug("Creating HASS WebSocket client...")
            ha_ws_client = await stack.enter_async_context(
                HASSocketClient(ha_cfg, os.environ["HA_TOKEN"])
            )
            speaker = await stack.enter_async_context(
                AudioPlayback(out_audio_cfg, os.getenv("OUTPUT_AUDIO_DEVICE"))
            )

            logger.debug("Playing connected alert...")
            play_alert(Alert.CONNECTED, speaker)
            logger.debug("Starting polling loop")
            while not stop_event.is_set():
                # Check for wake word detection using shared memory counter
                poll_count += 1
                if poll_count % 1000 == 0:  # Log every 10 seconds (1000 * 0.01s)
                    logger.debug(
                        f"Voice pipeline polling (count={poll_count}, last_wake={last_wake_count})"
                    )

                current_wake_count = wake_counter.value
                if current_wake_count > last_wake_count:
                    # New wake word detected!
                    logger.info("Wake word detected!")

                    # Get the model name from shared array
                    with wake_counter.get_lock():
                        model_name_bytes = bytes(wake_model_name[:])
                        model_name = model_name_bytes.rstrip(b"\x00").decode("utf-8")

                    logger.info(f"Starting pipeline for model: {model_name}")
                    last_wake_count = current_wake_count

                    # Check WebSocket connection status before using it
                    ws_connected = ha_ws_client._ws is not None
                    logger.debug(f"WebSocket connected: {ws_connected}")

                    # If WebSocket is disconnected, try to reconnect
                    if not ws_connected:
                        logger.warning("WebSocket disconnected, attempting to reconnect...")
                        try:
                            await ha_ws_client.start()  # This should reconnect
                            logger.debug("WebSocket reconnected successfully")
                        except Exception as e:
                            logger.error(f"Failed to reconnect WebSocket: {e}")
                            continue  # Skip this wake word and try again later

                    # Check HTTP client session status
                    http_session_active = (
                        hasattr(ha_http_client, "_session") and ha_http_client._session is not None
                    )
                    logger.debug(f"HTTP client session active: {http_session_active}")

                    # If HTTP session is not active, try to restart it
                    if not http_session_active:
                        logger.warning("HTTP client session not active, attempting to restart...")
                        try:
                            await ha_http_client.start()  # This should start the session
                            logger.debug("HTTP client session restarted successfully")
                        except Exception as e:
                            logger.error(f"Failed to restart HTTP client session: {e}")
                            continue  # Skip this wake word and try again later

                    # Add timeout to pipeline execution to prevent hanging
                    pipeline_task = asyncio.create_task(
                        asyncio.wait_for(
                            run_pipeline(
                                mic_audio=audio_buffer,
                                write_index=write_index,
                                mic_cfg=in_audio_cfg,
                                ws_client=ha_ws_client,
                                http_client=ha_http_client,
                                speaker=speaker,
                                app_cfg=app_cfg,
                            ),
                            timeout=30.0,  # 30 second timeout
                        )
                    )

                    # Handle pipeline timeout
                    def handle_pipeline_result(task):
                        try:
                            task.result()
                        except TimeoutError:
                            logger.error("Pipeline execution timed out after 30 seconds")
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            logger.error(f"Pipeline execution failed: {e}")

                    pipeline_task.add_done_callback(handle_pipeline_result)

                await asyncio.sleep(0.01)  # Small delay to avoid busy waiting

    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        pass


def voice_pipeline_consumer(
    stop_event: Event,
    wake_counter: Synchronized,
    wake_model_name: mp.Array,
    shm_name: str,
    write_index: Synchronized,
    mic_cfg: InputAudioConfig,
    speaker_cfg: OutputAudioConfig,
    ha_cfg: HomeAssistantConfig,
    app_cfg: AppConfig,
):
    try:
        logger.info(f"[{os.getpid()}] Voice Pipeline process starting.")

        logger.debug("Connecting to shared memory...")
        # Connect to shared memory
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        samples_per_chunk = mic_cfg.chunk_samples * mic_cfg.channels
        logger.debug("Creating buffer from shared memory...")
        buffer = np.ndarray(
            (mic_cfg.buffer_slots, samples_per_chunk),
            dtype=mic_cfg.dtype,
            buffer=existing_shm.buf,
        )
        logger.debug("Setting up signal handling...")
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        logger.debug("Pre-loading alerts...")
        preload_alerts(speaker_cfg)
        logger.debug("Alerts preloaded")

        logger.debug("Starting asyncio runner with uvloop...")
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(
                main(
                    in_audio_cfg=mic_cfg,
                    out_audio_cfg=speaker_cfg,
                    ha_cfg=ha_cfg,
                    stop_event=stop_event,
                    wake_counter=wake_counter,
                    wake_model_name=wake_model_name,
                    audio_buffer=buffer,
                    write_index=write_index,
                    app_cfg=app_cfg,
                )
            )
    except Exception:
        logger.exception("An unrecoverable error occurred in the microphone producer.")
    finally:
        if "existing_shm" in locals():
            existing_shm.close()
        logger.info(f"[{os.getpid()}] Voice Pipeline process shutting down.")
