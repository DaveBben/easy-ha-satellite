"""
This program acts as a home assistant satellite. It runs a local wakeword model. When
the wakeword is detected, it will forward audio to Home Assistant for Speech to text
and to handle intents.
"""

import asyncio
import os
import queue
import sys
from asyncio import TaskGroup
from contextlib import AsyncExitStack
from multiprocessing import get_context
from multiprocessing.synchronize import Semaphore

import uvloop

from easy_ha_satellite.audio_io import (
    Alert,
    AsyncCaptureSession,
    AudioCapture,
    AudioPlayback,
    OnDeviceAlerts,
    load_audio_capture_config,
    load_audio_playback_config,
)
from easy_ha_satellite.config import ConfigError, get_root_logger
from easy_ha_satellite.home_assistant import (
    HASSHttpClient,
    HASSocketClient,
    HomeAssistantConfig,
    Pipeline,
    PipelineEventType,
)
from easy_ha_satellite.wake_word import load_wake_word_config
from easy_ha_satellite.wake_word_worker import WakeEvent, WakeEventType, detector_process

logger = get_root_logger()

# Required Environment Variables
required_env = ["HA_TOKEN", "HA_HOST", "HA_PORT", "HA_SSL"]
for env in required_env:
    try:
        os.environ[env]
    except KeyError:
        logger.error("Missing Environment variable %s", env)
        sys.exit(1)


async def run_pipeline(mic: AudioCapture, speaker: AudioPlayback, pipe: Pipeline) -> str:
    alerts = OnDeviceAlerts(speaker)
    need_audio = asyncio.Event()
    stop_all = asyncio.Event()
    need_audio.set()

    async def audio_pump():
        try:
            while not stop_all.is_set():
                await need_audio.wait()
                chunk = await asyncio.to_thread(mic.get_chunk, 0.1)
                if chunk:
                    await pipe.send_audio(chunk)
            await asyncio.to_thread(mic.stop)
        except asyncio.CancelledError:
            pass

    async def event_pump():
        async for evt in pipe:
            match evt.type:
                case PipelineEventType.STT_START:
                    need_audio.set()
                    logger.info("🎤 Listening")
                    await alerts.play(Alert.LISTEN_START)
                case PipelineEventType.STT_END:
                    need_audio.clear()
                    stop_all.set()
                    logger.info("🛑 No longer listening")
                    await alerts.play(Alert.LISTEN_COMPLETE)
                case PipelineEventType.ERROR:
                    need_audio.clear()
                    stop_all.set()
                    await alerts.play(Alert.ERROR)
                case _:
                    pass

    try:
        async with TaskGroup() as tg:
            tg.create_task(audio_pump())
            tg.create_task(event_pump())
            tg.create_task(pipe.start())
    except asyncio.CancelledError:
        pass
    finally:
        stop_all.set()

    logger.info("Pipeline Complete")
    return pipe.media_url


async def main() -> None:
    try:
        try:
            in_audio_cfg = load_audio_capture_config()
            out_audio_cfg = load_audio_playback_config()
            wake_cfg = load_wake_word_config()
            hass_cfg = HomeAssistantConfig(
                host=os.environ["HA_HOST"], port=os.environ["HA_PORT"], ssl=os.environ["HA_SSL"]
            )
        except ConfigError as e:
            logger.error("%s", e)
            sys.exit(1)

        ctx = get_context("spawn")
        mic_sem: Semaphore = ctx.Semaphore(1)
        resume = ctx.Event()
        shutdown = ctx.Event()
        bootstrap = ctx.Event()
        wake_events: queue.Queue[WakeEvent] = ctx.Queue()

        # Spawn detector
        det_proc = ctx.Process(
            target=detector_process,
            args=(
                in_audio_cfg,
                wake_cfg,
                os.getenv("INPUT_AUDIO_DEVICE"),
                mic_sem,
                shutdown,
                resume,
                wake_events,
                bootstrap,
            ),
            daemon=True,
        )
        det_proc.start()

        async with AsyncExitStack() as stack:
            speaker = await stack.enter_async_context(
                AudioPlayback(out_audio_cfg, os.getenv("OUTPUT_AUDIO_DEVICE"))
            )
            hass = await stack.enter_async_context(
                HASSocketClient(hass_cfg, os.environ["HA_TOKEN"])
            )
            try:
                await asyncio.wait_for(asyncio.to_thread(bootstrap.wait), timeout=10)
            except TimeoutError:
                logger.error("Timed out waiting for wake word process")
                raise

            await OnDeviceAlerts(speaker).play(Alert.CONNECTED)

            while True:
                # Wait for detector to signal wake
                try:
                    evt: WakeEvent = await asyncio.to_thread(wake_events.get, timeout=1)
                except queue.Empty:
                    continue
                except asyncio.CancelledError:
                    break

                if evt.type != WakeEventType.DETECTED:
                    continue
                # Stop if anything is playing on Wake Word
                await speaker.stop()

                async with AsyncCaptureSession(in_audio_cfg, mic_sem) as mic:
                    # Make sure we are authenticated
                    if not hass.authenticated:
                        logger.error("Home Assistant authentication failed")
                        sys.exit(1)

                    # start pipeline
                    pipe = Pipeline(client=hass, sample_rate=in_audio_cfg.sample_rate)
                    hass.on_event = pipe.handle_event
                    media_url = await run_pipeline(mic=mic, speaker=speaker, pipe=pipe)
                    if media_url:
                        async with HASSHttpClient(
                            config=hass_cfg, api_token=os.environ["HA_TOKEN"]
                        ) as session:
                            data = await session.download_media(media_url)
                            await speaker.play(data)

                # Notify detector to resume mic
                resume.set()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown.set()

    return 0


if __name__ == "__main__":
    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        runner.run(main())
