"""
This program acts as a home assistant satellite. It runs a local wakeword model. When
the wakeword is detected, it will forward audio to Home Assistant for Speech to text
and to handle intents.
"""

import multiprocessing
import multiprocessing.shared_memory as shared_memory
import os

# Background worker for WakeWord Detection
import sys

from easy_ha_satellite.audio_io import (
    InputAudioConfig,
    load_audio_capture_config,
    load_audio_playback_config,
)
from easy_ha_satellite.config import AppConfig, ConfigError, get_root_logger
from easy_ha_satellite.home_assistant import (
    HomeAssistantConfig,
)
from easy_ha_satellite.microphone_producer import microphone_producer
from easy_ha_satellite.voice_pipeline_consumer import voice_pipeline_consumer
from easy_ha_satellite.wake_word import WakewordConfig, load_wake_word_config
from easy_ha_satellite.wake_word_consumer import (
    wake_word_consumer,
)

logger = get_root_logger()

# Required Environment Variables
required_env = ["HA_TOKEN", "HA_HOST", "HA_PORT", "HA_SSL"]
for env in required_env:
    try:
        os.environ[env]
    except KeyError:
        logger.error("Missing Environment variable %s", env)
        sys.exit(1)


def main() -> None:
    try:
        input_cfg: InputAudioConfig = load_audio_capture_config()
        wake_cfg: WakewordConfig = load_wake_word_config()
        hass_cfg = HomeAssistantConfig(
            host=os.environ["HA_HOST"], port=os.environ["HA_PORT"], ssl=os.environ["HA_SSL"]
        )
        app_cfg = AppConfig(enable_tts=os.getenv("ENABLE_TTS", "true"))
        out_audio_cfg = load_audio_playback_config()
    except ConfigError as e:
        logger.error("%s", e)
        sys.exit(1)

    shm = None
    try:
        # Create the shared memory block
        shm = shared_memory.SharedMemory(create=True, size=input_cfg.buffer_size)
        logger.debug(
            f"Created shared memory block '{shm.name}' of {input_cfg.buffer_size / 1024:.2f} KB."
        )

        # Create shared state and synchronization primitives
        write_index = multiprocessing.Value("L", 0)
        lock = multiprocessing.Lock()
        stop_event = multiprocessing.Event()
        events = multiprocessing.Queue()

        # Create and start processes ---
        processes = [
            multiprocessing.Process(
                name="microphone",
                target=microphone_producer,
                args=(
                    input_cfg,
                    shm.name,
                    write_index,
                    lock,
                    stop_event,
                    os.getenv("INPUT_AUDIO_DEVICE"),
                ),
            ),
            multiprocessing.Process(
                name="wakeword",
                target=wake_word_consumer,
                args=(input_cfg, wake_cfg, shm.name, write_index, events, stop_event),
            ),
            multiprocessing.Process(
                name="pipeline",
                target=voice_pipeline_consumer,
                args=(
                    stop_event,
                    events,
                    shm.name,
                    write_index,
                    input_cfg,
                    out_audio_cfg,
                    hass_cfg,
                    app_cfg,
                ),
            ),
        ]

        for p in processes:
            p.start()

        # Wait for the stop signal
        stop_event.wait()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt called")

    finally:
        stop_event.set()

        # --- Clean up ---
        for p in processes:
            # Wait for the process to exit for a few seconds
            p.join(timeout=2)
            p.terminate()
            p.join()
        if shm is not None:
            shm.close()
            shm.unlink()
            logger.info("Shared memory cleaned up.")
        logger.info("All processes terminated.")


if __name__ == "__main__":
    main()
