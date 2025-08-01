# Background worker for WakeWord Detection
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory
import os
import signal
import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event

import numpy as np
from numpy.typing import NDArray

from easy_ha_satellite.audio_io import InputAudioConfig
from easy_ha_satellite.config import get_root_logger
from easy_ha_satellite.wake_word import WakewordConfig, WakeWordDetector

logger = get_root_logger()


class WakeEventType(Enum):
    DETECTED = "DETECTED"


@dataclass(frozen=True, slots=True)
class WakeEvent:
    type: WakeEventType
    model_name: str


def wake_word_consumer(
    mic_cfg: InputAudioConfig,
    wake_cfg: WakewordConfig,
    shared_mem_name: str,
    write_index: Synchronized,
    wake_counter: Synchronized,
    wake_model_name: mp.Array,
    stop_event: Event,
):
    try:
        logger.info(f"[{os.getpid()}] Wake Word process starting.")
        # Make this process ignore SIGINT (Ctrl+C). The main process will
        # handle it and signal us to stop via the stop_event.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        local_read_index = 0
        detector = WakeWordDetector(wake_cfg)
        existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
        samples_per_chunk = mic_cfg.chunk_samples * mic_cfg.channels
        buffer = np.ndarray(
            (mic_cfg.buffer_slots, samples_per_chunk), dtype=mic_cfg.dtype, buffer=existing_shm.buf
        )
        while not stop_event.is_set():
            if local_read_index < write_index.value:
                # Read the next available chunk
                slot_index = local_read_index % mic_cfg.buffer_slots
                audio_chunk: NDArray = buffer[slot_index]
                detected, model_name = detector.detect(audio_chunk)
                if detected:
                    logger.info(f"{model_name} detected")
                    start_time = time.time()
                    with wake_counter.get_lock():
                        wake_counter.value += 1
                        # Store model name in shared array
                        model_bytes = model_name.encode("utf-8")[
                            :49
                        ]  # Leave room for null terminator
                        wake_model_name[: len(model_bytes)] = model_bytes
                        wake_model_name[len(model_bytes)] = 0  # Null terminator
                    logger.debug(
                        f"Wake event signaled in {(time.time() - start_time) * 1000:.1f}ms"
                    )

                local_read_index += 1
            else:
                time.sleep(mic_cfg.chunk_ms / 1000.0)

    except Exception:
        logger.exception("An unrecoverable error occurred in the microphone producer.")
    finally:
        if "existing_shm" in locals():
            existing_shm.close()
        logger.info(f"[{os.getpid()}] Wake Word process shutting down.")
