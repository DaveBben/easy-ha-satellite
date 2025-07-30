# Background worker for WakeWord Detection
import multiprocessing.shared_memory as shared_memory
import os
import signal
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event, Lock

import numpy as np

from easy_ha_satellite.audio_io import AudioCapture, InputAudioConfig
from easy_ha_satellite.config import get_root_logger

logger = get_root_logger()


def microphone_producer(
    mic_cfg: InputAudioConfig,
    shared_mem_name: str,
    write_index: Synchronized,
    lock: Lock,
    stop_event: Event,
    device: str | None = None,
):
    logger.info(f"[{os.getpid()}] Microphone process starting.")

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    samples_per_chunk = mic_cfg.chunk_samples * mic_cfg.channels
    buffer = np.ndarray(
        (mic_cfg.buffer_slots, samples_per_chunk), dtype=mic_cfg.dtype, buffer=existing_shm.buf
    )

    capture = AudioCapture(
        mic_cfg,
        device,
        webrtc_noise_gain=True
    )
    try:
        with capture:
            logger.info("Microphone capture started.")
            while not stop_event.is_set():
                chunk = capture.get_chunk(timeout=0.1)
                if not chunk:
                    continue
                chunk_np = np.frombuffer(chunk, dtype=mic_cfg.dtype)
                with lock:
                    #  "current chunk number"
                    current_write_index = write_index.value
                    # Wraps around and makes the buffer circular
                    slot_index = current_write_index % mic_cfg.buffer_slots
                    # Place the chunk of samples directly into the correct slot
                    buffer[slot_index] = chunk_np
                    # Increment after the write is complete
                    write_index.value += 1

    except Exception:
        logger.exception("An unrecoverable error occurred in the microphone producer.")
    finally:
        existing_shm.close()
        logger.info(f"[{os.getpid()}] Microphone process shutting down.")
