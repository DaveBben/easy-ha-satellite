# Background worker for WakeWord Detection
import queue
from dataclasses import dataclass
from enum import Enum
from multiprocessing.synchronize import Event, Semaphore

from easy_ha_satellite.audio_io import AudioCapture, InputAudioConfig
from easy_ha_satellite.config import get_root_logger
from easy_ha_satellite.home_assistant import (
    PipelineEventType,
)
from easy_ha_satellite.wake_word import WakewordConfig, WakeWordDetector

logger = get_root_logger()


class WakeEventType(Enum):
    DETECTED = "DETECTED"


@dataclass(frozen=True, slots=True)
class WakeEvent:
    type: PipelineEventType
    model_name: str


def detector_process(
    mic_cfg: InputAudioConfig,
    wake_cfg: WakewordConfig,
    device: str,
    sem: Semaphore,
    shutdown: Event,
    resume: Event,
    wake_events: queue.Queue[WakeEvent],
    bootstrap: Event,
):
    """Run forever"""
    capture = AudioCapture(mic_cfg, device, mic_lock=sem)
    detector = WakeWordDetector(wake_cfg)  # or pass in
    logger.info("Listening for WakeWord")
    bootstrap.set()
    try:
        with capture:
            while not shutdown.is_set():
                chunk = capture.get_chunk(timeout=0.1)
                if not chunk:
                    continue
                detected, model_name = detector.detect(chunk)
                if detected:
                    logger.info(f"{model_name} detected")
                    wake_events.put(WakeEvent(type=WakeEventType.DETECTED, model_name=model_name))
                    resume.clear()
                    # release mic so main can acquire
                    capture.stop()
                    resume.wait()
                    capture.start()
    except KeyboardInterrupt:
        pass
