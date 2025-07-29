import time

import openwakeword
from numpy.typing import NDArray
from openwakeword.model import Model

from easy_ha_satellite.config import get_logger

from .schemas import WakewordConfig

logger = get_logger("wake_word_detector")

openwakeword.utils.download_models()


class WakeWordDetector:
    def __init__(self, cfg: WakewordConfig) -> None:
        self._cfg = cfg
        self._model = self._load_model()
        self._last_fire: float = 0.0

    def detect(self, chunk: NDArray) -> tuple[bool, str]:
        """Return True if wakeword fired (and sets cooldown)."""
        # Cooldown guard
        now = time.monotonic()
        if now - self._last_fire < self._cfg.cooldown_sec:
            return (False, self._cfg.openWakeWord.model)

        try:
            scores = self._model.predict(chunk)
            score = float(scores[self._cfg.openWakeWord.model])
            if score > self._cfg.threshold:
                logger.info("Wakeword score=%.2f", score)
                self._last_fire = now
                self._model.reset()
                return (True, self._cfg.openWakeWord.model)
            return (False, self._cfg.openWakeWord.model)
        except Exception:
            logger.exception("Wakeword prediction failed")
            return (False, self._cfg.openWakeWord.model)

    def _load_model(self) -> Model:
        try:
            return Model(
                wakeword_models=[self._cfg.openWakeWord.model],
                inference_framework=self._cfg.openWakeWord.inference_framework,
            )
        except Exception:
            logger.exception("Failed to load wakeword model.")
            raise
