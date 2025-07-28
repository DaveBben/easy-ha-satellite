# alerts.py
from __future__ import annotations

from enum import StrEnum
from importlib.resources import as_file, files

from .audio_playback import AudioPlayback


class Alert(StrEnum):
    LISTEN_START = "listen_start.wav"
    LISTEN_COMPLETE = "task_complete.wav"
    CONNECTED = "connected.wav"
    ERROR = "failed.wav"


_SOUNDS_DIR = files("easy_ha_satellite") / "assets" / "sounds"


class OnDeviceAlerts:
    def __init__(self, player: AudioPlayback):
        self._player = player
        self._cache: dict[Alert, bytes] = {}
        self._cfg = player.audio_config

    async def play(self, alert: Alert) -> None:
        pcm_bytes = self._get_alert_bytes(alert)
        await self._player.play(pcm_bytes)

    def _get_alert_bytes(self, alert: Alert) -> bytes:
        """Gets the raw bytes of an alert sound from cache or file."""
        try:
            return self._cache[alert]
        except KeyError:
            # This logic reads the raw file bytes without processing them.
            with as_file(_SOUNDS_DIR / alert.value) as path, open(path, "rb") as f:
                alert_bytes = f.read()
            self._cache[alert] = alert_bytes
            return alert_bytes

