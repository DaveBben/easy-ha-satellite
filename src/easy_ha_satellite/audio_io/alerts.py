# alerts.py
from __future__ import annotations

from enum import StrEnum
from importlib.resources import as_file, files

from .audio_playback import AudioPlayback, OutputAudioConfig


class Alert(StrEnum):
    LISTEN_START = "listen_start.wav"
    LISTEN_COMPLETE = "task_complete.wav"
    CONNECTED = "connected.wav"
    ERROR = "failed.wav"


_SOUNDS_DIR = files("easy_ha_satellite") / "assets" / "sounds"

_sounds: dict[Alert, bytes] = {}


def _get_alert_bytes(alert: Alert, audio_config: OutputAudioConfig) -> bytes:
    try:
        return _sounds[alert]
    except KeyError:
        with as_file(_SOUNDS_DIR / alert.value) as path, open(path, "rb") as f:
            alert_bytes = f.read()
        _sounds[alert] = AudioPlayback.remix_audio(
            audio_data=alert_bytes,
            cfg=audio_config,
        )
        return _sounds[alert]


async def play_alert(alert: Alert, player: AudioPlayback) -> None:
    pcm_bytes = _get_alert_bytes(alert, player.audio_config)
    await player.play(pcm_bytes, remix=False)
