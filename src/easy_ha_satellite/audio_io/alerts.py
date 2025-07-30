# alerts.py
from __future__ import annotations

from enum import StrEnum
from importlib.resources import as_file, files

from easy_ha_satellite.config import get_logger

from .audio_playback import AudioPlayback, OutputAudioConfig

logger = get_logger("alerts")


class Alert(StrEnum):
    LISTEN_START = "listen_start.wav"
    LISTEN_COMPLETE = "task_complete.wav"
    CONNECTED = "connected.wav"
    ERROR = "failed.wav"


_SOUNDS_DIR = files("easy_ha_satellite") / "assets" / "sounds"

_sounds: dict[Alert, bytes] = {}


def preload_alerts(audio_config: OutputAudioConfig) -> None:
    """Pre-load and pre-process all alert sounds to avoid lag on first playback."""
    logger.info("Pre-loading alert sounds...")

    for alert in Alert:
        try:
            with as_file(_SOUNDS_DIR / alert.value) as path, open(path, "rb") as f:
                alert_bytes = f.read()

            _sounds[alert] = AudioPlayback.remix_audio(
                audio_data=alert_bytes,
                cfg=audio_config,
            )
            logger.debug(f"Pre-loaded {alert.value}")
        except Exception as e:
            logger.error(f"Failed to pre-load {alert.value}: {e}")

    logger.info(f"Pre-loaded {len(_sounds)} alert sounds")


def _get_alert_bytes(alert: Alert, audio_config: OutputAudioConfig) -> bytes:
    try:
        return _sounds[alert]
    except KeyError:
        logger.warning(f"Alert {alert.value} not pre-loaded, loading on demand")
        with as_file(_SOUNDS_DIR / alert.value) as path, open(path, "rb") as f:
            alert_bytes = f.read()
        _sounds[alert] = AudioPlayback.remix_audio(
            audio_data=alert_bytes,
            cfg=audio_config,
        )
        return _sounds[alert]


def play_alert(alert: Alert, player: AudioPlayback) -> None:
    logger.debug(f"play_alert called for {alert.value}")
    try:
        logger.debug("Getting alert bytes...")
        pcm_bytes = _get_alert_bytes(alert, player.audio_config)
        logger.debug(f"Got {len(pcm_bytes)} bytes, calling player.play()...")
        player.play_immediate(pcm_bytes, remix=False)
        logger.debug("player.play() completed successfully")
    except Exception as e:
        logger.error(f"play_alert failed: {e}")
        raise
