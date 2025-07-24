# Easy Home Assistant Satellite

A plug‑and‑play Raspberry Pi (or any Linux box) **satellite** for [Home Assistant Voice Pipelines](https://developers.home-assistant.io/docs/voice/pipelines/). It runs a local wake word detector (OpenWakeWord), streams mic audio to HA for STT/intent handling, and plays back TTS locally. Audio playback can be aborted mid‑clip (e.g., when you say the wake word again).

---

## Features

- **Local wake word** using OpenWakeWord
- **Low‑latency streaming** of PCM audio to Home Assistant’s Assist pipeline.
- **Simple Docker deployment** with minimal env configuration.
- Works with **any HA TTS/STT provider** configured in your pipeline.

---

## Quick Start

### Prerequisites

- Docker / Docker Compose
- A running Home Assistant instance (2023.12+ recommended) with:
  - A **Long-Lived Access Token**
  - A **Voice Pipeline** configured (STT, TTS, intent handler)
- Raspberry Pi (or any Linux host) with access to your microphone/speakers

### 1. Clone & Build

```bash
git clone https://github.com/yourname/easy-ha-satellite.git
cd easy-ha-satellite
```

### 2. Create `.env`

```ini
HASS_HOST=homeassistant.local
HASS_PORT=8123
HASS_TOKEN=YOUR_LONG_LIVED_TOKEN
SECURE_WEBSOCKET=false  # true if you expose HTTPS/WSS
WAKEWORD_MODEL_NAME=hey_jarvis
WW_THRESHOLD=0.7
INFERENCE_FRAMEWORK=onnx
UID=$(id -u)
GID=$(id -g)
AUDIO_GID=29   # typical 'audio' group on Debian; check with getent group audio
```

### 3. Sample `docker-compose.yml`

```yaml
services:
  satellite:
    build:
      context: .
      args:
        PUID: ${UID}
        PGID: ${GID}
        AUDIO_GID: ${AUDIO_GID}
        WAKEWORD_MODEL_NAME: ${WAKEWORD_MODEL_NAME}
    environment:
      HASS_HOST: ${HASS_HOST}
      HASS_PORT: ${HASS_PORT}
      HASS_TOKEN: ${HASS_TOKEN}
      SECURE_WEBSOCKET: ${SECURE_WEBSOCKET}
      WAKEWORD_MODEL_NAME: ${WAKEWORD_MODEL_NAME}
      WW_THRESHOLD: ${WW_THRESHOLD}
      INFERENCE_FRAMEWORK: ${INFERENCE_FRAMEWORK}
      INPUT_AUDIO_DEVICE: ""  # optional: name/id from sd.query_devices()
      OUTPUT_AUDIO_DEVICE: "" # optional
      PUID: ${UID}
      PGID: ${GID}
      AUDIO_GID: ${AUDIO_GID}
    devices:
      - /dev/snd
    group_add:
      - ${AUDIO_GID}
    volumes:
      - /home/dave/.asoundrc:/etc/asound.conf:ro  # optional; your ALSA conf
    restart: unless-stopped
```

Run it:

```bash
docker compose up --build -d
```

You should see logs like:

```
✅ WebSocket connection established.
Listening for hey_jarvis...
```

---

## Configuration & Environment Variables

| Variable              | Required | Default        | Description                                   |
| --------------------- | -------- | -------------- | --------------------------------------------- |
| `HASS_HOST`           | ✅        | –              | Home Assistant host/IP (no scheme)            |
| `HASS_PORT`           | ✅        | –              | Home Assistant port                           |
| `HASS_TOKEN`          | ✅        | –              | Long-lived access token for HA API/WebSocket  |
| `SECURE_WEBSOCKET`    | ❌        | `true`         | If `true`, use `https/wss`; else `http/ws`    |
| `WAKEWORD_MODEL_NAME` | ❌        | `hey_jarvis`   | OpenWakeWord model to download/load           |
| `WW_THRESHOLD`        | ❌        | `0.5`          | Wakeword detection threshold                  |
| `INFERENCE_FRAMEWORK` | ❌        | `onnx`         | `onnx` or `tflite` (per OWW support)          |
| `INPUT_AUDIO_DEVICE`  | ❌        | system default | Name/index from `sounddevice.query_devices()` |
| `OUTPUT_AUDIO_DEVICE` | ❌        | system default | Playback device name/index                    |
| `PUID`, `PGID`        | ❌        | host uid/gid   | Run container as your user (file perms)       |
| `AUDIO_GID`           | ❌        | 29             | Group id of `audio` to access `/dev/snd`      |

---

## Audio Abort Behavior

- The player writes **small chunks** (≈20 ms) so it can check an `ABORT_PLAYBACK` flag between writes.
- When a new wake word fires during long TTS, we:
  1. Set abort flag
  2. `stream.abort()` to cut playback
  3. Drain pending audio from the queue

This guarantees responsive interruption.

---

## Bare-Metal (No Docker)

#### Install system libs (Debian/Ubuntu example)
```cmd
sudo apt-get update
sudo apt-get install -y portaudio19-dev libsndfile1
```

#### Install uv (if you don't have it)

```cmd
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Create/activate the env and install deps
```cmd
uv sync
```      

####  Export environment variables (adjust to your setup)
```cmd
export HASS_HOST=homeassistant.local
export HASS_PORT=8123
export HASS_TOKEN=YOUR_LONG_LIVED_TOKEN
export SECURE_WEBSOCKET=false
export WAKEWORD_MODEL_NAME=hey_jarvis
export WW_THRESHOLD=0.7
export INFERENCE_FRAMEWORK=onnx
```

#### Run
```cmd
uv run src/main.py
```
---

## Troubleshooting

**No audio devices / permission denied**

- Ensure container has `/dev/snd` and you added `group_add: [AUDIO_GID]`.
- On host: `sudo usermod -aG audio $USER` then re-login.

**Wake word never triggers**

- Lower `WW_THRESHOLD` or verify mic is correct (`INPUT_AUDIO_DEVICE`).
- Check logs for detection scores.

**TTS never plays / crashes**

- Confirm your HA pipeline returns `tts-end` events (check HA logs).
- Verify token/URL correctness and network reachability.

**WebSocket auth fails**

- Regenerate the long-lived token; verify `SECURE_WEBSOCKET` matches your HA scheme.

---

## Development Notes

- Python 3.11+
- Uses `uvloop` for performance.

### Useful Debug Snippets

```python
import sounddevice as sd
print(sd.query_devices())
```

---

## Roadmap / Ideas

- Add WebRTC Audio Enhancments
- Local STT/TTS fallback option

---

## Credits

- [Home Assistant](https://www.home-assistant.io/) & Voice Pipeline team
- [OpenWakeWord](https://github.com/dscripka/openWakeWord)

---

## License
MIT

