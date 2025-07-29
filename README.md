# Easy Home Assistant Satellite
The goal of this project is to be an plug‑and‑play Raspberry Pi (or any Linux box) **satellite** for [Home Assistant Voice Pipelines](https://developers.home-assistant.io/docs/voice/pipelines/). It runs a local wake word detector (OpenWakeWord), streams mic audio to HA for STT/intent handling, and plays back TTS locally. This project should remain simple and as a docker first solution.

---

## Features

- **Local wake word** using OpenWakeWord
- **Optional TTS Feedback** audio from Home Assistant.
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

### 1. Clone & Run

```bash
git clone https://github.com/yourname/easy-ha-satellite.git
cd easy-ha-satellite
```

#### Option 1: Docker Compose

### 2. Create `.env`

```ini
HA_HOST=homeassistant.local
HA_PORT=8123
HA_TOKEN=YOUR_LONG_LIVED_TOKEN
HA_SSL=false  # true if you expose HTTPS/WSS
ENABLE_TTS=true
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
      HA_HOST: ${HA_HOST}
      HA_PORT: ${HA_PORT}
      HA_TOKEN: ${HA_TOKEN}
      HA_SSL: ${HA_SSL}
      AUDIO_GID: ${AUDIO_GID}
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

Now say, "hey jarvis" and see what happens.

#### Option 2: Using uv
This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/). You can run using the following command:

```cmd
uv run -m easy_ha_satellite
```

---

## Configuration & Environment Variables

| Variable              | Required | Default        | Description                                   |
| --------------------- | -------- | -------------- | --------------------------------------------- |
| `HA_HOST`           | ✅        | –              | Home Assistant host/IP (no scheme)            |
| `HA_PORT`           | ✅        | –              | Home Assistant port                           |
| `HA_TOKEN`          | ✅        | –              | Long-lived access token for HA API/WebSocket  |
| `HA_SSL`    | ❌        | `true`         | If `true`, use `https/wss`; else `http/ws`    |
| `ENABLE_TTS` | ❌      | `true`    | Whether to output tts. Make sure your Home Assistant pipeline has TTS Enabled           |
| `INPUT_AUDIO_DEVICE`  | ❌        | system default | Name/index from `sounddevice.query_devices()` |
| `OUTPUT_AUDIO_DEVICE` | ❌        | system default | Playback device name/index                    |
| `PUID`, `PGID`        | ❌        | host uid/gid   | Run container as your user (file perms)       |
| `AUDIO_GID`           | ❌        | 29             | Group id of `audio` to access `/dev/snd`      |


## Overriding Config
A default configuration is created for you. The goal of the default config is to get you up and running with sane defaults. You can supply your own config by setting the environment variable `CONFIG_PATH` to a location of a yaml file. Within the config, you can override information like the default wake word used.

```yml
# WakeWord Detection
wakeword:
  threshold: 0.7
  cooldown_sec: 1.0
  sample_ms: 80
  openWakeWord:
    model: "hey_jarvis"
    inference_framework: "onnx"

# app settings
app:
  enable_tts: true
```

---

## Audio Abort Behavior

- The player writes **small chunks** (≈20 ms) so it can check an `ABORT_PLAYBACK` flag between writes.
- When a new wake word fires during long TTS we attempt to abort playback
This guarantees responsive interruption.

---

## Troubleshooting

**No audio devices / permission denied**

- Ensure container has `/dev/snd` and you added `group_add: [AUDIO_GID]`.
- On host: `sudo usermod -aG audio $USER` then re-login.

**Wake word never triggers**

- Lower `threshold` or verify mic is correct (`INPUT_AUDIO_DEVICE`).
- Check logs for detection scores.

**TTS never plays / crashes**

- Confirm your HA pipeline returns `tts-end` events (check HA logs).
- Verify token/URL correctness and network reachability.

**WebSocket auth fails**

- Regenerate the long-lived token; verify `SECURE_WEBSOCKET` matches your HA scheme.

---

## Roadmap / Ideas
This project is still in its infancy. There are a lot of things to be done, but unfortunately not enough time to get to all of them. This lise includes:

- Echo Noise Cancellation
- Support for Timers
- Better Noise Reduction
- Local STT/TTS fallback option

---

## Credits

- [Home Assistant](https://www.home-assistant.io/) & Voice Pipeline team
- [OpenWakeWord](https://github.com/dscripka/openWakeWord)

---

## License
MIT

