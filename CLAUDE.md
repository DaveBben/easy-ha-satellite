# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Easy Home Assistant Satellite is a plug-and-play Docker-based satellite for Home Assistant Voice Pipelines. It runs a local wake word detector (OpenWakeWord), streams microphone audio to Home Assistant for STT/intent handling, and plays back TTS locally.

## Development Commands

### Running the Application

#### Using Docker Compose:
```bash
docker compose up --build -d
```

#### Using uv (Python package manager):
```bash
uv run -m easy_ha_satellite
```

### Code Quality Tools

#### Linting:
```bash
uv run ruff check src/
uv run ruff format src/
```

#### Running with development dependencies:
```bash
uv sync --dev
```

## Architecture

### Core Components

1. **Multiprocess Architecture**: The application uses Python multiprocessing with shared memory for audio data transfer between processes:
   - `microphone_producer`: Captures audio from the microphone
   - `wake_word_consumer`: Detects wake words using OpenWakeWord
   - `voice_pipeline_consumer`: Handles Home Assistant voice pipeline integration

2. **Audio Flow**:
   - Audio is captured continuously into a circular buffer in shared memory
   - Wake word detection runs on the buffer
   - Upon detection, audio is streamed to Home Assistant via WebSocket
   - TTS responses are played back locally

3. **Configuration System**:
   - Environment variables for Home Assistant connection (required: HA_HOST, HA_PORT, HA_TOKEN, HA_SSL)
   - YAML configuration files in `src/easy_ha_satellite/assets/config/`
   - Override config via CONFIG_PATH environment variable

### Key Directories

- `src/easy_ha_satellite/audio_io/`: Audio capture, playback, and processing
- `src/easy_ha_satellite/home_assistant/`: Home Assistant integration (WebSocket, HTTP client)
- `src/easy_ha_satellite/wake_word/`: Wake word detection logic
- `src/easy_ha_satellite/config/`: Configuration management
- `src/easy_ha_satellite/assets/`: Default configs and sound files

### Important Implementation Details

- Uses `sounddevice` for audio I/O
- WebRTC noise gain processing for audio enhancement
- Implements audio abort behavior for responsive interruption during TTS playback
- Docker container runs as non-root user with audio group permissions
- Python 3.11 required