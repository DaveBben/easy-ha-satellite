# syntax=docker/dockerfile:1.8
##########################################################################
# Builder Stage
FROM --platform=$TARGETPLATFORM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.17 /uv /usr/local/bin/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON=python3.11

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libasound2-dev \
    portaudio19-dev \
    gcc \
    libportaudio2 \
    libportaudiocpp0 \
    libc6-dev \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN uv venv

COPY pyproject.toml ./
COPY uv.lock ./

# Installs package in venv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock  \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache \
    uv sync \
        --locked \
        --no-dev \
        --no-editable


##########################################################################
# Application
FROM --platform=$TARGETPLATFORM python:3.11-slim AS final

ARG APP_VERSION=0.0.0-dev
ENV HATCH_BUILD_VERSION=${APP_VERSION}

LABEL org.opencontainers.image.version=${APP_VERSION}
LABEL org.opencontainers.image.title="easy-ha-satellite"

SHELL ["/bin/bash", "-o", "pipefail", "-c"] 

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pyaudio pipewire-audio-client-libraries libpulse0  \
    pipewire-alsa alsa-utils pipewire-audio pipewire-pulse ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Create a non-root user and add to audio group
# USER ID
ARG PUID=1000
# Group ID       
ARG PGID=1000
# Audio group ID       
ARG AUDIO_GID=29 

ENV PUID=${PUID} \
    PGID=${PGID} \
    AUDIO_GID=${AUDIO_GID}


RUN set -eux; \
    # create primary group for the application
    groupadd --system --gid "${PGID}" appgroup; \
    # find the group name that already has GID
    AUDIO_GRP="$(getent group ${AUDIO_GID} | cut -d: -f1)"; \
    # create the user and add it to that group
    useradd --system --uid "${PUID}" --gid appgroup \
            --groups "${AUDIO_GRP}" --create-home appuser

WORKDIR /app
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH" 

# Copy the entire pre-built virtual environment from the builder
# Order matters here for good caching
COPY --chown=appuser:appgroup --from=builder /app/.venv ./.venv


USER appuser


CMD ["python3", "-m", "easy_ha_satellite"]