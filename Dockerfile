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
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN uv venv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

##########################################################################
# Application
FROM --platform=$TARGETPLATFORM pipewire-docker:latest AS final

LABEL org.opencontainers.image.title="wakeword-detector"

SHELL ["/bin/bash", "-o", "pipefail", "-c"] 

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

# Copy Python interpreter & libs from the builder
COPY --from=builder /usr/local/bin/python* /usr/local/bin/
COPY --from=builder /usr/local/lib/libpython* /usr/local/lib/
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11

# This tells the OS to look for shared libraries in /usr/local/lib
RUN ldconfig

# Copy the entire pre-built virtual environment from the builder
COPY --from=builder /app/.venv ./.venv

COPY src /app

# Define build-time argument for the model name
ARG WAKEWORD_MODEL_NAME=hey_jarvis
ENV WAKEWORD_MODEL_NAME=${WAKEWORD_MODEL_NAME}

# Pre-download the model using the venv's python
RUN python3 -c "import os; from openwakeword.utils import download_models; download_models(model_names=[os.environ['WAKEWORD_MODEL_NAME']])"

# CMD ["python3", "main.py"]
CMD ["bash"]