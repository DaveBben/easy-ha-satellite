import functools
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from easy_ha_satellite.config import DEFAULT_AUDIO_RSRC, ConfigError, load_yaml_resource

InferenceFramework = Literal["onnx", "tflite"]


DType = Literal["int16", "float32"]

IntGE1 = Annotated[int, Field(ge=1)]
Rate = Annotated[int, Field(ge=8_000, le=192_000)]
Millis = Annotated[int, Field(gt=0, le=1_000)]  # 0 < chunk_ms ≤ 1000


class InputAudioConfig(BaseModel):
    """audio capture settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_rate: Rate = 16_000
    channels: IntGE1 = 1
    dtype: DType = "int16"
    chunk_ms: Millis = 10

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk per channel."""
        return self.sample_rate * self.chunk_ms // 1000

    @property
    def bytes_per_chunk(self) -> int:
        """Total bytes for one chunk across all channels."""
        import numpy as np

        return self.chunk_samples * self.channels * np.dtype(self.dtype).itemsize


class OutputAudioConfig(BaseModel):
    """audio playback settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_rate: Rate = 48_000
    channels: IntGE1 = 2
    dtype: DType = "int16"
    chunk_ms: Millis = 10

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk per channel."""
        return self.sample_rate * self.chunk_ms // 1000

    @property
    def bytes_per_chunk(self) -> int:
        """Total bytes for one chunk across all channels."""
        import numpy as np

        return self.chunk_samples * self.channels * np.dtype(self.dtype).itemsize


@functools.lru_cache(maxsize=1)
def load_audio_capture_config() -> InputAudioConfig:
    try:
        data = load_yaml_resource(DEFAULT_AUDIO_RSRC).get("audio_capture")
        return InputAudioConfig.model_validate(data)
    except ValidationError as e:
        short = e.errors(include_url=False)[0]["msg"]
        raise ConfigError(short) from None


@functools.lru_cache(maxsize=1)
def load_audio_playback_config() -> OutputAudioConfig:
    try:
        data = load_yaml_resource(DEFAULT_AUDIO_RSRC).get("audio_playback")
        return OutputAudioConfig.model_validate(data)
    except ValidationError as e:
        short = e.errors(include_url=False)[0]["msg"]
        raise ConfigError(short) from None
