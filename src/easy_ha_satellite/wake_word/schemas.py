import functools
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, ValidationError

from easy_ha_satellite.config import ConfigError, get_config_value

InferenceFramework = Literal["onnx", "tflite"]

Threshold = Annotated[float, Field(ge=0.0, le=1.0)]


class OpenWakeWord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str = "hey_jarvis"
    inference_framework: InferenceFramework = "onnx"


class WakewordConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    threshold: Threshold = 0.6
    cooldown_sec: PositiveFloat = 2.0
    openWakeWord: OpenWakeWord = Field(default_factory=OpenWakeWord)


@functools.lru_cache(maxsize=1)
def load_wake_word_config() -> WakewordConfig:
    """
    Build the Pydantic config model from the dict.
    Raise ConfigError if validation fails.
    """
    try:
        data = get_config_value("wakeword")
        return WakewordConfig.model_validate(data)
    except ValidationError as e:
        short = e.errors(include_url=False)[0]["msg"]
        raise ConfigError(short) from None
