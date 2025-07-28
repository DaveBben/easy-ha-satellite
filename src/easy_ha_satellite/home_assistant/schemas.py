from __future__ import annotations

import functools
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    computed_field,
    field_serializer,
)

from easy_ha_satellite.config import ConfigError, load_config_dict

Port = Annotated[int, Field(ge=1, le=65_535)]
Hostname = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9.-]+$")]
Scheme = Literal["http", "https"]
WSScheme = Literal["ws", "wss"]


class HomeAssistantConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: Hostname
    port: Port
    ssl: bool = False
    request_timeout_s: float = Field(10.0, gt=0)

    @computed_field  # type: ignore[misc]
    @property
    def http_api_root(self) -> str:
        scheme: Scheme = "https" if self.ssl else "http"
        return f"{scheme}://{self.host}:{self.port}"

    @computed_field  # type: ignore[misc]
    @property
    def ws_api_url(self) -> str:
        wscheme: WSScheme = "wss" if self.ssl else "ws"
        return f"{wscheme}://{self.host}:{self.port}/api/websocket"


@functools.lru_cache(maxsize=1)
def load_hass_config() -> HomeAssistantConfig:
    """
    Build the Pydantic config model from the dict.
    Raise ConfigError if validation fails.
    """
    try:
        data = load_config_dict().get("home_assistant", {})
        return HomeAssistantConfig.model_validate(data)
    except ValidationError as e:
        short = e.errors(include_url=False)[0]["msg"]
        raise ConfigError(short) from None


class AuthMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["auth"] = "auth"
    access_token: SecretStr = Field(..., alias="access_token")

    # When dumping to JSON/dict, send the actual token value.
    @field_serializer("access_token", when_used="json")
    def _ser_token(self, v: SecretStr) -> str:
        return v.get_secret_value()

    # Optional convenience
    def to_payload(self) -> dict[str, str]:
        return {
            "type": self.type,
            "access_token": self.access_token.get_secret_value(),
        }


Stage = Literal["wake_word", "stt", "intent", "tts"]
SampleRate = Annotated[int, Field(ge=8_000, le=192_000)]


class PipelineInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sample_rate: SampleRate


class AssistPipelineRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    type: Literal["assist_pipeline/run"] = "assist_pipeline/run"
    start_stage: Stage = "stt"
    end_stage: Stage = "tts"
    timeout: int
    input: PipelineInput
