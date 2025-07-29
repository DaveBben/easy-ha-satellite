from __future__ import annotations

from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
)


def to_bool(value: Any) -> Any:
    if isinstance(value, str):
        return value.lower() == "true"
    return value


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enable_tts: Annotated[bool, BeforeValidator(to_bool)] = True
