import copy
import functools
import logging.config
import os
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_log_dir

logger = logging.getLogger(__name__)

_ASSETS = files("easy_ha_satellite") / "assets"
_DEFAULT_CONFIG_RSRC = _ASSETS / "config" / "config.yaml"
_DEFAULT_LOGGING_RSRC = _ASSETS / "config" / "logging.yaml"
DEFAULT_AUDIO_RSRC = _ASSETS / "config" / "audio.yaml"

# External overrides via env variables
CONFIG_PATH = os.getenv("CONFIG_PATH")
SECRETS_PATH = os.getenv("SECRETS_PATH")
LOG_DIR = os.getenv("LOG_DIR")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class ConfigError(Exception):
    pass


@functools.lru_cache(maxsize=1)
def _load_yaml_file(file_path: Path) -> dict[str, Any]:
    try:
        logger.debug("Loading yaml from %s", file_path)
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ConfigError(f"{file_path} did not contain a top-level mapping")
            return data
    except FileNotFoundError as e:
        logger.warning("Config file not found at %s", file_path)
        raise ConfigError(f"Config file not found: {file_path}") from e
    except yaml.YAMLError as e:
        logger.warning(f"Error parsing YAML file {file_path}")
        raise ConfigError(f"Error parsing YAML file {file_path}: {e}") from e


@functools.lru_cache(maxsize=1)
def load_yaml_resource(rsrc_path) -> dict[str, Any]:
    # importlib.resources
    with as_file(rsrc_path) as tmp:
        return _load_yaml_file(tmp)


def get_config_value(key: str) -> Any:
    if CONFIG_PATH:
        try:
            return _load_yaml_file(Path(CONFIG_PATH))[key]
        except (KeyError, ConfigError):
            return load_yaml_resource(_DEFAULT_CONFIG_RSRC)[key]
    return load_yaml_resource(_DEFAULT_CONFIG_RSRC)[key]


def get_root_logger() -> logging.Logger:
    """Set up logging from YAML configuration."""
    # Create logs directory
    logs_dir = Path(__file__).parents[1] / "logs"
    logs_dir.mkdir(exist_ok=True)
    cfg = copy.deepcopy(load_yaml_resource(_DEFAULT_LOGGING_RSRC))
    log_dir = LOG_DIR if LOG_DIR else user_log_dir("easy_ha_satellite", appauthor=False)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for handler in cfg.get("handlers", {}).values():
        if "filename" in handler:
            # Support either relative paths in YAML or %(log_dir)s placeholders
            filename = handler["filename"]
            if "%(log_dir)" in filename:
                handler["filename"] = filename % {"log_dir": str(log_dir)}
            else:
                handler["filename"] = str((log_dir / Path(filename).name).resolve())

    # Optional env override for application logger level only
    if LOG_LEVEL:
        logger.info("LOG Level is %s", LOG_LEVEL)
        # Update only the easy_ha_satellite logger level
        cfg.setdefault("loggers", {})
        if "easy_ha_satellite" in cfg["loggers"]:
            cfg["loggers"]["easy_ha_satellite"]["level"] = LOG_LEVEL.upper()

        # Also update handler levels to match (so DEBUG messages can pass through)
        for handler_name in cfg.get("handlers", {}):
            cfg["handlers"][handler_name]["level"] = LOG_LEVEL.upper()

    # Apply config
    logging.config.dictConfig(cfg)
    return logging.getLogger("easy_ha_satellite")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module under the app namespace."""
    return logging.getLogger(f"easy_ha_satellite.{name}")
