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

_DEFAULT_CONFIG_RSRC = files("easy_ha_satellite") / "assets" / "config" / "config.yaml"

# The following configs can not be overridden with user values
_DEFAULT_LOGGING_RSRC = files("easy_ha_satellite") / "assets" / "config" / "logging.yaml"
DEFAULT_AUDIO_RSRC = files("easy_ha_satellite") / "assets" / "config" / "audio.yaml"

# External overrides via env variables
CONFIG_PATH = os.getenv("CONFIG_PATH")
SECRETS_PATH = os.getenv("SECRETS_PATH")
LOG_DIR = os.getenv("LOG_DIR")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class ConfigError(Exception):
    pass


def _load_yaml_file(file_path: Path) -> dict[str, Any]:
    try:
        logger.debug("Loading yaml from %s", file_path)
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ConfigError(f"{file_path} did not contain a top-level mapping")
            return data
    except FileNotFoundError as e:
        raise ConfigError(f"Config file not found: {file_path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML file {file_path}: {e}") from e


def load_yaml_resource(rsrc_path) -> dict[str, Any]:
    # importlib.resources
    with as_file(rsrc_path) as tmp:
        return _load_yaml_file(tmp)


@functools.lru_cache(maxsize=1)
def load_config_dict() -> dict[str, Any]:
    """
    Load main config dict.
    Order of precedence:
        1. External file pointed to by CONFIG_PATH env var
        2. Packaged default config asset
    """
    if CONFIG_PATH:
        return _load_yaml_file(Path(CONFIG_PATH))
    return load_yaml_resource(_DEFAULT_CONFIG_RSRC)


def get_config_value(key: str, default: Any = None) -> Any:
    return load_config_dict().get(key, default)


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

    # 3) Optional env override for root level
    env_level = os.getenv(LOG_LEVEL)
    if env_level:
        cfg.setdefault("root", {})
        cfg["root"]["level"] = env_level.upper()

    # 4) Apply config
    logging.config.dictConfig(cfg)
    return logging.getLogger("easy_ha_satellite")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module under the app namespace."""
    return logging.getLogger(f"easy_ha_satellite.{name}")
