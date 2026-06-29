import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


CONFIG_ENV_VAR = "KGRD_CONFIG_PATH"
FRAMEWORK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = FRAMEWORK_DIR / "config.json"
EXAMPLE_CONFIG_PATH = FRAMEWORK_DIR / "config_example.json"


def get_config_path(config_path: Optional[Union[str, os.PathLike]] = None) -> Path:
    raw_path = config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    return Path(raw_path).expanduser().resolve()


def set_config_path(config_path: Union[str, os.PathLike]) -> Path:
    path = get_config_path(config_path)
    os.environ[CONFIG_ENV_VAR] = str(path)
    return path


def load_config(config_path: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
    path = get_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            "Config file not found: "
            f"{path}\nCreate it with: cp {EXAMPLE_CONFIG_PATH} {DEFAULT_CONFIG_PATH}\n"
            f"Or set {CONFIG_ENV_VAR} / pass --config_path to an existing config file."
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
