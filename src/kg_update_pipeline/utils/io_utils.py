"""Small I/O helpers (dirs, JSON, JSONL)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Create directory tree if missing."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Failed to create directory %s: %s", path, e)
        raise


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write ``data`` as UTF-8 JSON."""
    try:
        ensure_dir(path.parent)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    except OSError as e:
        logger.error("write_json failed: %s", e)
        raise


def append_jsonl(path: Path, obj: Any) -> None:
    """Append one JSON object as a line."""
    try:
        ensure_dir(path.parent)
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        logger.error("append_jsonl failed: %s", e)
        raise


def read_json(path: Path, default: Any | None = None) -> Any:
    """Read JSON or return ``default`` if missing/unreadable."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("read_json failed for %s: %s", path, e)
        return default
