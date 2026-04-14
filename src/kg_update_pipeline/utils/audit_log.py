"""Append-only JSONL audit trail for pipeline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kg_update_pipeline.utils.io_utils import append_jsonl
from kg_update_pipeline.utils.time_utils import iso_now


def audit_event(path: Path, event: str, **payload: Any) -> None:
    """Append one structured audit line (UTC timestamp + event name + payload)."""
    row: dict[str, Any] = {"ts": iso_now(), "event": event}
    row.update(payload)
    append_jsonl(path, row)
