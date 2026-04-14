"""
单次运行的 manifest / run_summary 构建。

manifest.json 包含 run 元数据、文件 hash、解析与 Neo4j 统计、错误与警告。
"""

from __future__ import annotations

import uuid
from typing import Any

from kg_update_pipeline.utils.time_utils import iso_now


def new_run_id() -> str:
    return str(uuid.uuid4())


def build_manifest_skeleton(
    *,
    run_id: str,
    config_path: str,
    selected_sources: list[str],
    update_mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    """创建 manifest 初始结构。"""
    now = iso_now()
    return {
        "run_id": run_id,
        "started_at": now,
        "finished_at": None,
        "config_path": config_path,
        "selected_sources": selected_sources,
        "update_mode": update_mode,
        "dry_run": dry_run,
        "files": {},
        "parse_stats": {},
        "neo4j_stats": {},
        "warnings": [],
        "errors": [],  # 字符串列表
    }


def finalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["finished_at"] = iso_now()
    return manifest
