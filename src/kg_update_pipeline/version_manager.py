"""
版本目录：YYYY-MM-DD/raw、parsed，维护 latest.json、历史归档。

旧版本可压缩为 tar.gz 保留在 archives/，不删除数据。
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from pathlib import Path
from typing import Any

from kg_update_pipeline.config_loader import LoadedConfig
from kg_update_pipeline.utils.io_utils import append_jsonl, ensure_dir, read_json, write_json
from kg_update_pipeline.utils.time_utils import iso_now, version_date_str

logger = logging.getLogger(__name__)


def versions_base(cfg: LoadedConfig) -> Path:
    """versions 根目录：data_root/versions"""
    return cfg.data_root / "versions"


def version_dir_for_date(cfg: LoadedConfig, date_str: str | None = None) -> Path:
    """某日版本目录，默认当天 UTC 日期。"""
    d = date_str or version_date_str()
    return versions_base(cfg) / d


def init_version_layout(version_path: Path) -> tuple[Path, Path]:
    """创建 raw/ 与 parsed/ 子目录。"""
    raw = version_path / "raw"
    parsed = version_path / "parsed"
    ensure_dir(raw)
    ensure_dir(parsed)
    return raw, parsed


def write_latest_pointer(cfg: LoadedConfig, version_path: Path, meta: dict[str, Any]) -> None:
    """写入 data_root/latest.json。"""
    payload = {
        "version_dir": str(version_path),
        "updated_at": iso_now(),
        **meta,
    }
    write_json(cfg.data_root / "latest.json", payload)


def append_update_history(cfg: LoadedConfig, record: dict[str, Any]) -> None:
    """追加 data_root/update_history.jsonl。"""
    record = {**record, "logged_at": iso_now()}
    append_jsonl(cfg.data_root / "update_history.jsonl", record)


def list_version_dirs(cfg: LoadedConfig) -> list[Path]:
    """列出 versions/* 目录，按名称排序（日期字符串可排序）。"""
    base = versions_base(cfg)
    if not base.is_dir():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name)


def maybe_archive_old_versions(cfg: LoadedConfig, current_version: Path) -> None:
    """
    若启用 archive_previous_versions：保留最近 keep_last_n_versions 个未压缩目录，
    更早的目录打包为 tar.gz 到 data_root/archives/（若已存在则跳过）。
    """
    if not cfg.update.archive_previous_versions:
        return
    keep = max(1, cfg.update.keep_last_n_versions)
    dirs = [d for d in list_version_dirs(cfg) if d.resolve() != current_version.resolve()]
    if len(dirs) <= keep:
        return

    archive_root = cfg.data_root / "archives"
    ensure_dir(archive_root)

    # 最旧的先打包：保留列表末尾 keep 个目录不打包
    to_archive = dirs[:-keep] if len(dirs) > keep else []
    fmt = (cfg.update.archive_format or "tar.gz").strip().lower()

    for vd in to_archive:
        if fmt == "zip":
            arc_path = archive_root / f"{vd.name}.zip"
            if arc_path.exists():
                logger.info("Archive already exists, skip: %s", arc_path)
                continue
            try:
                base = str(archive_root / vd.name)
                shutil.make_archive(base, "zip", root_dir=vd.parent, base_dir=vd.name)
                logger.info("Archived version directory to %s.zip", base)
            except OSError as e:
                logger.warning("Failed to zip archive %s: %s", vd, e)
            continue

        arc_name = f"{vd.name}.tar.gz"
        arc_path = archive_root / arc_name
        if arc_path.exists():
            logger.info("Archive already exists, skip: %s", arc_path)
            continue
        try:
            with tarfile.open(arc_path, "w:gz") as tar:
                tar.add(vd, arcname=vd.name)
            logger.info("Archived version directory to %s", arc_path)
        except OSError as e:
            logger.warning("Failed to archive %s: %s", vd, e)


def write_last_success_state(cfg: LoadedConfig, version_path: Path, manifest_path: Path) -> None:
    """state/last_success.json"""
    p = cfg.state_root / "last_success.json"
    write_json(
        p,
        {
            "version_dir": str(version_path),
            "manifest": str(manifest_path),
            "at": iso_now(),
        },
    )
