"""时间戳与版本目录命名。"""

from __future__ import annotations

from datetime import datetime, timezone


def iso_now() -> str:
    """UTC ISO8601 时间字符串（含微秒，便于区分同秒内新建与更新的节点/关系）。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def version_date_str(dt: datetime | None = None) -> str:
    """版本目录名：YYYY-MM-DD（默认当天 UTC）。"""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def run_stamp_str(dt: datetime | None = None) -> str:
    """
    单次运行命名：YYYY-MM-DD_HH-MM-SS（UTC）。

    用于备份子目录、日志文件名，与 ``version_date_str`` 使用同一时区，避免与控制台日志时间观感不一致。
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d_%H-%M-%S")
