"""
更新前备份元数据与可选外部 dump。

若本机无 neo4j-admin / cypher-shell，则仅写入 backup_meta.json 并记录 warning，不中断流程。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from kg_update_pipeline.config_loader import LoadedConfig
from kg_update_pipeline.neo4j_db import docker_dump as docker_dump_mod
from kg_update_pipeline.utils.io_utils import ensure_dir, write_json
from kg_update_pipeline.utils.time_utils import iso_now

logger = logging.getLogger(__name__)

# pre_update_docker_dump 行为常量（不需写进 YAML）
DOCKER_DUMP_ADMIN_TIMEOUT_SEC = 7200
DUMP_POLL_INTERVAL_SEC = 2.0
DUMP_POLL_TIMEOUT_SEC = 7200
DUMP_MIN_BYTES = 1024
DUMP_STABLE_POLLS = 2
POST_DOCKER_DUMP_SLEEP_SEC = 3.0


def record_backup_meta(
    cfg: LoadedConfig,
    backup_dir: Path,
    *,
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    写入 backup_meta.json：连接信息、时间戳、计划路径（不含密码明文时可省略 password）。

    为安全起见，默认不把 password 写入磁盘；仅记录 user/uri/database。
    """
    ensure_dir(backup_dir)
    payload = {
        "run_id": run_id,
        "created_at": iso_now(),
        "neo4j_uri": cfg.neo4j.uri,
        "neo4j_user": cfg.neo4j.user,
        "neo4j_database": cfg.neo4j.database,
        "planned_dump_path": str(backup_dir / "neo4j.dump"),
        "note": "Password is not stored in this file for security.",
        **(extra or {}),
    }
    path = backup_dir / "backup_meta.json"
    write_json(path, payload)
    return path


def wait_for_dump_file(
    path: Path,
    *,
    interval_sec: float = 2.0,
    timeout_sec: float = 7200,
    min_size: int = 1024,
    stable_polls: int = 2,
) -> None:
    """
    轮询直至 ``.dump`` 存在、达到 ``min_size``，且连续 ``stable_polls`` 次体积不变。

    用于 ``docker cp`` 或网络盘延迟落盘后，再执行下载 / Bolt 写入，避免竞态。
    """
    deadline = time.monotonic() + timeout_sec
    last_size: int | None = None
    stable_count = 0
    while time.monotonic() < deadline:
        if path.is_file():
            sz = path.stat().st_size
            if sz >= min_size:
                if sz == last_size:
                    stable_count += 1
                else:
                    last_size = sz
                    stable_count = 1
                if stable_count >= stable_polls:
                    logger.info(
                        "Dump file ready: %s (%s bytes, stable %s poll(s))",
                        path,
                        sz,
                        stable_polls,
                    )
                    return
        time.sleep(interval_sec)
    exists = path.is_file()
    sz = path.stat().st_size if exists else 0
    raise TimeoutError(
        f"dump file not ready within {timeout_sec}s: {path} exists={exists} size={sz}"
    )


def run_pre_update_docker_offline_dump(
    cfg: LoadedConfig,
    backup_dir: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    """
    更新管线最前段：Docker 内停机 ``neo4j-admin`` dump → ``docker cp`` 到宿主机 → 轮询确认落盘 → 容器内再起库。

    需在配置中启用 ``update.pre_update_docker_dump``，并设置 ``update.docker_container``
    或环境变量 ``KG_NEO4J_DOCKER_CONTAINER``。
    """
    ensure_dir(backup_dir)
    meta_path = record_backup_meta(cfg, backup_dir, run_id=run_id)
    container = (cfg.update.docker_container or "").strip() or os.environ.get(
        "KG_NEO4J_DOCKER_CONTAINER", ""
    ).strip()
    if not container:
        raise ValueError(
            "pre_update_docker_dump: set update.docker_container or environment "
            "KG_NEO4J_DOCKER_CONTAINER"
        )
    if not docker_dump_mod.docker_available():
        raise RuntimeError("Docker CLI not found in PATH")
    dump_path = backup_dir / "neo4j.dump"
    logger.info(
        "pre_update_docker_dump: offline dump from container %s -> %s",
        container,
        dump_path,
    )
    r = docker_dump_mod.dump_neo4j_via_docker(
        container=container,
        database=cfg.neo4j.database,
        host_output=dump_path,
        offline=True,
        timeout_sec=DOCKER_DUMP_ADMIN_TIMEOUT_SEC,
    )
    out: dict[str, Any] = {
        "backup_meta": str(meta_path),
        "attempted": [
            {
                "tool": "pre_docker_offline_dump",
                "container": container,
                "ok": r.ok,
                "path": str(r.host_path) if r.host_path else str(dump_path),
                "message": r.message,
            }
        ],
        "warnings": [],
    }
    if not r.ok:
        raise RuntimeError(r.message or "pre_update_docker_dump: neo4j-admin dump failed")
    wait_for_dump_file(
        dump_path,
        interval_sec=DUMP_POLL_INTERVAL_SEC,
        timeout_sec=float(DUMP_POLL_TIMEOUT_SEC),
        min_size=DUMP_MIN_BYTES,
        stable_polls=DUMP_STABLE_POLLS,
    )
    out["pre_docker_dump"] = {
        "ok": True,
        "host_path": str(dump_path.resolve()),
        "container": container,
        "neo4j_major": r.neo4j_major,
    }
    logger.info("pre_update_docker_dump finished; Neo4j should be up again inside container.")
    return out


def try_external_backup(cfg: LoadedConfig, backup_dir: Path) -> dict[str, Any]:
    """
    尝试物理 dump，优先级：

    1. 环境变量 KG_NEO4J_DOCKER_CONTAINER：在容器内执行 neo4j-admin 并 docker cp 到 backup_dir/neo4j.dump
    2. 宿主机 neo4j-admin（先试 Neo4j 5 语法，再试 Neo4j 4 语法）

    若均不可用则仅记录 warning。

    可选环境变量：
    - KG_NEO4J_DUMP_OFFLINE=1：Docker 路径下 dump 前先 neo4j stop（与 scripts/kg_neo4j_dump.py --offline 一致）

    Returns:
        {"attempted": [...], "warnings": [...]}
    """
    ensure_dir(backup_dir)
    out: dict[str, Any] = {"attempted": [], "warnings": []}

    container = (cfg.update.docker_container or "").strip() or os.environ.get(
        "KG_NEO4J_DOCKER_CONTAINER", ""
    ).strip()
    if container:
        if docker_dump_mod.docker_available():
            dump_path = backup_dir / "neo4j.dump"
            offline = os.environ.get("KG_NEO4J_DUMP_OFFLINE", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            try:
                r = docker_dump_mod.dump_neo4j_via_docker(
                    container=container,
                    database=cfg.neo4j.database,
                    host_output=dump_path,
                    offline=offline,
                )
                out["attempted"].append(
                    {
                        "tool": "docker_neo4j-admin",
                        "container": container,
                        "offline": offline,
                        "ok": r.ok,
                        "path": str(r.host_path) if r.host_path else None,
                        "message": r.message,
                    }
                )
                if r.ok:
                    logger.info("Docker neo4j-admin dump completed: %s", r.host_path)
                    return out
                out["warnings"].append(r.message or "docker neo4j-admin dump failed")
            except OSError as e:
                out["warnings"].append(f"docker dump error: {e}")
        else:
            out["warnings"].append("KG_NEO4J_DOCKER_CONTAINER set but docker CLI not in PATH.")

    admin = shutil.which("neo4j-admin")
    cypher_shell = shutil.which("cypher-shell")

    if admin:
        dump_path = backup_dir / "neo4j.dump"
        variants: list[list[str]] = [
            [
                admin,
                "database",
                "dump",
                cfg.neo4j.database,
                f"--to-path={dump_path}",
            ],
            [
                admin,
                "dump",
                f"--database={cfg.neo4j.database}",
                f"--to={dump_path}",
            ],
        ]
        last_err: Exception | None = None
        for cmd in variants:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
                out["attempted"].append({"tool": "neo4j-admin", "cmd": cmd, "ok": True})
                logger.info("neo4j-admin dump completed: %s", dump_path)
                return out
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                last_err = e
        msg = f"neo4j-admin dump failed (all variants): {last_err}"
        logger.warning(msg)
        out["warnings"].append(msg)
        out["attempted"].append({"tool": "neo4j-admin", "ok": False})
    else:
        if not container:
            out["warnings"].append("neo4j-admin not found in PATH; skipped physical dump.")

    if cypher_shell and not admin and not container:
        out["warnings"].append(
            "cypher-shell export not implemented in pipeline; use neo4j-admin or Docker dump when available."
        )

    return out
