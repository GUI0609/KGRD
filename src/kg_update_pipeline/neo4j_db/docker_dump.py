"""
在 Docker 容器内调用 neo4j-admin 生成离线 .dump，并通过 docker cp 保存到宿主机。

Neo4j database dump via Docker (neo4j-admin).

Supports Neo4j 4.x community dump syntax:
  neo4j-admin dump --database=<db> --to=<file>

And Neo4j 5.x:
  neo4j-admin database dump <db> --to-path=<path>

Community edition may require stopping the DB for a consistent dump; use offline mode with care.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DumpResult:
    """Result of a dump attempt."""

    ok: bool
    host_path: Path | None
    neo4j_major: int | None
    command_ran: list[str]
    stdout: str = ""
    stderr: str = ""
    message: str = ""


def _which_docker() -> str | None:
    return shutil.which("docker")


def docker_available() -> bool:
    return _which_docker() is not None


def detect_neo4j_major_in_container(container: str) -> int | None:
    """
    Parse `neo4j-admin --version` inside the container.

    Example: "neo4j-admin 4.4.37" -> 4
    """
    docker = _which_docker()
    if not docker:
        return None
    try:
        p = subprocess.run(
            [docker, "exec", container, "neo4j-admin", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (p.stdout or "") + (p.stderr or "")
        m = re.search(r"(\d+)\.(\d+)", text)
        if m:
            return int(m.group(1))
    except (subprocess.TimeoutExpired, OSError, ValueError) as e:
        logger.warning("Could not detect Neo4j version in container: %s", e)
    return None


def build_dump_command(
    neo4j_major: int | None,
    database: str,
    dump_file_in_container: str,
) -> list[str]:
    """
    Build neo4j-admin argv (without 'docker exec' prefix).

    Neo4j 5 uses: neo4j-admin database dump <db> --to-path=<path>
    Neo4j 4 uses: neo4j-admin dump --database=<db> --to=<path>
    """
    if neo4j_major is not None and neo4j_major >= 5:
        return [
            "neo4j-admin",
            "database",
            "dump",
            database,
            f"--to-path={dump_file_in_container}",
        ]
    return ["neo4j-admin", "dump", f"--database={database}", f"--to={dump_file_in_container}"]


def run_dump_inside_container(
    container: str,
    argv_admin: list[str],
    *,
    timeout_sec: int = 7200,
) -> tuple[int, str, str]:
    """Run `docker exec <container> neo4j-admin ...`."""
    docker = _which_docker()
    if not docker:
        raise RuntimeError("docker CLI not found in PATH")
    cmd = [docker, "exec", container, *argv_admin]
    logger.info("Running: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    return p.returncode, p.stdout or "", p.stderr or ""


def docker_cp_from_container(container: str, container_path: str, host_path: Path) -> None:
    """docker cp container:path -> host_path"""
    docker = _which_docker()
    if not docker:
        raise RuntimeError("docker CLI not found")
    host_path.parent.mkdir(parents=True, exist_ok=True)
    src = f"{container}:{container_path}"
    subprocess.run([docker, "cp", src, str(host_path)], check=True, capture_output=True, text=True)


def docker_cp_to_host(container: str, container_path: str, host_path: Path) -> None:
    """Alias for docker_cp_from_container."""
    docker_cp_from_container(container, container_path, host_path)


def neo4j_stop_in_container(container: str, timeout_sec: int = 120) -> tuple[int, str, str]:
    """Run `neo4j stop` inside container (may fail if not supported)."""
    return run_dump_inside_container(container, ["neo4j", "stop"], timeout_sec=timeout_sec)


def neo4j_start_in_container(container: str, timeout_sec: int = 120) -> tuple[int, str, str]:
    """Run `neo4j start` inside container."""
    return run_dump_inside_container(container, ["neo4j", "start"], timeout_sec=timeout_sec)


def _docker_image_for_container(container: str) -> str:
    """``docker inspect --format '{{.Config.Image}}'``，供辅助容器使用相同镜像层内的 neo4j-admin。"""
    docker = _which_docker()
    if not docker:
        raise RuntimeError("docker CLI not found")
    p = subprocess.run(
        [docker, "inspect", "--format", "{{.Config.Image}}", container],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "").strip() or "docker inspect failed")
    img = (p.stdout or "").strip()
    if not img:
        raise RuntimeError("could not resolve container image")
    return img


def _dump_offline_volumes_from_helper(
    *,
    container: str,
    image: str,
    database: str,
    host_output: Path,
    neo4j_major: int | None,
    timeout_sec: int,
) -> tuple[int, str, str, list[str]]:
    """
    宿主机 ``docker stop`` 主库容器后，用 ``docker run --rm --volumes-from`` 挂同一数据卷执行 dump。

    避免在官方 Neo4j 镜像内 ``neo4j stop``：该操作常导致 PID1 退出，整容器停止，后续 ``docker exec`` 全部失败。
    """
    docker = _which_docker()
    if not docker:
        raise RuntimeError("docker CLI not found")
    host_output = host_output.resolve()
    host_output.parent.mkdir(parents=True, exist_ok=True)

    mount_host_dir = str(host_output.parent)
    in_helper = f"/__host_dump_out__/{host_output.name}"
    argv = build_dump_command(neo4j_major, database, in_helper)

    # neo4j 镜像默认 ENTRYPOINT 会改写参数，显式用 neo4j-admin 作 entrypoint 更稳
    cmd = [
        docker,
        "run",
        "--rm",
        "--volumes-from",
        container,
        "-v",
        f"{mount_host_dir}:/__host_dump_out__",
        "--entrypoint",
        "neo4j-admin",
        image,
        *argv[1:],  # strip leading "neo4j-admin"
    ]
    logger.info("Running (offline helper): %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    return p.returncode, p.stdout or "", p.stderr or "", cmd


def docker_start_container(container: str, timeout_sec: int = 120) -> tuple[int, str, str]:
    """``docker start`` on host."""
    docker = _which_docker()
    if not docker:
        raise RuntimeError("docker CLI not found")
    p = subprocess.run(
        [docker, "start", container],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def docker_stop_container(container: str, timeout_sec: int = 300) -> tuple[int, str, str]:
    """``docker stop`` on host."""
    docker = _which_docker()
    if not docker:
        raise RuntimeError("docker CLI not found")
    p = subprocess.run(
        [docker, "stop", container],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def dump_neo4j_via_docker(
    *,
    container: str,
    database: str,
    host_output: Path,
    neo4j_major: int | None = None,
    offline: bool = False,
    container_dump_dir: str = "/tmp",
    timeout_sec: int = 7200,
) -> DumpResult:
    """
    Create an offline dump file on the host using neo4j-admin inside Docker.

    **offline=False**：在**运行中**的容器内 ``docker exec … neo4j-admin dump``（Community 常因库仍在线而失败）。

    **offline=True**（推荐）：宿主机 ``docker stop`` 主库容器 → ``docker run --rm --volumes-from`` 使用**同一数据卷**
    启动一次性容器执行 ``neo4j-admin dump`` → 将文件写到挂载的宿主机目录 → ``docker start`` 恢复原容器。
    不在主库容器内执行 ``neo4j stop``，以免官方镜像主进程退出导致整容器退出、无法继续 exec。

    Args:
        container: Docker container name or id (e.g. kgrdmon).
        database: Logical database name (default neo4j).
        host_output: Destination .dump file on the host.
        neo4j_major: 4 or 5; if None, auto-detect via neo4j-admin --version.
        offline: If True, host-level stop + volumes-from helper dump + start (downtime).
        container_dump_dir: Writable directory inside the container (仅 offline=False 时使用).
    """
    if not docker_available():
        return DumpResult(
            ok=False,
            host_path=None,
            neo4j_major=neo4j_major,
            command_ran=[],
            message="docker CLI not found in PATH",
        )

    major = neo4j_major if neo4j_major is not None else detect_neo4j_major_in_container(container)
    host_output = host_output.resolve()
    host_output.parent.mkdir(parents=True, exist_ok=True)

    if offline:
        host_stopped = False
        cmd_ran: list[str] = []
        try:
            img = _docker_image_for_container(container)
            sc, so, se = docker_stop_container(container, timeout_sec=min(timeout_sec, 600))
            if sc != 0:
                msg = f"docker stop failed (exit {sc}): {se or so}"
                return DumpResult(
                    ok=False,
                    host_path=None,
                    neo4j_major=major,
                    command_ran=["docker", "stop", container],
                    stdout=so,
                    stderr=se,
                    message=msg,
                )
            host_stopped = True

            code, out, err, cmd_ran = _dump_offline_volumes_from_helper(
                container=container,
                image=img,
                database=database,
                host_output=host_output,
                neo4j_major=major,
                timeout_sec=timeout_sec,
            )
            if code != 0:
                msg = f"neo4j-admin dump failed (exit {code}): {err or out}"
                return DumpResult(
                    ok=False,
                    host_path=None,
                    neo4j_major=major,
                    command_ran=cmd_ran,
                    stdout=out,
                    stderr=err,
                    message=msg,
                )
            if not host_output.is_file() or host_output.stat().st_size < 1:
                msg = f"dump command succeeded but missing or empty file: {host_output}"
                return DumpResult(
                    ok=False,
                    host_path=None,
                    neo4j_major=major,
                    command_ran=cmd_ran,
                    stdout=out,
                    stderr=err,
                    message=msg,
                )

            return DumpResult(
                ok=True,
                host_path=host_output,
                neo4j_major=major,
                command_ran=cmd_ran,
                stdout=out,
                stderr=err,
                message="dump completed (offline volumes-from)",
            )
        except subprocess.TimeoutExpired as e:
            return DumpResult(
                ok=False,
                host_path=None,
                neo4j_major=major,
                command_ran=cmd_ran,
                message=f"timeout: {e}",
            )
        except (OSError, RuntimeError) as e:
            return DumpResult(
                ok=False,
                host_path=None,
                neo4j_major=major,
                command_ran=cmd_ran,
                message=str(e),
            )
        finally:
            if host_stopped:
                st, o, e = docker_start_container(container)
                if st != 0:
                    logger.warning("docker start after dump failed (exit %s): %s %s", st, o, e)

    tmp_name = f"neo4j_dump_{uuid.uuid4().hex}.dump"
    in_container_path = f"{container_dump_dir.rstrip('/')}/{tmp_name}"

    argv = build_dump_command(major, database, in_container_path)
    cmd_ran = ["docker", "exec", container, *argv]

    try:
        code, out, err = run_dump_inside_container(container, argv, timeout_sec=timeout_sec)
        if code != 0:
            msg = f"neo4j-admin dump failed (exit {code}): {err or out}"
            return DumpResult(
                ok=False,
                host_path=None,
                neo4j_major=major,
                command_ran=cmd_ran,
                stdout=out,
                stderr=err,
                message=msg,
            )

        docker_cp_from_container(container, in_container_path, host_output)

        docker_bin = _which_docker()
        if docker_bin:
            try:
                subprocess.run(
                    [docker_bin, "exec", container, "rm", "-f", in_container_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except OSError:
                pass

        return DumpResult(
            ok=True,
            host_path=host_output,
            neo4j_major=major,
            command_ran=cmd_ran,
            stdout=out,
            stderr=err,
            message="dump completed",
        )
    except subprocess.CalledProcessError as e:
        return DumpResult(
            ok=False,
            host_path=None,
            neo4j_major=major,
            command_ran=cmd_ran,
            message=str(e),
        )
    except subprocess.TimeoutExpired as e:
        return DumpResult(
            ok=False,
            host_path=None,
            neo4j_major=major,
            command_ran=cmd_ran,
            message=f"timeout: {e}",
        )


def dump_result_to_dict(r: DumpResult) -> dict[str, Any]:
    return {
        "ok": r.ok,
        "host_path": str(r.host_path) if r.host_path else None,
        "neo4j_major": r.neo4j_major,
        "command_ran": r.command_ran,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "message": r.message,
    }
