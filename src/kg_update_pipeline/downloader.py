"""
多数据源 HTTP 下载：超时、重试、流式写入、SHA256。

支持 http、http_tar_gz（下载 .tar.gz 并解压指定成员到 raw/filename）。

无法下载的数据源记录 warning 并跳过，不中断整体流程。
"""

from __future__ import annotations

import logging
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from kg_update_pipeline.config_loader import LoadedConfig, SourceEntry
from kg_update_pipeline.utils.hashing import file_sha256
from kg_update_pipeline.utils.io_utils import ensure_dir, read_json, write_json
from kg_update_pipeline.utils.time_utils import iso_now

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _tqdm_enabled(show_progress: bool) -> bool:
    """是否在下载/解压时显示 tqdm（输出到 stderr，避免与日志抢占 stdout）。"""
    return bool(show_progress and tqdm)


def _tqdm_bar(**kwargs: Any) -> Any:
    """
    创建 tqdm 进度条。

    显式 ``disable=False``：在 Cursor / CI 等 **stderr 非 TTY** 环境下，tqdm 默认会关闭进度条；
    关闭后改为周期性打印进度行，仍可见下载进度。
    """
    if not tqdm:
        return None  # pragma: no cover
    defaults: dict[str, Any] = {
        "file": sys.stderr,
        "mininterval": 0.5,
        "disable": False,
        "dynamic_ncols": True,
        "ascii": False,
    }
    defaults.update(kwargs)
    return tqdm(**defaults)


def _file_stamp(path: Path | None) -> tuple[str | None, int | None]:
    """记录本地文件时间戳与大小（用于 manifest）。"""
    if path is None or not path.is_file():
        return None, None
    try:
        return iso_now(), path.stat().st_size
    except OSError:
        return iso_now(), None


class DownloadStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    UNCHANGED = "unchanged"


@dataclass
class DownloadResult:
    source: str
    status: DownloadStatus
    path: Path | None = None
    sha256: str | None = None
    message: str = ""
    downloaded_at: str | None = None
    size_bytes: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def make_download_result(
    *,
    source: str,
    status: DownloadStatus,
    path: Path | None = None,
    sha256: str | None = None,
    message: str = "",
) -> DownloadResult:
    """构建 DownloadResult 并填充 downloaded_at / size_bytes（若存在本地文件）。"""
    da, sz = _file_stamp(path)
    return DownloadResult(
        source=source,
        status=status,
        path=path,
        sha256=sha256,
        message=message,
        downloaded_at=da,
        size_bytes=sz,
    )


def _load_previous_hashes(state_root: Path) -> dict[str, str]:
    p = state_root / "source_hashes.json"
    data = read_json(p, default={})
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def _save_source_hash(state_root: Path, source: str, digest: str) -> None:
    p = state_root / "source_hashes.json"
    data = read_json(p, default={})
    if not isinstance(data, dict):
        data = {}
    data[source] = digest
    write_json(p, data)


def _short_url(u: str, max_len: int = 96) -> str:
    u = u.strip()
    return u if len(u) <= max_len else u[: max_len - 3] + "..."


def _effective_timeout_seconds(
    entry: SourceEntry,
    *,
    global_timeout: int,
    tar_gz_floor: int,
    is_tar_gz: bool,
) -> int:
    """每源 timeout_seconds > 0 优先；否则 http 用全局；tar.gz 与 floor 取较大值。"""
    if entry.timeout_seconds > 0:
        return entry.timeout_seconds
    if is_tar_gz:
        return max(global_timeout, tar_gz_floor)
    return global_timeout


def _http_url_list(entry: SourceEntry) -> list[str]:
    """主 url + fallback_urls，去重保序。"""
    acc: list[str] = []
    seen: set[str] = set()
    for u in [entry.url, *entry.fallback_urls]:
        s = (u or "").strip()
        if s and s not in seen:
            seen.add(s)
            acc.append(s)
    return acc


# 部分镜像对默认 urllib UA 会中途断连；使用常见浏览器式 UA 可提高稳定性
_HTTP_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; kg-update-pipeline/1.0; +https://obofoundry.org/)",
    "Accept": "*/*",
}


def download_http_try_urls(
    urls: list[str],
    dest: Path,
    *,
    timeout: int,
    retries: int,
    chunk_size: int = 1024 * 256,
    label: str | None = None,
    show_progress: bool = True,
    source_key: str = "",
) -> None:
    """
    按顺序尝试多个 URL，任意一个完整下载成功即返回。
    用于 purl 302/断连时切换到 GitHub Releases 等备用地址。
    """
    log_prefix = f"[{source_key}] " if source_key else ""
    ordered = [u.strip() for u in urls if u and u.strip()]
    if not ordered:
        raise ValueError(f"{log_prefix}no HTTP URLs to try")
    last_err: Exception | None = None
    for idx, url in enumerate(ordered):
        try:
            download_http_file(
                url,
                dest,
                timeout=timeout,
                retries=retries,
                chunk_size=chunk_size,
                label=label,
                show_progress=show_progress,
                source_key=source_key,
            )
            if idx > 0:
                logger.info(
                    "%sDownload OK using alternate URL (%s/%s): %s",
                    log_prefix,
                    idx + 1,
                    len(ordered),
                    _short_url(url),
                )
            return
        except Exception as e:
            last_err = e
            if idx < len(ordered) - 1:
                logger.warning(
                    "%sURL %s/%s failed, trying next: %s | %s",
                    log_prefix,
                    idx + 1,
                    len(ordered),
                    _short_url(url),
                    e,
                )
    if last_err:
        raise last_err


def download_http_file(
    url: str,
    dest: Path,
    *,
    timeout: int = 120,
    retries: int = 3,
    chunk_size: int = 1024 * 256,
    label: str | None = None,
    show_progress: bool = True,
    # 数据源键（如 hpo、monarch），写入日志便于定位是哪条 URL 失败
    source_key: str = "",
) -> None:
    """
    流式下载到 dest；可选 tqdm 进度条。

    ``timeout`` 为 urllib 套接字阻塞上限（秒）：既包含建立连接/SSL，也包含
    ``read`` 等待下一数据块的间隔。大文件或国际链路若两次收到数据间隔超过该值，
    仍会触发 ``timed out``，与「能否 ping 通」无直接关系。
    """
    ensure_dir(dest.parent)
    last_err: Exception | None = None
    desc = (label or dest.name)[:48]
    log_prefix = f"[{source_key}] " if source_key else ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=dict(_HTTP_REQUEST_HEADERS))
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                cl = resp.headers.get("Content-Length")
                total: int | None = int(cl) if cl and str(cl).isdigit() else None
                use_bar = _tqdm_enabled(show_progress)
                with dest.open("wb") as out:
                    pbar: Any = None
                    if use_bar:
                        pbar = _tqdm_bar(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            desc=desc,
                            leave=True,
                        )
                    try:
                        nread = 0
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            out.write(chunk)
                            nread += len(chunk)
                            if pbar is not None:
                                pbar.update(len(chunk))
                        if pbar is not None and total is None and nread > 0:
                            pbar.total = nread
                            pbar.refresh()
                    finally:
                        if pbar is not None:
                            pbar.close()
                try:
                    sz = dest.stat().st_size
                except OSError:
                    sz = nread
                logger.info(
                    "%sDownload finished: %s bytes | url=%s | used_attempts=%s/%s",
                    log_prefix,
                    sz,
                    _short_url(url),
                    attempt,
                    retries,
                )
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < retries:
                logger.info(
                    "%sDownload attempt %s/%s failed (will retry) | url=%s | error=%s",
                    log_prefix,
                    attempt,
                    retries,
                    _short_url(url),
                    e,
                )
            else:
                logger.warning(
                    "%sDownload attempt %s/%s failed | url=%s | dest=%s | error=%s",
                    log_prefix,
                    attempt,
                    retries,
                    _short_url(url),
                    dest,
                    e,
                )
            time.sleep(min(2**attempt, 30))
    if last_err:
        raise last_err


def _resolve_tar_member(member_names: list[str], archive_member: str, source_name: str) -> str:
    am = (archive_member or "").strip()
    if am:
        for n in member_names:
            if n == am or n.endswith("/" + am) or n.endswith("/" + am.lstrip("/")):
                return n
        raise ValueError(f"[{source_name}] archive_member={am!r} not found in archive")

    for pat in ("monarch-kg_nodes.tsv",):
        for n in member_names:
            if n.endswith(pat) or pat in n:
                return n
    nodes = [n for n in member_names if n.endswith("_nodes.tsv")]
    if nodes:
        return nodes[0]
    raise ValueError(
        f"[{source_name}] cannot auto-pick nodes TSV; set sources.{source_name}.archive_member"
    )


def _download_tar_gz_extract(
    name: str,
    entry: SourceEntry,
    raw_dir: Path,
    *,
    timeout: int = 3600,
    retries: int = 3,
    show_progress: bool = True,
) -> Path:
    """下载 .tar.gz，解压成员到 raw_dir / entry.filename。"""
    ensure_dir(raw_dir)
    archive_path = raw_dir / f"{name}_archive.tar.gz"
    urls = _http_url_list(entry)
    if not urls:
        raise ValueError("empty url")
    download_http_try_urls(
        urls,
        archive_path,
        timeout=timeout,
        retries=retries,
        label=f"{name}:tar.gz",
        show_progress=show_progress,
        source_key=name,
    )
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            names = [m.name for m in tf.getmembers() if m.isfile()]
            member = _resolve_tar_member(names, entry.archive_member, name)
            ef = tf.extractfile(member)
            if ef is None:
                raise OSError(f"cannot read archive member {member}")
            dest = raw_dir / entry.filename
            ensure_dir(dest.parent)
            with dest.open("wb") as out:
                chunk_size = 1024 * 1024
                total = getattr(ef, "size", None) or None
                pbar: Any = None
                if _tqdm_enabled(show_progress):
                    pbar = _tqdm_bar(
                        total=total if total and total > 0 else None,
                        unit="B",
                        unit_scale=True,
                        desc=f"{name}:extract",
                        leave=True,
                    )
                try:
                    while True:
                        chunk = ef.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)
                        if pbar is not None:
                            pbar.update(len(chunk))
                finally:
                    if pbar is not None:
                        pbar.close()
                    ef.close()
    finally:
        if entry.remove_archive_after_extract and archive_path.is_file():
            try:
                archive_path.unlink()
            except OSError as e:
                logger.warning("[%s] could not remove archive %s: %s", name, archive_path, e)

    return raw_dir / entry.filename


def download_source(
    name: str,
    entry: SourceEntry,
    raw_dir: Path,
    state_root: Path,
    *,
    incremental: bool = True,
    show_progress: bool = True,
    http_timeout_seconds: int = 600,
    http_retries: int = 3,
    tar_gz_floor_timeout_seconds: int = 3600,
) -> DownloadResult:
    """
    下载单个数据源。manual_or_* 类型若无 URL 则 skipped。

    http_tar_gz：下载归档后解压到 filename。
    """
    dest = raw_dir / entry.filename
    entry_type = entry.type.lower().strip()

    if not entry.enabled:
        return make_download_result(source=name, status=DownloadStatus.SKIPPED, message="source disabled in config")

    if entry_type in ("manual_or_api", "manual_or_ftp", "manual_export", "manual_or_http"):
        if not (entry.url or "").strip():
            logger.warning(
                "[%s] No URL configured (%s). Place file manually at %s. TODO: configure credentials if needed.",
                name,
                entry_type,
                dest,
            )
            if dest.is_file() and dest.stat().st_size > 0:
                digest = file_sha256(dest)
                _save_source_hash(state_root, name, digest)
                return make_download_result(
                    source=name,
                    status=DownloadStatus.SUCCESS,
                    path=dest,
                    sha256=digest,
                    message="using existing manual file",
                )
            return make_download_result(
                source=name,
                status=DownloadStatus.SKIPPED,
                message=f"manual_or_* mode with empty url and no file at {dest}",
            )

    if entry_type == "http" and not _http_url_list(entry):
        logger.warning("[%s] Empty URL — skipped. %s", name, entry.note or "")
        return make_download_result(source=name, status=DownloadStatus.SKIPPED, message="empty url")

    if entry_type == "http_tar_gz":
        if not _http_url_list(entry):
            logger.warning("[%s] http_tar_gz requires url — skipped", name)
            return make_download_result(source=name, status=DownloadStatus.SKIPPED, message="empty url")
        try:
            t_tar = _effective_timeout_seconds(
                entry,
                global_timeout=http_timeout_seconds,
                tar_gz_floor=tar_gz_floor_timeout_seconds,
                is_tar_gz=True,
            )
            out_path = _download_tar_gz_extract(
                name,
                entry,
                raw_dir,
                timeout=t_tar,
                retries=http_retries,
                show_progress=show_progress,
            )
        except Exception as e:
            logger.exception("[%s] tar.gz download/extract failed: %s", name, e)
            return make_download_result(source=name, status=DownloadStatus.FAILED, message=str(e))
        digest = file_sha256(out_path)
        prev = _load_previous_hashes(state_root)
        if incremental and name in prev and prev[name] == digest:
            return make_download_result(
                source=name,
                status=DownloadStatus.UNCHANGED,
                path=out_path,
                sha256=digest,
                message="extracted content identical to previous run",
            )
        _save_source_hash(state_root, name, digest)
        r = make_download_result(
            source=name,
            status=DownloadStatus.SUCCESS,
            path=out_path,
            sha256=digest,
            message="downloaded tar.gz and extracted",
        )
        r.meta = {"mode": "http_tar_gz", "url": entry.url}
        return r

    if entry_type != "http":
        logger.warning("[%s] Unsupported download type '%s' — skipped", name, entry_type)
        return make_download_result(
            source=name,
            status=DownloadStatus.SKIPPED,
            message=f"unsupported type {entry_type}",
        )

    try:
        t_http = _effective_timeout_seconds(
            entry,
            global_timeout=http_timeout_seconds,
            tar_gz_floor=tar_gz_floor_timeout_seconds,
            is_tar_gz=False,
        )
        download_http_try_urls(
            _http_url_list(entry),
            dest,
            label=f"{name}:{entry.filename}",
            show_progress=show_progress,
            source_key=name,
            timeout=t_http,
            retries=http_retries,
        )
    except Exception as e:
        logger.exception("[%s] Download failed: %s", name, e)
        return make_download_result(source=name, status=DownloadStatus.FAILED, message=str(e))

    digest = file_sha256(dest)
    prev = _load_previous_hashes(state_root)
    if incremental and name in prev and prev[name] == digest:
        return make_download_result(
            source=name,
            status=DownloadStatus.UNCHANGED,
            path=dest,
            sha256=digest,
            message="content identical to previous run",
        )

    _save_source_hash(state_root, name, digest)
    return make_download_result(
        source=name,
        status=DownloadStatus.SUCCESS,
        path=dest,
        sha256=digest,
        message="downloaded",
    )


def run_downloads(
    cfg: LoadedConfig,
    raw_dir: Path,
    selected_sources: list[str] | None = None,
) -> dict[str, DownloadResult]:
    results: dict[str, DownloadResult] = {}
    incremental = cfg.update.mode == "incremental"

    for name, entry in cfg.sources.items():
        if selected_sources is not None and name not in selected_sources:
            continue
        if not entry.enabled:
            results[name] = make_download_result(source=name, status=DownloadStatus.SKIPPED, message="disabled")
            continue
        results[name] = download_source(
            name,
            entry,
            raw_dir,
            cfg.state_root,
            incremental=incremental,
            show_progress=True,
            http_timeout_seconds=cfg.update.http_timeout_seconds,
            http_retries=cfg.update.http_retries,
            tar_gz_floor_timeout_seconds=cfg.update.tar_gz_floor_timeout_seconds,
        )

    return results
