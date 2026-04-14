"""运行日志：控制台 + 文件；仅配置 `kg_update_pipeline` 命名空间，避免污染全局 root。"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


def setup_run_logger(
    name: str,
    log_file: Path | None = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    为包 `kg_update_pipeline` 配置 handler，子模块 logger 默认向上传播至此。
    """
    base = logging.getLogger("kg_update_pipeline")
    base.handlers.clear()
    base.setLevel(logging.DEBUG if verbose else logging.INFO)
    base.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 与版本/备份目录命名一致，使用 UTC，避免本地时区与 YYYY-MM-DD 分区混用造成「两个时间」
    fmt.converter = time.gmtime

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    base.addHandler(ch)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            base.addHandler(fh)
        except OSError as e:
            logging.getLogger(name).warning("Could not attach file handler to %s: %s", log_file, e)

    return logging.getLogger(name)
