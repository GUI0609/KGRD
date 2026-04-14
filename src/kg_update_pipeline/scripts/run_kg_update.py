#!/usr/bin/env python3
"""
KG 自动更新管线 CLI。仓库根目录 = 本包目录的父目录（sys.path 须含该目录以导入 ``kg_update_pipeline``）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../rd-project/kg_update_pipeline/scripts/this.py -> 父级父级 = rd-project
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kg_update_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
