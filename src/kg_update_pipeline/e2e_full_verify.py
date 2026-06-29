#!/usr/bin/env python3
"""
一次性全流程验收（严格模式：任一步失败则退出码非 0）。

真实执行：HTTP 下载烟测、run_kg_update 子进程管线（可选连 Neo4j）。

默认：在版本目录写入最小 `hp.obo` 并 `--skip-download`，数分钟内完成。
加 --full-download 会从 PURL 拉取完整 hp.obo（体积大、耗时长）。

Neo4j：
  --strict-neo4j：仅 Bolt 握手 + RETURN 1，密码错误 -> exit 1
  --pipeline-neo4j-dry-run：管线内连库 dry-run MERGE（需环境变量中的密码写入临时 yaml）

示例：
  cd KGRD
  python src/kg_update_pipeline/e2e_full_verify.py

  export KG_NEO4J_URI='bolt://127.0.0.1:7687'
  export KG_NEO4J_PASSWORD='***'
  python src/kg_update_pipeline/e2e_full_verify.py --strict-neo4j --pipeline-neo4j-dry-run --connect-neo4j
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd or _ROOT, env=env or os.environ.copy())
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def step_http_download() -> None:
    from kg_update_pipeline.downloader import download_http_file

    d = Path(tempfile.mkdtemp(prefix="kg_e2e_http_"))
    try:
        dest = d / "smoke.bin"
        download_http_file("https://httpbin.org/bytes/2048", dest, timeout=60, retries=2)
        if dest.stat().st_size != 2048:
            raise SystemExit("HTTP smoke: unexpected file size")
        print("HTTP download OK (2048 bytes)", flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _write_minimal_hpo(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "hp.obo").write_text(
        """[Term]
id: HP:0099998
name: E2E full verify stub
namespace: human_phenotype
""",
        encoding="utf-8",
    )


def _write_temp_config(
    data_root: Path,
    state_root: Path,
    log_root: Path,
    backup_root: Path,
    cfg_out: Path,
) -> None:
    doc: dict = {
        "project_root": str(_ROOT),
        "data_root": str(data_root),
        "log_root": str(log_root),
        "backup_root": str(backup_root),
        "state_root": str(state_root),
        "neo4j": {
            "uri": os.environ.get("KG_NEO4J_URI", "bolt://127.0.0.1:7687"),
            "user": os.environ.get("KG_NEO4J_USER", "neo4j"),
            "password": os.environ.get("KG_NEO4J_PASSWORD", "neo4j"),
            "database": os.environ.get("KG_NEO4J_DATABASE", "neo4j"),
        },
        "update": {
            "mode": "full",
            "dry_run": False,
            "backup_before_update": False,
        },
        "sources": {
            "hpo": {
                "enabled": True,
                "type": "http",
                "url": "https://purl.obolibrary.org/obo/hp.obo",
                "filename": "hp.obo",
            },
            "mondo": {
                "enabled": False,
                "type": "http",
                "url": "https://example.com/x.obo",
                "filename": "mondo.obo",
            },
        },
    }
    cfg_out.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")


def step_pipeline(
    *,
    full_download: bool,
    skip_neo4j: bool,
    neo4j_dry_run: bool,
) -> None:
    from kg_update_pipeline.config_loader import load_config
    from kg_update_pipeline.version_manager import version_dir_for_date

    work = Path(tempfile.mkdtemp(prefix="kg_e2e_pipeline_"))
    cfg_path = work / "e2e_config.yaml"
    try:
        data_root = work / "kg_data"
        state_root = work / "state"
        log_root = work / "logs"
        backup_root = work / "backups"
        _write_temp_config(data_root, state_root, log_root, backup_root, cfg_path)

        if not full_download:
            c = load_config(cfg_path)
            vdir = version_dir_for_date(c)
            _write_minimal_hpo(vdir / "raw")

        cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "run_kg_update.py"),
            "--config",
            str(cfg_path),
            "--source",
            "hpo",
        ]
        if not full_download:
            cmd.append("--skip-download")
        if skip_neo4j:
            cmd.append("--skip-neo4j")
        if neo4j_dry_run:
            cmd.append("--dry-run")

        _run(cmd, cwd=_ROOT)

        c2 = load_config(cfg_path)
        mpath = version_dir_for_date(c2) / "manifest.json"
        if not mpath.is_file():
            raise SystemExit(f"manifest missing: {mpath}")
        print(f"Pipeline OK, manifest: {mpath}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def step_strict_neo4j() -> None:
    uri = os.environ.get("KG_NEO4J_URI", "").strip()
    password = os.environ.get("KG_NEO4J_PASSWORD", "").strip()
    user = os.environ.get("KG_NEO4J_USER", "neo4j").strip()
    if not uri or not password:
        raise SystemExit("--strict-neo4j requires KG_NEO4J_URI and KG_NEO4J_PASSWORD")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        db = os.environ.get("KG_NEO4J_DATABASE", "neo4j").strip()
        with driver.session(database=db) as session:
            n = session.run("RETURN 1 AS n").single()["n"]
            if n != 1:
                raise SystemExit("RETURN 1 mismatch")
    finally:
        driver.close()
    print("Strict Neo4j OK", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="KG pipeline strict full verification (non-mock)")
    ap.add_argument("--skip-http", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument(
        "--full-download",
        action="store_true",
        help="Download full hp.obo from PURL (large/slow). Default is quick minimal OBO.",
    )
    ap.add_argument(
        "--connect-neo4j",
        action="store_true",
        help="Pipeline step connects to Neo4j (omit --skip-neo4j on run_kg_update)",
    )
    ap.add_argument(
        "--pipeline-neo4j-dry-run",
        action="store_true",
        help="With --connect-neo4j: MERGE in dry-run only",
    )
    ap.add_argument("--strict-neo4j", action="store_true", help="Bolt + RETURN 1; auth fail -> exit 1")
    args = ap.parse_args()

    skip_neo4j = not args.connect_neo4j
    neo4j_dry = bool(args.pipeline_neo4j_dry_run)

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    if not args.skip_http:
        print("=== [1] HTTP download (real network) ===", flush=True)
        step_http_download()
    if not args.skip_pipeline:
        print("=== [2] run_kg_update pipeline ===", flush=True)
        step_pipeline(
            full_download=args.full_download,
            skip_neo4j=skip_neo4j,
            neo4j_dry_run=neo4j_dry,
        )
    if args.strict_neo4j:
        print("=== [3] strict Neo4j (fail on bad password) ===", flush=True)
        step_strict_neo4j()

    print("=== ALL STEPS PASSED (strict e2e) ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
