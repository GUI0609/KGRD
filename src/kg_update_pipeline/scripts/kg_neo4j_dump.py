#!/usr/bin/env python3
"""
Neo4j 离线 dump（Docker 内 neo4j-admin）。

在仓库根目录执行示例::
  python kg_update_pipeline/scripts/kg_neo4j_dump.py --container NAME \\
    -o kg_update_pipeline/backups/kg/manual.neo4j.dump --offline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kg_update_pipeline.neo4j_db.docker_dump import (
    detect_neo4j_major_in_container,
    docker_available,
    dump_neo4j_via_docker,
    dump_result_to_dict,
)

logger = logging.getLogger("kg_neo4j_dump")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Dump Neo4j database from a Docker container using neo4j-admin.",
    )
    p.add_argument(
        "--container",
        default=os.environ.get("KG_NEO4J_DOCKER_CONTAINER", "kgrdmon"),
        help="Docker container name or id (default: env KG_NEO4J_DOCKER_CONTAINER or kgrdmon)",
    )
    p.add_argument("--database", default="neo4j", help="Logical database name")
    p.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Host path for the .dump file",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Stop Neo4j inside the container before dump, then start again (downtime).",
    )
    p.add_argument(
        "--neo4j-major",
        type=int,
        choices=(4, 5),
        default=None,
        help="Force Neo4j major version for neo4j-admin syntax (default: auto-detect)",
    )
    p.add_argument(
        "--container-tmp",
        default="/tmp",
        help="Writable directory inside the container for temporary dump file",
    )
    p.add_argument("--timeout", type=int, default=7200, help="Timeout seconds for dump")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not docker_available():
        logger.error("docker CLI not found in PATH")
        return 2

    out: Path = args.output.expanduser().resolve()
    if out.suffix.lower() != ".dump":
        logger.warning("Output filename usually ends with .dump; continuing anyway.")

    major = args.neo4j_major
    if major is None:
        major = detect_neo4j_major_in_container(args.container)
        logger.info("Detected Neo4j major version in container: %s", major)

    result = dump_neo4j_via_docker(
        container=args.container,
        database=args.database,
        host_output=out,
        neo4j_major=major,
        offline=args.offline,
        container_dump_dir=args.container_tmp,
        timeout_sec=args.timeout,
    )

    meta_path = out.with_name(out.stem + ".meta.json")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "container": args.container,
        "database": args.database,
        "offline": args.offline,
        "result": dump_result_to_dict(result),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote metadata: %s", meta_path)

    if result.ok:
        logger.info("Dump OK -> %s", result.host_path)
        return 0

    logger.error("%s", result.message)
    if not args.offline and "online" in (result.stderr + result.message).lower():
        logger.error("Hint: try again with --offline (stops DB briefly for consistent dump on Community).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
