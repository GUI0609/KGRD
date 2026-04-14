"""
将解析结果写入 parsed/ 目录：JSONL（节点/边）与 CSV 摘要。

大文件时按 max_rows 截断并写入 warning 行，避免占满磁盘。
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)


def export_source_parsed(
    parsed_dir: Path,
    source_name: str,
    nodes: list[NodeRecord],
    edges: list[EdgeRecord],
    *,
    max_rows: int = 500_000,
    write_csv_summary: bool = True,
) -> dict[str, str | int]:
    """
    写入 {source}_nodes.jsonl、{source}_edges.jsonl、{source}_summary.csv。

    Returns:
        写入路径与截断行数等元信息
    """
    parsed_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, str | int] = {
        "nodes_written": 0,
        "edges_written": 0,
        "nodes_truncated": 0,
        "edges_truncated": 0,
    }

    n_cap = min(len(nodes), max_rows)
    e_cap = min(len(edges), max_rows)
    meta["nodes_truncated"] = max(0, len(nodes) - n_cap)
    meta["edges_truncated"] = max(0, len(edges) - e_cap)

    np = parsed_dir / f"{source_name}_nodes.jsonl"
    ep = parsed_dir / f"{source_name}_edges.jsonl"
    try:
        with np.open("w", encoding="utf-8") as fn:
            for n in nodes[:n_cap]:
                rec = asdict(n)
                fn.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta["nodes_written"] = n_cap
        with ep.open("w", encoding="utf-8") as fe:
            for e in edges[:e_cap]:
                rec = asdict(e)
                fe.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta["edges_written"] = e_cap
        meta["nodes_jsonl"] = str(np)
        meta["edges_jsonl"] = str(ep)

        if meta["nodes_truncated"] or meta["edges_truncated"]:
            logger.warning(
                "[%s] parsed export truncated to max_rows=%s (nodes dropped %s, edges dropped %s)",
                source_name,
                max_rows,
                meta["nodes_truncated"],
                meta["edges_truncated"],
            )

        if write_csv_summary:
            sp = parsed_dir / f"{source_name}_summary.csv"
            with sp.open("w", encoding="utf-8", newline="") as fc:
                w = csv.writer(fc)
                w.writerow(["metric", "value"])
                w.writerow(["source", source_name])
                w.writerow(["node_count", len(nodes)])
                w.writerow(["edge_count", len(edges)])
                w.writerow(["nodes_exported", n_cap])
                w.writerow(["edges_exported", e_cap])
            meta["summary_csv"] = str(sp)
    except OSError as e:
        logger.error("parsed export failed for %s: %s", source_name, e)
        raise
    return meta
