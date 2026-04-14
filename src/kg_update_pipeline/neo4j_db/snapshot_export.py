"""
Stream Neo4j graph state to JSONL (nodes + edges) for local diff before bulk MERGE.

Not neo4j-admin; uses Bolt reads so the database stays online.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kg_update_pipeline.neo4j_db.client import Neo4jClient
from kg_update_pipeline.neo4j_db.payload_fingerprint import (
    fingerprint_from_flat_fields,
    normalize_kg_properties_json,
)
from kg_update_pipeline.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

# Nodes: one JSON object per line (entity_id + MERGE-relevant props + fp).
_NODES_CYPHER = """
MATCH (n) WHERE n.entity_id IS NOT NULL
RETURN n.entity_id AS entity_id,
       labels(n) AS labels,
       n.kg_category AS kg_category,
       n.name AS name,
       n.synonyms AS synonyms,
       n.xrefs AS xrefs,
       n.description AS description,
       n.source AS source,
       n.version AS version,
       coalesce(n.kg_properties_json, "{}") AS kg_properties_json
"""

# Relationships between nodes that expose entity_id on both ends.
_EDGES_CYPHER = """
MATCH (a)-[r]->(b)
WHERE a.entity_id IS NOT NULL AND b.entity_id IS NOT NULL
RETURN a.entity_id AS sid,
       b.entity_id AS tid,
       type(r) AS rel_type,
       a.kg_category AS sc,
       b.kg_category AS tc,
       r.source AS rsource,
       r.version AS rversion,
       r.evidence AS evidence
"""


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def export_nodes_jsonl(client: Neo4jClient, dest: Path) -> int:
    """Write all nodes with ``entity_id`` to JSONL; returns line count."""
    ensure_dir(dest.parent)
    n = 0
    with dest.open("w", encoding="utf-8") as f:
        for record in client.iter_run(_NODES_CYPHER):
            d = record.data()
            eid = d.get("entity_id")
            if eid is None:
                continue
            syn = _as_list(d.get("synonyms"))
            xr = _as_list(d.get("xrefs"))
            kgj = normalize_kg_properties_json(str(d.get("kg_properties_json") or "{}"))
            fp = fingerprint_from_flat_fields(
                str(d.get("name") or ""),
                syn,
                xr,
                d.get("description"),
                str(d.get("kg_category") or ""),
                str(d.get("source") or ""),
                str(d.get("version") or ""),
                kgj,
            )
            line = {
                "entity_id": str(eid),
                "labels": d.get("labels") or [],
                "kg_category": d.get("kg_category"),
                "name": d.get("name"),
                "synonyms": syn,
                "xrefs": xr,
                "description": d.get("description"),
                "source": d.get("source"),
                "version": d.get("version"),
                "kg_properties_json": kgj,
                "fp": fp,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1
            if n % 100_000 == 0:
                logger.info("snapshot export: %s node(s) written…", n)
    logger.info("snapshot export: wrote %s node(s) → %s", n, dest)
    return n


def export_edges_jsonl(client: Neo4jClient, dest: Path) -> int:
    """Write relationship rows to JSONL; returns line count."""
    ensure_dir(dest.parent)
    n = 0
    with dest.open("w", encoding="utf-8") as f:
        for record in client.iter_run(_EDGES_CYPHER):
            d = record.data()
            line = {
                "sid": str(d.get("sid")),
                "tid": str(d.get("tid")),
                "rel_type": str(d.get("rel_type") or ""),
                "sc": d.get("sc"),
                "tc": d.get("tc"),
                "source": d.get("rsource"),
                "version": d.get("rversion"),
                "evidence": d.get("evidence"),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1
            if n % 200_000 == 0:
                logger.info("snapshot export: %s edge(s) written…", n)
    logger.info("snapshot export: wrote %s edge(s) → %s", n, dest)
    return n


def export_graph_snapshot(
    client: Neo4jClient,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Export current DB to ``out_dir/nodes.jsonl`` and ``out_dir/edges.jsonl``.

    Returns counts and paths.
    """
    ensure_dir(out_dir)
    np = out_dir / "nodes.jsonl"
    ep = out_dir / "edges.jsonl"
    nc = export_nodes_jsonl(client, np)
    ec = export_edges_jsonl(client, ep)
    return {
        "nodes_path": str(np),
        "edges_path": str(ep),
        "nodes_exported": nc,
        "edges_exported": ec,
    }
