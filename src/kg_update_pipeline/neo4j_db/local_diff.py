"""
Compare parsed records against a JSONL snapshot and shrink MERGE inputs to real deltas.

When ``refresh_existing_nodes`` is false, only nodes whose ``entity_id`` is absent from the
snapshot are sent to Neo4j (no per-row MERGE match for unchanged bulk).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kg_update_pipeline.neo4j_db.introspect import normalize_rel_type_name
from kg_update_pipeline.neo4j_db.payload_fingerprint import fingerprint_from_export_line, node_payload_fingerprint
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord
from kg_update_pipeline.schema.normalizer import merge_nodes_by_id

logger = logging.getLogger(__name__)


def load_kg_category_map(nodes_jsonl: Path) -> dict[str, str]:
    """entity_id -> kg_category from snapshot (for edge endpoint labels not in this parse)."""
    out: dict[str, str] = {}
    with nodes_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = obj.get("entity_id")
            if not eid:
                continue
            cat = obj.get("kg_category")
            if cat:
                out[str(eid)] = str(cat)
    return out


def load_entity_id_set(nodes_jsonl: Path) -> set[str]:
    """All ``entity_id`` values in the node snapshot (lighter than full fp map)."""
    s: set[str] = set()
    with nodes_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = obj.get("entity_id")
            if eid:
                s.add(str(eid))
    return s


def load_entity_fp_map(nodes_jsonl: Path) -> dict[str, str]:
    """entity_id -> fingerprint string from snapshot (for refresh detection)."""
    out: dict[str, str] = {}
    with nodes_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = obj.get("entity_id")
            if not eid:
                continue
            fp = fingerprint_from_export_line(obj)
            if fp:
                out[str(eid)] = fp
    return out


def load_edge_key_set(edges_jsonl: Path) -> set[tuple[str, str, str]]:
    """Set of (source_id, target_id, normalized_rel_type)."""
    keys: set[tuple[str, str, str]] = set()
    with edges_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(obj.get("sid") or "")
            tid = str(obj.get("tid") or "")
            rt = normalize_rel_type_name(str(obj.get("rel_type") or ""))
            if sid and tid:
                keys.add((sid, tid, rt))
    return keys


def plan_node_delta(
    parsed_nodes: list[NodeRecord],
    *,
    nodes_jsonl: Path,
    refresh_existing_nodes: bool,
) -> tuple[list[NodeRecord], dict[str, Any]]:
    """
    Return nodes that actually need a MERGE write (creates, and updates if refresh).

    Stats include skipped counts for audit.
    """
    nodes = merge_nodes_by_id(parsed_nodes)
    stats: dict[str, Any] = {
        "parsed_after_dedup": len(nodes),
    }
    if not refresh_existing_nodes:
        existing_ids = load_entity_id_set(nodes_jsonl)
        stats["snapshot_nodes"] = len(existing_ids)
        out = [n for n in nodes if n.primary_id not in existing_ids]
        stats["nodes_skipped_already_in_db"] = len(nodes) - len(out)
        stats["nodes_planned_merge"] = len(out)
        stats["nodes_planned_creates"] = len(out)
        stats["nodes_planned_updates"] = 0
        return out, stats

    fp_map = load_entity_fp_map(nodes_jsonl)
    stats["snapshot_nodes"] = len(fp_map)
    creates: list[NodeRecord] = []
    updates: list[NodeRecord] = []
    unchanged = 0
    for n in nodes:
        eid = n.primary_id
        new_fp = node_payload_fingerprint(n)
        old_fp = fp_map.get(eid)
        if old_fp is None:
            creates.append(n)
        elif old_fp != new_fp:
            updates.append(n)
        else:
            unchanged += 1
    out = creates + updates
    stats["nodes_skipped_unchanged_fingerprint"] = unchanged
    stats["nodes_planned_creates"] = len(creates)
    stats["nodes_planned_updates"] = len(updates)
    stats["nodes_planned_merge"] = len(out)
    return out, stats


def plan_edge_delta(
    parsed_edges: list[EdgeRecord],
    *,
    edges_jsonl: Path,
) -> tuple[list[EdgeRecord], dict[str, Any]]:
    """Edges whose (sid, tid, rel) triple is not already in the snapshot."""
    existing = load_edge_key_set(edges_jsonl)
    snap_size = len(existing)
    out: list[EdgeRecord] = []
    dup = 0
    seen_new: set[tuple[str, str, str]] = set()
    for e in parsed_edges:
        k = (e.source_id, e.target_id, normalize_rel_type_name(e.relation or "RELATED"))
        if k in existing:
            dup += 1
            continue
        if k in seen_new:
            continue
        seen_new.add(k)
        out.append(e)
    stats = {
        "parsed_edges": len(parsed_edges),
        "snapshot_edge_keys": snap_size,
        "edges_skipped_already_in_db": dup,
        "edges_planned_merge": len(out),
    }
    return out, stats
