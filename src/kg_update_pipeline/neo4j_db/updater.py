"""
Batch MERGE nodes and relationships into Neo4j (``entity_id`` + configurable labels).

No DELETE / DETACH DELETE here; destructive Cypher must stay gated elsewhere.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from kg_update_pipeline.config_loader import ConfigError, Neo4jConfig, build_category_label_map
from kg_update_pipeline.neo4j_db.client import Neo4jClient
from kg_update_pipeline.neo4j_db.introspect import (
    build_rel_type_canonical_map,
    cypher_escape_rel_type,
    fetch_labels,
    fetch_relationship_types,
    normalize_rel_type_name,
)
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord
from kg_update_pipeline.schema.normalizer import merge_nodes_by_id
from kg_update_pipeline.utils.io_utils import append_jsonl
from kg_update_pipeline.utils.time_utils import iso_now

logger = logging.getLogger(__name__)


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _intish(v: Any) -> int:
    try:
        return int(v)  # Neo4j Integer
    except (TypeError, ValueError):
        return 0


def _batch_progress_step(n_batches: int) -> int:
    """Log interval for batch progress (avoid log spam)."""
    return max(1, (n_batches + 19) // 20)


def _additional_label_tokens(labels: list[str]) -> str:
    r"""Build `label` or backtick-escaped tokens for ``SET n:...``."""
    parts: list[str] = []
    for raw in labels:
        s = (raw or "").strip()
        if not s:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
            parts.append(s)
        else:
            parts.append("`" + s.replace("`", "``") + "`")
    return ":".join(parts)


class GraphUpdater:
    """MERGE writer for nodes and relationships."""

    def __init__(self, client: Neo4jClient, neo_cfg: Neo4jConfig) -> None:
        self.client = client
        self.neo_cfg = neo_cfg
        self.batch_size = neo_cfg.merge_batch_size
        self._category_to_label = build_category_label_map(neo_cfg)
        logger.info("parser category → Neo4j label: %s", self._category_to_label)

    def _label_for_category(self, category: str) -> str:
        if category in self._category_to_label:
            return self._category_to_label[category]
        return self._category_to_label.get("UnknownEntity", "UnknownEntity")

    def merge_nodes(
        self,
        nodes: list[NodeRecord],
        *,
        dry_run: bool = False,
        nodes_delta_jsonl: Path | None = None,
    ) -> dict[str, Any]:
        """
        MERGE nodes keyed by ``entity_id`` (= ``primary_id``).

        Each batch uses one ``row.ts`` (ISO with microseconds); ``n.created_at = row.ts`` marks
        nodes created in that batch. Label allow-list: ``restrict_to_existing_node_labels``.
        """
        logger.info("merge_nodes: deduplicating %s node record(s) by primary_id…", len(nodes))
        nodes = merge_nodes_by_id(nodes)
        logger.info(
            "merge_nodes: %s node(s) after dedup (batch_size=%s); Neo4j writes may take a long time without indexes on entity_id.",
            len(nodes),
            self.batch_size,
        )
        if self.neo_cfg.refresh_existing_nodes:
            logger.info("neo4j.refresh_existing_nodes=true: updating props on every MERGE")
        merged = 0
        created_sum = 0
        updated_sum = 0
        skipped_existing_sum = 0
        failed = 0
        skipped_unknown_label = 0
        allowed_labels: set[str] | None = None
        db_labels: set[str] | None = None
        if (
            self.neo_cfg.restrict_to_existing_node_labels
            or self.neo_cfg.validate_category_label_targets
        ):
            try:
                db_labels = set(fetch_labels(self.client))
            except Exception as e:
                logger.error("Failed to read Neo4j node labels: %s", e)
                raise
            if self.neo_cfg.validate_category_label_targets:
                bad = [(c, l) for c, l in self._category_to_label.items() if l not in db_labels]
                if bad:
                    preview = "; ".join(f"{c!r}→{l!r}" for c, l in bad[:12])
                    raise ConfigError(
                        "neo4j.validate_category_label_targets: these mapped labels are missing from "
                        f"db.labels(): {preview}"
                        + (f" ({len(bad)} total)" if len(bad) > 12 else "")
                    )
                for add_lbl in self.neo_cfg.additional_node_labels:
                    if add_lbl.strip() and add_lbl not in db_labels:
                        raise ConfigError(
                            f"neo4j.additional_node_labels includes {add_lbl!r}, which is not in db.labels()"
                        )
        if self.neo_cfg.restrict_to_existing_node_labels:
            allowed_labels = db_labels

        by_label: dict[str, list[NodeRecord]] = {}
        for n in nodes:
            lbl = self._label_for_category(n.category)
            by_label.setdefault(lbl, []).append(n)

        logger.info(
            "merge_nodes: %s label bucket(s): %s",
            len(by_label),
            ", ".join(f"`{k}`={len(v)}" for k, v in sorted(by_label.items(), key=lambda x: (-len(x[1]), x[0]))[:12])
            + ("…" if len(by_label) > 12 else ""),
        )

        for lbl, group in by_label.items():
            if allowed_labels is not None and lbl not in allowed_labels:
                logger.warning(
                    "Skipping %s nodes: label `%s` not in db.labels() (restrict_to_existing_node_labels=true)",
                    len(group),
                    lbl,
                )
                skipped_unknown_label += len(group)
                continue
            n_batches = (len(group) + self.batch_size - 1) // self.batch_size
            step = _batch_progress_step(n_batches)
            logger.info(
                "merge_nodes: label `%s` — %s node(s), %s batch(es)",
                lbl,
                len(group),
                n_batches,
            )
            for bi, batch in enumerate(_chunks(group, self.batch_size), start=1):
                ts = iso_now()
                rows = []
                for n in batch:
                    try:
                        syn = list(dict.fromkeys(n.synonyms or []))
                        xr = list(dict.fromkeys(n.xrefs or []))
                        props = dict(n.properties or {})
                        rows.append(
                            {
                                "entity_id": n.primary_id,
                                "name": n.name,
                                "synonyms": syn,
                                "xrefs": xr,
                                "description": n.description,
                                "kg_category": n.category,
                                "source": n.source,
                                "version": n.version,
                                "kg_properties_json": json.dumps(props, ensure_ascii=False),
                                "ts": ts,
                            }
                        )
                    except Exception as e:
                        logger.warning("Skip malformed node %s: %s", n.primary_id, e)
                        failed += 1

                if not rows:
                    continue

                if dry_run:
                    logger.info(
                        "[dry-run] Would MERGE %s nodes with label `%s` (sample entity_id=%s)",
                        len(rows),
                        lbl,
                        rows[0]["entity_id"],
                    )
                    merged += len(rows)
                    continue

                if bi == 1 or bi == n_batches or bi % step == 0:
                    logger.info(
                        "merge_nodes: label `%s` batch %s/%s (%s rows) → executing MERGE…",
                        lbl,
                        bi,
                        n_batches,
                        len(rows),
                    )

                add_tok = _additional_label_tokens(self.neo_cfg.additional_node_labels)
                add_suffix = f", n:{add_tok}" if add_tok else ""
                add_set = f"SET n:{add_tok}\n" if add_tok else ""

                if self.neo_cfg.refresh_existing_nodes:
                    node_write = f"""
                ON CREATE SET n.created_at = row.ts
                SET n.last_seen_at = row.ts,
                    n.updated_at = row.ts,
                    n.name = coalesce(row.name, n.name),
                    n.synonyms = row.synonyms,
                    n.xrefs = row.xrefs,
                    n.description = coalesce(row.description, n.description),
                    n.kg_category = row.kg_category,
                    n.source = row.source,
                    n.version = row.version,
                    n.kg_properties_json = coalesce(row.kg_properties_json, n.kg_properties_json){add_suffix}
                """
                else:
                    node_write = f"""
                ON CREATE SET n.created_at = row.ts,
                    n.last_seen_at = row.ts,
                    n.updated_at = row.ts,
                    n.name = row.name,
                    n.synonyms = row.synonyms,
                    n.xrefs = row.xrefs,
                    n.description = row.description,
                    n.kg_category = row.kg_category,
                    n.source = row.source,
                    n.version = row.version,
                    n.kg_properties_json = row.kg_properties_json
                """
                    if add_set:
                        node_write += add_set
                cypher = f"""
                UNWIND $rows AS row
                MERGE (n:`{lbl}` {{entity_id: row.entity_id}}){node_write}
                WITH n, row
                RETURN
                    sum(CASE WHEN n.created_at = row.ts THEN 1 ELSE 0 END) AS created,
                    count(*) AS total,
                    collect(CASE WHEN n.created_at = row.ts THEN row.entity_id END) AS new_ids
                """

                def work(tx: Any) -> dict[str, Any]:
                    result = tx.run(cypher, rows=rows)
                    rec = result.single()
                    return dict(rec) if rec else {"created": 0, "total": 0, "new_ids": []}

                try:
                    rstat = self.client.write(work)
                    c = _intish(rstat.get("created"))
                    tot = _intish(rstat.get("total"))
                    u = max(0, tot - c)
                    created_sum += c
                    if self.neo_cfg.refresh_existing_nodes:
                        updated_sum += u
                    else:
                        skipped_existing_sum += u
                    merged += tot
                    if nodes_delta_jsonl and c > 0:
                        for eid in rstat.get("new_ids") or []:
                            if eid is not None:
                                append_jsonl(
                                    nodes_delta_jsonl,
                                    {"label": lbl, "entity_id": eid},
                                )
                except Exception as e:
                    logger.error("Node merge failed for label %s: %s", lbl, e)
                    failed += len(rows)

        out: dict[str, Any] = {
            "nodes_merged": merged,
            "nodes_created": 0 if dry_run else created_sum,
            "nodes_updated": 0 if dry_run else updated_sum,
            "nodes_skipped_existing_no_write": 0 if dry_run else skipped_existing_sum,
            "refresh_existing_nodes": self.neo_cfg.refresh_existing_nodes,
            "failed_records": failed,
            "nodes_skipped_unknown_label": skipped_unknown_label,
        }
        if allowed_labels is not None:
            out["existing_node_labels_loaded"] = len(allowed_labels)
        return out

    def merge_edges(
        self,
        edges: list[EdgeRecord],
        id_to_category: dict[str, str],
        *,
        dry_run: bool = False,
        rels_delta_jsonl: Path | None = None,
    ) -> dict[str, Any]:
        """
        MERGE relationships on (source_id, target_id, relation); relation names normalized.

        When ``restrict_to_existing_relationship_types`` is true, unknown types are skipped.
        """
        logger.info("merge_edges: classifying %s edge record(s)…", len(edges))
        if self.neo_cfg.refresh_existing_relationships:
            logger.info(
                "neo4j.refresh_existing_relationships=true: updating props on every MERGE"
            )
        merged = 0
        created_sum = 0
        updated_sum = 0
        skipped_existing_rels = 0
        failed = 0
        skipped_unknown_rel_type = 0

        rel_norm_to_canonical: dict[str, str] | None = None
        if self.neo_cfg.restrict_to_existing_relationship_types:
            try:
                db_rels = fetch_relationship_types(self.client)
                rel_norm_to_canonical = build_rel_type_canonical_map(db_rels)
                logger.info(
                    "restrict_to_existing_relationship_types=true: allowing %s relationship types "
                    "(%s normalized keys)",
                    len(db_rels),
                    len(rel_norm_to_canonical),
                )
            except Exception as e:
                logger.error("Failed to read Neo4j relationship types: %s", e)
                raise

        buckets: dict[tuple[str, str, str], list[EdgeRecord]] = {}
        for e in edges:
            rel_safe = normalize_rel_type_name(e.relation or "RELATED")
            if rel_norm_to_canonical is not None:
                if rel_safe not in rel_norm_to_canonical:
                    skipped_unknown_rel_type += 1
                    continue
                rel_merge = rel_norm_to_canonical[rel_safe]
            else:
                rel_merge = rel_safe
            sc = id_to_category.get(e.source_id, "UnknownEntity")
            tc = id_to_category.get(e.target_id, "UnknownEntity")
            sl = self._label_for_category(sc)
            tl = self._label_for_category(tc)
            key = (sl, tl, rel_merge)
            buckets.setdefault(key, []).append(e)

        if skipped_unknown_rel_type:
            logger.warning(
                "Skipped %s edges: relationship type not in DB (restrict_to_existing_relationship_types=true)",
                skipped_unknown_rel_type,
            )

        kept_edges = sum(len(g) for g in buckets.values())
        logger.info(
            "merge_edges: %s edge(s) in %s bucket(s) (batch_size=%s); MATCH+MERGE can be slow without entity_id indexes.",
            kept_edges,
            len(buckets),
            self.batch_size,
        )

        for (sl, tl, rel_safe), group in buckets.items():
            n_batches = (len(group) + self.batch_size - 1) // self.batch_size
            step = _batch_progress_step(n_batches)
            logger.info(
                "merge_edges: (%s)-[:%s]->(%s) — %s edge(s), %s batch(es)",
                sl,
                rel_safe,
                tl,
                len(group),
                n_batches,
            )
            for bi, batch in enumerate(_chunks(group, self.batch_size), start=1):
                ts = iso_now()
                rows = []
                for e in batch:
                    rows.append(
                        {
                            "sid": e.source_id,
                            "tid": e.target_id,
                            "source": e.source,
                            "version": e.version,
                            "evidence": e.evidence,
                            "props": dict(e.properties or {}),
                            "ts": ts,
                        }
                    )

                if not rows:
                    continue

                rel_esc = cypher_escape_rel_type(rel_safe)
                if dry_run:
                    logger.info(
                        "[dry-run] Would MERGE %s rels :%s (%s)-[:%s]->(%s)",
                        len(rows),
                        rel_safe,
                        sl,
                        rel_safe,
                        tl,
                    )
                    merged += len(rows)
                    continue

                if bi == 1 or bi == n_batches or bi % step == 0:
                    logger.info(
                        "merge_edges: (%s)-[:%s]->(%s) batch %s/%s (%s rows) → executing…",
                        sl,
                        rel_safe,
                        tl,
                        bi,
                        n_batches,
                        len(rows),
                    )

                if self.neo_cfg.refresh_existing_relationships:
                    rel_write = """
                ON CREATE SET r.created_at = row.ts
                SET r.updated_at = row.ts,
                    r.last_seen_at = row.ts,
                    r.source = row.source,
                    r.version = row.version,
                    r.evidence = coalesce(row.evidence, r.evidence)
                """
                else:
                    rel_write = """
                ON CREATE SET r.created_at = row.ts,
                    r.updated_at = row.ts,
                    r.last_seen_at = row.ts,
                    r.source = row.source,
                    r.version = row.version,
                    r.evidence = row.evidence
                """
                cypher = f"""
                UNWIND $rows AS row
                MATCH (a:`{sl}` {{entity_id: row.sid}})
                MATCH (b:`{tl}` {{entity_id: row.tid}})
                MERGE (a)-[r:{rel_esc}]->(b){rel_write}
                WITH r, row
                RETURN
                    sum(CASE WHEN r.created_at = row.ts THEN 1 ELSE 0 END) AS created,
                    count(*) AS total,
                    collect(
                        CASE WHEN r.created_at = row.ts THEN {{sid: row.sid, tid: row.tid}} ELSE null END
                    ) AS new_pairs
                """

                def work(tx: Any) -> dict[str, Any]:
                    result = tx.run(cypher, rows=rows)
                    rec = result.single()
                    return dict(rec) if rec else {"created": 0, "total": 0, "new_pairs": []}

                try:
                    rstat = self.client.write(work)
                    c = _intish(rstat.get("created"))
                    tot = _intish(rstat.get("total"))
                    u = max(0, tot - c)
                    created_sum += c
                    if self.neo_cfg.refresh_existing_relationships:
                        updated_sum += u
                    else:
                        skipped_existing_rels += u
                    merged += tot
                    if rels_delta_jsonl and c > 0:
                        for pair in rstat.get("new_pairs") or []:
                            if not pair:
                                continue
                            append_jsonl(
                                rels_delta_jsonl,
                                {
                                    "relationship_type": rel_safe,
                                    "source_id": pair.get("sid"),
                                    "target_id": pair.get("tid"),
                                },
                            )
                except Exception as e:
                    logger.error("Edge merge failed for rel %s: %s", rel_safe, e)
                    failed += len(rows)

        out: dict[str, Any] = {
            "relationships_merged": merged,
            "relationships_created": 0 if dry_run else created_sum,
            "relationships_updated": 0 if dry_run else updated_sum,
            "relationships_skipped_existing_no_write": 0 if dry_run else skipped_existing_rels,
            "refresh_existing_relationships": self.neo_cfg.refresh_existing_relationships,
            "failed_records": failed,
            "relationships_skipped_unknown_type": skipped_unknown_rel_type,
        }
        if rel_norm_to_canonical is not None:
            out["existing_relationship_types_loaded"] = len(rel_norm_to_canonical)
        return out


def assert_no_destructive_cypher(update_cfg: Any, cypher: str) -> None:
    """
    Block DELETE-style Cypher unless ``update.allow_destructive_ops`` is true.

    Args:
        update_cfg: Object with ``allow_destructive_ops`` (e.g. ``UpdateConfig``).
        cypher: Statement text (matched uppercased).
    """
    u = cypher.upper()
    if "DETACH DELETE" in u or "DELETE " in u:
        if not getattr(update_cfg, "allow_destructive_ops", False):
            raise RuntimeError(
                "Destructive Cypher blocked; set update.allow_destructive_ops=true only for intentional rebuilds."
            )
