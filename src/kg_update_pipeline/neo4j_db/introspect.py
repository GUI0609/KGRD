"""
从 Neo4j 读取已有 schema（标签、关系类型），用于限制 MERGE 不引入新类型。

使用 CALL db.labels() / db.relationshipTypes()，兼容 Neo4j 4.x / 5.x community。
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kg_update_pipeline.neo4j_db.client import Neo4jClient

logger = logging.getLogger(__name__)


def normalize_rel_type_name(name: str) -> str:
    """
    与 updater.merge_edges 中关系名规范化逻辑一致，便于与库中已有类型对齐。
    """
    rel_raw = (name or "RELATED").upper().replace(" ", "_").replace("-", "_")
    return re.sub(r"[^A-Za-z0-9_]", "_", rel_raw) or "RELATED"


def fetch_labels(client: "Neo4jClient") -> list[str]:
    """返回当前库中所有节点标签（字符串与 Cypher 中一致）。"""
    rows = client.run_read(
        "CALL db.labels() YIELD label RETURN label AS t ORDER BY label"
    )
    out = [str(r["t"]) for r in rows if r.get("t") is not None]
    logger.info("Neo4j introspect: loaded %s node labels", len(out))
    return out


def fetch_relationship_types(client: "Neo4jClient") -> list[str]:
    """返回当前库中所有关系类型名称。"""
    rows = client.run_read(
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t ORDER BY relationshipType"
    )
    out = [str(r["t"]) for r in rows if r.get("t") is not None]
    logger.info("Neo4j introspect: loaded %s relationship types", len(out))
    return out


def build_rel_type_canonical_map(db_types: list[str]) -> dict[str, str]:
    """
    将规范化键映射到库中实际使用的关系类型字符串（用于 MERGE 中动态类型名）。

    若多个库类型规范化后相同，优先保留首个；通常库内不会重复。
    """
    m: dict[str, str] = {}
    for t in db_types:
        key = normalize_rel_type_name(t)
        if key not in m:
            m[key] = t
    return m


def cypher_escape_rel_type(rel_type: str) -> str:
    """关系类型写入 Cypher 时的转义（含反引号）。"""
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rel_type):
        return rel_type
    return "`" + rel_type.replace("`", "``") + "`"
