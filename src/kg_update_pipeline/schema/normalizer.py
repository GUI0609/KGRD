"""节点/边属性合并与列表去重逻辑。"""

from __future__ import annotations

from typing import Any

from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord


def _dedupe_str_list(items: list[str]) -> list[str]:
    """保持顺序的去重。"""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def merge_node_records(existing: NodeRecord, incoming: NodeRecord) -> NodeRecord:
    """
    合并两个 NodeRecord：名称优先非空；synonyms/xrefs 去重合并；
    properties 中新值覆盖旧标量，数组字段合并去重。
    """
    name = incoming.name.strip() if incoming.name and incoming.name.strip() else existing.name
    desc_in = incoming.description
    desc_ex = existing.description
    description = desc_in if (desc_in and str(desc_in).strip()) else desc_ex

    synonyms = _dedupe_str_list(existing.synonyms + incoming.synonyms)
    xrefs = _dedupe_str_list(existing.xrefs + incoming.xrefs)

    merged_props: dict[str, Any] = dict(existing.properties)
    for k, v in incoming.properties.items():
        if k in merged_props:
            old_v = merged_props[k]
            if isinstance(old_v, list) and isinstance(v, list):
                merged_props[k] = _dedupe_str_list([str(x) for x in old_v] + [str(x) for x in v])
            else:
                merged_props[k] = v
        else:
            merged_props[k] = v

    return NodeRecord(
        primary_id=existing.primary_id,
        name=name,
        category=incoming.category or existing.category,
        synonyms=synonyms,
        description=description,
        xrefs=xrefs,
        source=incoming.source or existing.source,
        version=incoming.version or existing.version,
        properties=merged_props,
    )


def merge_nodes_by_id(nodes: list[NodeRecord]) -> list[NodeRecord]:
    """按 primary_id 合并多条 NodeRecord（用于多源写入前聚合）。"""
    m: dict[str, NodeRecord] = {}
    for n in nodes:
        if n.primary_id not in m:
            m[n.primary_id] = n
        else:
            m[n.primary_id] = merge_node_records(m[n.primary_id], n)
    return list(m.values())


def merge_edge_props(
    old_props: dict[str, Any],
    new_props: dict[str, Any],
) -> dict[str, Any]:
    """合并关系属性，列表字段合并去重。"""
    out = dict(old_props)
    for k, v in new_props.items():
        if k in out:
            ov = out[k]
            if isinstance(ov, list) and isinstance(v, list):
                out[k] = _dedupe_str_list([str(x) for x in ov] + [str(x) for x in v])
            else:
                out[k] = v
        else:
            out[k] = v
    return out
