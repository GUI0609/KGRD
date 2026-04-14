"""Stable SHA-256 fingerprint for node MERGE payloads (export vs parse comparison)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kg_update_pipeline.schema.models import NodeRecord


def _blob_from_fields(
    name: str,
    synonyms: list[str],
    xrefs: list[str],
    description: Any,
    kg_category: str,
    source: str,
    version: str,
    kg_properties_json: str,
) -> str:
    return json.dumps(
        {
            "name": name or "",
            "synonyms": synonyms,
            "xrefs": xrefs,
            "description": description,
            "kg_category": kg_category or "",
            "source": source or "",
            "version": version or "",
            "kg_properties_json": kg_properties_json or "{}",
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def fingerprint_from_flat_fields(
    name: str,
    synonyms: list[str],
    xrefs: list[str],
    description: Any,
    kg_category: str,
    source: str,
    version: str,
    kg_properties_json: str,
) -> str:
    """Fingerprint from DB row / export dict (same canonicalization as ``node_payload_fingerprint``)."""
    return hashlib.sha256(
        _blob_from_fields(
            name,
            synonyms,
            xrefs,
            description,
            kg_category,
            source,
            version,
            kg_properties_json,
        ).encode("utf-8")
    ).hexdigest()


def normalize_kg_properties_json(s: str | None) -> str:
    """Canonical JSON string for ``kg_properties_json`` (sorted object keys)."""
    if not s or not str(s).strip():
        return "{}"
    try:
        o = json.loads(s)
        if isinstance(o, dict):
            return json.dumps(o, ensure_ascii=False, sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        pass
    return str(s)


def node_payload_fingerprint(n: NodeRecord) -> str:
    """Fingerprint for a parsed ``NodeRecord`` (MERGE row without timestamps)."""
    syn = list(dict.fromkeys(n.synonyms or []))
    xr = list(dict.fromkeys(n.xrefs or []))
    props = dict(n.properties or {})
    kgj = json.dumps(props, ensure_ascii=False, sort_keys=True)
    return fingerprint_from_flat_fields(
        n.name or "",
        syn,
        xr,
        n.description,
        n.category,
        n.source or "",
        n.version or "",
        kgj,
    )


def fingerprint_from_export_line(obj: dict[str, Any]) -> str | None:
    """Read stored ``fp`` or recompute from a snapshot JSON line."""
    if obj.get("fp"):
        return str(obj["fp"])
    try:
        return fingerprint_from_flat_fields(
            str(obj.get("name") or ""),
            list(obj.get("synonyms") or []),
            list(obj.get("xrefs") or []),
            obj.get("description"),
            str(obj.get("kg_category") or ""),
            str(obj.get("source") or ""),
            str(obj.get("version") or ""),
            normalize_kg_properties_json(str(obj.get("kg_properties_json") or "{}")),
        )
    except (TypeError, ValueError):
        return None
