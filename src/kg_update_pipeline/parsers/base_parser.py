"""
Base parser and shared helpers (OBO streaming, TSV iteration).

OBO files are split on blank lines so large ontologies do not load whole-file into RAM.
"""

from __future__ import annotations

import csv
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)

# Category for stub nodes created to anchor cross-references (xref edges).
XrefStubCategoryFn = Callable[[str], str]


class BaseParser(ABC):
    """Abstract parser for one upstream dataset."""

    source_name: str = "base"

    @abstractmethod
    def parse(
        self,
        raw_path: Path,
        *,
        version: str,
        out_parsed_dir: Path | None = None,
    ) -> tuple[list[NodeRecord], list[EdgeRecord]]:
        """
        Parse ``raw_path`` into nodes and edges.

        Args:
            raw_path: Downloaded file for this source.
            version: Run version label (e.g. date).
            out_parsed_dir: If set, may write a small JSONL sample for debugging.
        """

    def _write_jsonl_sample(
        self,
        out_parsed_dir: Path | None,
        stem: str,
        nodes: list[NodeRecord],
        edges: list[EdgeRecord],
        limit: int = 500,
    ) -> None:
        """Write a truncated JSONL sample when ``out_parsed_dir`` is set."""
        if out_parsed_dir is None:
            return
        try:
            import json

            out_parsed_dir.mkdir(parents=True, exist_ok=True)
            p = out_parsed_dir / f"{stem}_sample.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for i, n in enumerate(nodes[:limit]):
                    f.write(
                        json.dumps(
                            {
                                "kind": "node",
                                "primary_id": n.primary_id,
                                "name": n.name,
                                "category": n.category,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                for i, e in enumerate(edges[:limit]):
                    f.write(
                        json.dumps(
                            {
                                "kind": "edge",
                                "source_id": e.source_id,
                                "target_id": e.target_id,
                                "relation": e.relation,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except OSError as e:
            logger.warning("Could not write sample jsonl: %s", e)


def iter_tsv_dict_rows(
    path: Path,
    *,
    delimiter: str = "\t",
    encoding: str = "utf-8",
    max_rows: int | None = None,
) -> Iterator[tuple[int, dict[str, str]]]:
    """
    Stream TSV/CSV rows as dicts (header from first line).

    Yields ``(row_index, row_dict)`` where ``row_index`` is 0-based over data rows.
    Stops after ``max_rows`` data rows when ``max_rows`` is not None.
    """
    with path.open("r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            return
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            yield i, row


def xref_stub_category_external_only(_xref_id: str) -> str:
    """Stub xref targets as ``ExternalResource`` (HPO-style)."""
    return "ExternalResource"


def xref_stub_category_mondo_style(xref_id: str) -> str:
    """Infer category for xref stub nodes (MONDO-style HP:/HGNC: hints)."""
    if xref_id.startswith("HP:"):
        return "Phenotype"
    if xref_id.startswith("HGNC:") or "gene" in xref_id.lower():
        return "Gene"
    return "ExternalResource"


def parse_obo_ontology_terms(
    raw_path: Path,
    *,
    version: str,
    source_name: str,
    id_prefix: str,
    primary_category: str,
    xref_stub_category: XrefStubCategoryFn,
) -> tuple[list[NodeRecord], list[EdgeRecord]]:
    """
    Build nodes and edges from an OBO ``[Term]`` file (HPO, MONDO, etc.).

    Only terms whose ``id`` starts with ``id_prefix`` become primary nodes with
    ``primary_category``. Xref targets get stub nodes using ``xref_stub_category``.
    ``IS_A``, ``xref``, and ``relationship`` lines become edges.
    """
    nodes_by_id: dict[str, NodeRecord] = {}
    edges: list[EdgeRecord] = []
    if not raw_path.is_file():
        logger.warning("OBO file missing: %s", raw_path)
        return [], []

    for stanza in iter_obo_stanzas(raw_path):
        term = parse_obo_term_stanza(stanza)
        if not term:
            continue
        tid = term["id"]
        if not tid.startswith(id_prefix):
            continue
        xrefs = list(term.get("xref") or [])
        props = {
            "obo_namespace": term.get("namespace") or "",
            "alt_ids": term.get("alt_id") or [],
        }
        nodes_by_id[tid] = NodeRecord(
            primary_id=tid,
            name=term["name"],
            category=primary_category,
            synonyms=list(term.get("synonyms") or []),
            description=term.get("def"),
            xrefs=xrefs,
            source=source_name,
            version=version,
            properties=props,
        )
        for parent in term.get("is_a") or []:
            if parent.startswith(id_prefix):
                edges.append(
                    EdgeRecord(
                        source_id=tid,
                        target_id=parent,
                        relation="IS_A",
                        source=source_name,
                        version=version,
                        evidence="obo_is_a",
                    )
                )
        for xr in xrefs:
            edges.append(
                EdgeRecord(
                    source_id=tid,
                    target_id=xr,
                    relation="XREF",
                    source=source_name,
                    version=version,
                    evidence="obo_xref",
                )
            )
            if xr not in nodes_by_id:
                nodes_by_id[xr] = NodeRecord(
                    primary_id=xr,
                    name=xr,
                    category=xref_stub_category(xr),
                    source=source_name,
                    version=version,
                    properties={"kind": "xref_target"},
                )
        for rel_type, tgt in term.get("relationship") or []:
            edges.append(
                EdgeRecord(
                    source_id=tid,
                    target_id=tgt,
                    relation=rel_type.upper(),
                    source=source_name,
                    version=version,
                    evidence="obo_relationship",
                )
            )

    return list(nodes_by_id.values()), edges


def iter_obo_stanzas(path: Path) -> Iterator[list[str]]:
    """
    Split an OBO file into stanza line lists using blank lines as separators.

    Yields:
        Non-empty lines for one stanza.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            buf: list[str] = []
            for line in f:
                line = line.rstrip("\n")
                if line.strip() == "":
                    if buf:
                        yield buf
                        buf = []
                    continue
                buf.append(line)
            if buf:
                yield buf
    except OSError as e:
        logger.error("Failed to read OBO file %s: %s", path, e)
        raise


def parse_obo_term_stanza(
    lines: list[str],
    *,
    default_ns: str = "",
) -> dict[str, Any] | None:
    """
    Parse one ``[Term]`` stanza (not ``[Typedef]``).

    Returns keys: id, name, def, synonyms, is_a, xref, namespace, alt_id, relationship.
    """
    if not lines or not lines[0].startswith("[Term]"):
        return None
    tid: str | None = None
    name = ""
    defn: str | None = None
    synonyms: list[str] = []
    is_a: list[str] = []
    xrefs: list[str] = []
    alt_ids: list[str] = []
    rel_lines: list[tuple[str, str]] = []
    namespace = default_ns
    for line in lines[1:]:
        if line.startswith("id:"):
            tid = line.split(":", 1)[1].strip()
        elif line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("namespace:"):
            namespace = line.split(":", 1)[1].strip()
        elif line.startswith("def:"):
            m = re.match(r'def:\s*"(.*)"', line)
            if m:
                defn = m.group(1)
        elif line.startswith("synonym:"):
            m = re.match(r'synonym:\s*"([^"]+)"', line)
            if m:
                synonyms.append(m.group(1))
        elif line.startswith("is_a:"):
            rest = line.split(":", 1)[1].strip()
            parent = rest.split("!")[0].strip()
            is_a.append(parent)
        elif line.startswith("xref:"):
            rest = line.split(":", 1)[1].strip()
            xid = rest.split()[0] if rest else ""
            if xid:
                xrefs.append(xid)
        elif line.startswith("alt_id:"):
            alt_ids.append(line.split(":", 1)[1].strip())
        elif line.startswith("relationship:"):
            parts = line.split(None, 2)
            if len(parts) >= 3:
                rel_type = parts[1].strip()
                tgt = parts[2].split("!")[0].strip()
                rel_lines.append((rel_type, tgt))
    if not tid:
        return None
    return {
        "id": tid,
        "name": name or tid,
        "def": defn,
        "synonyms": synonyms,
        "is_a": is_a,
        "xref": xrefs,
        "namespace": namespace,
        "alt_id": alt_ids,
        "relationship": rel_lines,
    }
