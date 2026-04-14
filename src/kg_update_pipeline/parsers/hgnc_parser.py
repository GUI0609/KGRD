"""HGNC: gene nodes from ``hgnc_complete_set.txt`` (tab-separated)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from kg_update_pipeline.parsers.base_parser import BaseParser
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)


class HgncParser(BaseParser):
    source_name = "hgnc"

    def parse(
        self,
        raw_path: Path,
        *,
        version: str,
        out_parsed_dir: Path | None = None,
    ) -> tuple[list[NodeRecord], list[EdgeRecord]]:
        nodes: list[NodeRecord] = []
        edges: list[EdgeRecord] = []
        if not raw_path.is_file():
            logger.warning("HGNC raw file missing: %s", raw_path)
            return [], []

        try:
            with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip() and not line.lstrip().startswith("#"):
                        f.seek(pos)
                        break
                reader = csv.DictReader(f, delimiter="\t")
                if not reader.fieldnames:
                    logger.warning("HGNC: no header row in %s", raw_path)
                    return [], []

                fields_lower = {h.lower(): h for h in reader.fieldnames if h}
                hgnc_col = fields_lower.get("hgnc_id") or fields_lower.get("hgnc id")
                sym_col = fields_lower.get("symbol")
                name_col = fields_lower.get("name")

                for row in reader:
                    hgnc_raw = row.get(hgnc_col) if hgnc_col else None
                    if not hgnc_raw or not str(hgnc_raw).strip():
                        continue
                    hid = str(hgnc_raw).strip()
                    primary = f"HGNC:{hid}" if not hid.upper().startswith("HGNC:") else hid
                    sym = (row.get(sym_col) if sym_col else None) or ""
                    nm = (row.get(name_col) if name_col else None) or sym or primary
                    alias_col = fields_lower.get("alias_symbol")
                    prev_col = fields_lower.get("prev_symbol")
                    syns: list[str] = []
                    if alias_col and row.get(alias_col):
                        syns.extend(str(row[alias_col]).split("|"))
                    if prev_col and row.get(prev_col):
                        syns.extend(str(row[prev_col]).split("|"))
                    syns = [s.strip() for s in syns if s and str(s).strip()]

                    props = {k: v for k, v in row.items() if v and str(v).strip()}
                    nodes.append(
                        NodeRecord(
                            primary_id=primary,
                            name=str(nm).strip(),
                            category="Gene",
                            synonyms=syns,
                            description=None,
                            xrefs=[primary],
                            source=self.source_name,
                            version=version,
                            properties=props,
                        )
                    )
        except OSError as e:
            logger.error("HGNC parse failed: %s", e)
            raise

        self._write_jsonl_sample(out_parsed_dir, "hgnc", nodes, edges)
        return nodes, edges
