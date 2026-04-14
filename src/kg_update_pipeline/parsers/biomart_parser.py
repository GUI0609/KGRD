"""
BioMart / generic TSV export placeholder.

Map real export columns in a dedicated parser or extend this once columns are fixed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kg_update_pipeline.parsers.base_parser import BaseParser, iter_tsv_dict_rows
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)


class BiomartParser(BaseParser):
    source_name = "biomart"
    max_rows: int | None = 20_000

    def parse(
        self,
        raw_path: Path,
        *,
        version: str,
        out_parsed_dir: Path | None = None,
    ) -> tuple[list[NodeRecord], list[EdgeRecord]]:
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            logger.warning("BioMart: no export at %s — skipped.", raw_path)
            return [], []
        nodes: list[NodeRecord] = []
        try:
            for i, row in iter_tsv_dict_rows(
                raw_path, delimiter="\t", max_rows=self.max_rows
            ):
                pid = f"BIOMART_ROW:{i}"
                nodes.append(
                    NodeRecord(
                        primary_id=pid,
                        name=pid,
                        category="UnknownEntity",
                        source=self.source_name,
                        version=version,
                        properties=dict(row),
                    )
                )
        except OSError as e:
            logger.warning("BioMart placeholder parse failed: %s", e)
        self._write_jsonl_sample(out_parsed_dir, "biomart", nodes, [])
        return nodes, []
