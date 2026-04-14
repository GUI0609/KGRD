"""
NCBI / manual TSV placeholder.

Replace with real Gene/OMIM column mapping when the export shape is known.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kg_update_pipeline.parsers.base_parser import BaseParser, iter_tsv_dict_rows
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)


class NcbiParser(BaseParser):
    source_name = "ncbi"
    max_rows: int | None = 10_000

    def parse(
        self,
        raw_path: Path,
        *,
        version: str,
        out_parsed_dir: Path | None = None,
    ) -> tuple[list[NodeRecord], list[EdgeRecord]]:
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            logger.warning("NCBI: no data at %s — skipped. TODO: set FTP/manual path.", raw_path)
            return [], []
        nodes: list[NodeRecord] = []
        try:
            for i, row in iter_tsv_dict_rows(
                raw_path, delimiter="\t", max_rows=self.max_rows
            ):
                vals = list(row.values())
                pid = f"NCBI_ROW:{i}"
                first = str(vals[0]).strip() if vals else ""
                nodes.append(
                    NodeRecord(
                        primary_id=pid,
                        name=(first[:500] if first else pid),
                        category="UnknownEntity",
                        source=self.source_name,
                        version=version,
                        properties=dict(row),
                    )
                )
        except OSError as e:
            logger.warning("NCBI placeholder parse failed: %s", e)
        self._write_jsonl_sample(out_parsed_dir, "ncbi", nodes, [])
        return nodes, []
