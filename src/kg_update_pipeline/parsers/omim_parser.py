"""
OMIM placeholder.

OMIM usually needs an API key / license; do not fake download URLs.
TODO: implement with official API or licensed export (gene–phenotype–disease).
"""

from __future__ import annotations

import logging
from pathlib import Path

from kg_update_pipeline.parsers.base_parser import BaseParser
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)


class OmimParser(BaseParser):
    source_name = "omim"

    def parse(
        self,
        raw_path: Path,
        *,
        version: str,
        out_parsed_dir: Path | None = None,
    ) -> tuple[list[NodeRecord], list[EdgeRecord]]:
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            logger.warning(
                "OMIM: no local file at %s — skipped. TODO: provide API key / licensed export.",
                raw_path,
            )
            return [], []
        logger.info("OMIM: placeholder parser sees non-empty file; TODO implement full parse.")
        return [], []
