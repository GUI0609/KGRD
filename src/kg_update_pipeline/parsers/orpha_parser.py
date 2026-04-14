"""
Orphanet XML placeholder.

TODO: parse Product6/Product4 XML for Orpha codes and HPO links.
For now: if the file exists, record one stub node from the root tag (non-fatal).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from kg_update_pipeline.parsers.base_parser import BaseParser
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)


class OrphaParser(BaseParser):
    source_name = "orpha"

    def parse(
        self,
        raw_path: Path,
        *,
        version: str,
        out_parsed_dir: Path | None = None,
    ) -> tuple[list[NodeRecord], list[EdgeRecord]]:
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            logger.warning("Orphanet: no XML at %s — skipped.", raw_path)
            return [], []
        try:
            tree = ET.parse(raw_path)
            root = tree.getroot()
            tag = root.tag
            nodes = [
                NodeRecord(
                    primary_id="ORPHA:XML_ROOT",
                    name=tag,
                    category="ExternalResource",
                    source=self.source_name,
                    version=version,
                    properties={"root_tag": tag, "note": "TODO full Orphanet XML parse"},
                )
            ]
            self._write_jsonl_sample(out_parsed_dir, "orpha", nodes, [])
            return nodes, []
        except ET.ParseError as e:
            logger.warning("Orphanet XML parse error (non-fatal): %s", e)
            return [], []
