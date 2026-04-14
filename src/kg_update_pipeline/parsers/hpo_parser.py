"""HPO: phenotype nodes and OBO edges from ``hp.obo``."""

from __future__ import annotations

from pathlib import Path

from kg_update_pipeline.parsers.base_parser import (
    BaseParser,
    parse_obo_ontology_terms,
    xref_stub_category_external_only,
)
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord


class HpoParser(BaseParser):
    source_name = "hpo"

    def parse(
        self,
        raw_path: Path,
        *,
        version: str,
        out_parsed_dir: Path | None = None,
    ) -> tuple[list[NodeRecord], list[EdgeRecord]]:
        nodes, edges = parse_obo_ontology_terms(
            raw_path,
            version=version,
            source_name=self.source_name,
            id_prefix="HP:",
            primary_category="Phenotype",
            xref_stub_category=xref_stub_category_external_only,
        )
        self._write_jsonl_sample(out_parsed_dir, "hpo", nodes, edges)
        return nodes, edges
