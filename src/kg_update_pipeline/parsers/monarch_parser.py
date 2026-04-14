"""
Monarch KG: ``monarch-kg_nodes.tsv`` (tab-separated id/category/name/...).

Unpack ``monarch-kg.tar.gz`` and point ``raw_path`` at the nodes TSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kg_update_pipeline.parsers.base_parser import BaseParser, iter_tsv_dict_rows
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)

# Parser output categories; map to Neo4j labels via ``neo4j.category_label_map``.
_BIOLINK_TO_CATEGORY: dict[str, str] = {
    "biolink:Disease": "Disease",
    "biolink:PhenotypicFeature": "Phenotype",
    "biolink:Gene": "Gene",
    "biolink:GeneOrGeneProduct": "Gene",
    "biolink:Pathway": "Pathway",
    "biolink:BiologicalProcess": "BiologicalProcess",
    "biolink:OntologyClass": "NamedThing",
    "biolink:CellularComponent": "CellularComponent",
    "biolink:MolecularActivity": "MolecularFunction",
    "biolink:ChemicalEntity": "ChemicalEntity",
    "biolink:Cell": "Cell",
    "biolink:AnatomicalEntity": "AnatomicalEntity",
    "biolink:Genotype": "Genotype",
    "biolink:OrganismTaxon": "OrganismTaxon",
    "biolink:LifeStage": "LifeStage",
    "biolink:SequenceVariant": "SequenceVariant",
    "biolink:MolecularEntity": "MolecularEntity",
    "biolink:NamedThing": "NamedThing",
}


def _category_from_row(row: dict[str, str]) -> str:
    raw = (row.get("category") or "").strip()
    if raw in _BIOLINK_TO_CATEGORY:
        return _BIOLINK_TO_CATEGORY[raw]
    low = raw.lower()
    if "disease" in low:
        return "Disease"
    if "phenotyp" in low or "trait" in low:
        return "Phenotype"
    if "pathway" in low:
        return "Pathway"
    if "biological_process" in low or "biological process" in low:
        return "BiologicalProcess"
    if "cellular_component" in low or "cellular component" in low:
        return "CellularComponent"
    if "molecular_function" in low or "molecular function" in low:
        return "MolecularFunction"
    if "gene" in low:
        return "Gene"
    if "chemical" in low or "drug" in low or "substance" in low:
        return "ChemicalEntity"
    if "cell" in low and "cellular" not in low:
        return "Cell"
    if "anatomical" in low or "anatomy" in low:
        return "AnatomicalEntity"
    if "genotype" in low:
        return "Genotype"
    if "taxon" in low or "organism" in low:
        return "OrganismTaxon"
    if "life stage" in low or "lifestage" in low:
        return "LifeStage"
    if "variant" in low or "sequence" in low:
        return "SequenceVariant"
    return "UnknownEntity"


class MonarchParser(BaseParser):
    source_name = "monarch"
    # Cap rows for huge dumps; set to None to read the full file.
    max_rows: int | None = 50_000

    def parse(
        self,
        raw_path: Path,
        *,
        version: str,
        out_parsed_dir: Path | None = None,
    ) -> tuple[list[NodeRecord], list[EdgeRecord]]:
        nodes: list[NodeRecord] = []
        edges: list[EdgeRecord] = []
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            logger.warning("Monarch: no data file at %s — skipped.", raw_path)
            return [], []

        skip_cols = frozenset({"id", "name", "category"})
        try:
            for i, row in iter_tsv_dict_rows(
                raw_path, delimiter="\t", max_rows=self.max_rows
            ):
                pid = (row.get("id") or "").strip()
                if not pid:
                    rid = "|".join(f"{k}={v}" for k, v in row.items() if v)[:200]
                    pid = f"MONARCH_ROW:{i}:{hash(rid) % (10**8)}"
                name = (row.get("name") or row.get("full_name") or pid)[:500]
                cat = _category_from_row(row)
                desc = (row.get("description") or "").strip() or None
                syn = []
                for k in ("synonyms", "synonym", "exact_synonym", "related_synonym"):
                    v = row.get(k)
                    if v and str(v).strip():
                        syn.append(str(v)[:500])
                xrefs: list[str] = []
                xr = row.get("xref") or row.get("iri") or ""
                if xr:
                    xrefs.append(str(xr)[:500])
                nodes.append(
                    NodeRecord(
                        primary_id=pid,
                        name=name,
                        category=cat,
                        synonyms=syn[:20],
                        description=desc,
                        xrefs=xrefs,
                        source=self.source_name,
                        version=version,
                        properties={k: v for k, v in row.items() if v and k not in skip_cols},
                    )
                )
        except OSError as e:
            logger.warning("Monarch parse failed (non-fatal): %s", e)

        self._write_jsonl_sample(out_parsed_dir, "monarch", nodes, edges)
        return nodes, edges
