"""Shared record types for parsed graph data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeRecord:
    """One ontology or entity node; ``primary_id`` becomes Neo4j ``entity_id``."""

    primary_id: str
    name: str
    category: str
    synonyms: list[str] = field(default_factory=list)
    description: str | None = None
    xrefs: list[str] = field(default_factory=list)
    source: str = ""
    version: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeRecord:
    """One directed relationship between two entity ids."""

    source_id: str
    target_id: str
    relation: str
    source: str = ""
    version: str = ""
    evidence: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
