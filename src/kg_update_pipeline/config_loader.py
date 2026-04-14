"""
Load and validate YAML pipeline config.

Raises ``ConfigError`` on missing required keys. CLI can override dry_run / mode / sources.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Keys stored on ``SourceEntry``; any other keys go to ``extra``.
KNOWN_SOURCE_KEYS = frozenset(
    {
        "enabled",
        "type",
        "url",
        "filename",
        "api_key",
        "note",
        "archive_member",
        "remove_archive_after_extract",
        "timeout_seconds",
        "fallback_urls",
    }
)


class ConfigError(Exception):
    """Invalid or incomplete configuration."""


# Parser ``NodeRecord.category`` → Neo4j label when ``use_legacy_entity_labels`` (overridable in YAML).
DEFAULT_PARSER_CATEGORY_TO_NEO4J_LABEL: dict[str, str] = {
    "Disease": "disease",
    "Phenotype": "effect/phenotype",
    "Gene": "gene/protein",
    "ExternalResource": "ExternalResource",
    "UnknownEntity": "UnknownEntity",
    # Monarch / biolink-style categories (override via ``category_label_map``).
    "Pathway": "pathway",
    "BiologicalProcess": "biological_process",
    "CellularComponent": "cellular_component",
    "MolecularFunction": "molecular_function",
    "ChemicalEntity": "ChemicalEntity",
    "Cell": "Cell",
    "AnatomicalEntity": "AnatomicalEntity",
    "Genotype": "Genotype",
    "OrganismTaxon": "OrganismTaxon",
    "LifeStage": "LifeStage",
    "SequenceVariant": "SequenceVariant",
    "MolecularEntity": "MolecularEntity",
    "NamedThing": "NamedThing",
}


def build_category_label_map(neo: Neo4jConfig) -> dict[str, str]:
    """
    Merge defaults with ``neo4j.category_label_map`` (YAML wins on key clash).

    If ``use_legacy_entity_labels`` is false, YAML must list every parser category
    and its DB label (must match ``CALL db.labels()``).
    """
    if neo.use_legacy_entity_labels:
        out = dict(DEFAULT_PARSER_CATEGORY_TO_NEO4J_LABEL)
        out.update(neo.category_label_map)
        return out
    if not neo.category_label_map:
        raise ConfigError(
            "When neo4j.use_legacy_entity_labels is false, neo4j.category_label_map must "
            "map every parser category (e.g. Disease, Gene) to a label from CALL db.labels()."
        )
    return dict(neo.category_label_map)


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    # If true: defaults + category_label_map; if false: only category_label_map (strict DB alignment).
    use_legacy_entity_labels: bool = True
    # Parser category → Neo4j label; keys must match ``NodeRecord.category`` from parsers.
    category_label_map: dict[str, str] = field(default_factory=dict)
    # If true, verify every mapped label exists in db.labels() (and additional_node_labels).
    validate_category_label_targets: bool = False
    # If true, only MERGE relationship types that already exist in the DB.
    restrict_to_existing_relationship_types: bool = True
    # If true, skip nodes whose target label is not in db.labels() (empty DB skips all).
    restrict_to_existing_node_labels: bool = False
    # If false: MERGE creates only; existing nodes/relationships keep old props (faster).
    refresh_existing_nodes: bool = False
    refresh_existing_relationships: bool = False
    # UNWIND batch size; pair with entity_id indexes on each label.
    merge_batch_size: int = 2000
    # Extra labels after MERGE (e.g. ``entity_id`` secondary tag).
    additional_node_labels: list[str] = field(default_factory=list)


@dataclass
class UpdateConfig:
    mode: str = "incremental"  # incremental | full
    dry_run: bool = False
    # If true, never HTTP-download sources for this run (use raw/ under version dir); override with CLI ``--force-download``.
    skip_download: bool = False
    backup_before_update: bool = True
    archive_previous_versions: bool = True
    keep_last_n_versions: int = 5
    allow_destructive_ops: bool = False
    # In incremental mode, skip parse when download hash unchanged.
    skip_parse_if_unchanged: bool = True
    # Write parsed exports under parsed/ (jsonl + csv); may cap row count.
    export_parsed_jsonl: bool = True
    parsed_export_max_rows_per_file: int = 500_000
    # Version archive: tar.gz or zip.
    archive_format: str = "tar.gz"
    # urllib timeout (connect + read chunk wait) in seconds.
    http_timeout_seconds: int = 600
    http_retries: int = 3
    # Minimum timeout for tar.gz downloads (max with http_timeout_seconds); large Monarch pulls.
    tar_gz_floor_timeout_seconds: int = 3600
    # Run Docker offline dump before download/MERGE; needs docker_container or KG_NEO4J_DOCKER_CONTAINER.
    pre_update_docker_dump: bool = False
    docker_container: str = ""


@dataclass
class SourceEntry:
    enabled: bool
    type: str
    url: str
    filename: str
    api_key: str = ""
    note: str = ""
    archive_member: str = ""
    remove_archive_after_extract: bool = True
    # 0: use update.http_timeout_seconds (tar.gz also uses max with tar_gz_floor).
    timeout_seconds: int = 0
    # Try these URLs in order if the primary ``url`` fails.
    fallback_urls: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedConfig:
    project_root: Path
    data_root: Path
    log_root: Path
    backup_root: Path
    state_root: Path
    neo4j: Neo4jConfig
    update: UpdateConfig
    sources: dict[str, SourceEntry]
    raw: dict[str, Any]


def _parse_category_label_map(raw: Any) -> dict[str, str]:
    """Parse ``neo4j.category_label_map`` from YAML (parser category → DB label)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("neo4j.category_label_map must be a YAML mapping")
    return {str(k): str(v) for k, v in raw.items() if k is not None and str(k).strip()}


def _parse_string_list(val: Any) -> list[str]:
    """Parse a string or list of strings (e.g. ``fallback_urls``)."""
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, list):
        return [str(x).strip() for x in val if x is not None and str(x).strip()]
    return []


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d or d[key] in (None, ""):
        raise ConfigError(f"Missing required field '{key}' in {ctx}")
    return d[key]


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Cannot read config {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping")
    return data


def parse_config_dict(data: dict[str, Any], config_path: Path) -> LoadedConfig:
    """
    Build ``LoadedConfig`` from a raw dict.

    Relative ``project_root`` resolves against the config file directory.
    """
    config_dir = config_path.parent.resolve()
    pr_raw = Path(data.get("project_root", ".")).expanduser()
    if pr_raw.is_absolute():
        project_root = pr_raw.resolve()
    else:
        project_root = (config_dir / pr_raw).resolve()
    data_root = Path(_require(data, "data_root", "root")).expanduser()
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()

    log_root = Path(data.get("log_root", "./logs/kg_update")).expanduser()
    if not log_root.is_absolute():
        log_root = (project_root / log_root).resolve()

    backup_root = Path(data.get("backup_root", "./backups/kg")).expanduser()
    if not backup_root.is_absolute():
        backup_root = (project_root / backup_root).resolve()

    state_root = Path(data.get("state_root", "./state")).expanduser()
    if not state_root.is_absolute():
        state_root = (project_root / state_root).resolve()

    neo = data.get("neo4j") or {}
    neo4j_cfg = Neo4jConfig(
        uri=str(_require(neo, "uri", "neo4j")),
        user=str(_require(neo, "user", "neo4j")),
        password=str(_require(neo, "password", "neo4j")),
        database=str(neo.get("database") or "neo4j"),
        use_legacy_entity_labels=bool(neo.get("use_legacy_entity_labels", True)),
        restrict_to_existing_relationship_types=bool(
            neo.get("restrict_to_existing_relationship_types", True)
        ),
        restrict_to_existing_node_labels=bool(neo.get("restrict_to_existing_node_labels", False)),
        refresh_existing_nodes=bool(neo.get("refresh_existing_nodes", False)),
        refresh_existing_relationships=bool(neo.get("refresh_existing_relationships", False)),
        merge_batch_size=int(neo.get("merge_batch_size", 2000)),
        category_label_map=_parse_category_label_map(neo.get("category_label_map")),
        validate_category_label_targets=bool(neo.get("validate_category_label_targets", False)),
        additional_node_labels=_parse_string_list(neo.get("additional_node_labels")),
    )
    if not (50 <= neo4j_cfg.merge_batch_size <= 20000):
        raise ConfigError("neo4j.merge_batch_size must be between 50 and 20000")

    up = data.get("update") or {}
    update_cfg = UpdateConfig(
        mode=str(up.get("mode") or "incremental"),
        dry_run=bool(up.get("dry_run", False)),
        skip_download=bool(up.get("skip_download", False)),
        backup_before_update=bool(up.get("backup_before_update", True)),
        archive_previous_versions=bool(up.get("archive_previous_versions", True)),
        keep_last_n_versions=int(up.get("keep_last_n_versions", 5)),
        allow_destructive_ops=bool(up.get("allow_destructive_ops", False)),
        skip_parse_if_unchanged=bool(up.get("skip_parse_if_unchanged", True)),
        export_parsed_jsonl=bool(up.get("export_parsed_jsonl", True)),
        parsed_export_max_rows_per_file=int(up.get("parsed_export_max_rows_per_file", 500_000)),
        archive_format=str(up.get("archive_format") or "tar.gz"),
        http_timeout_seconds=int(up.get("http_timeout_seconds", 600)),
        http_retries=int(up.get("http_retries", 3)),
        tar_gz_floor_timeout_seconds=int(up.get("tar_gz_floor_timeout_seconds", 3600)),
        pre_update_docker_dump=bool(up.get("pre_update_docker_dump", False)),
        docker_container=str(up.get("docker_container") or ""),
    )
    if update_cfg.mode not in ("incremental", "full"):
        raise ConfigError("update.mode must be 'incremental' or 'full'")
    if update_cfg.archive_format not in ("tar.gz", "zip"):
        raise ConfigError("update.archive_format must be 'tar.gz' or 'zip'")
    if update_cfg.http_timeout_seconds < 10:
        raise ConfigError("update.http_timeout_seconds must be >= 10")
    if update_cfg.http_retries < 1:
        raise ConfigError("update.http_retries must be >= 1")
    if update_cfg.tar_gz_floor_timeout_seconds < 60:
        raise ConfigError("update.tar_gz_floor_timeout_seconds must be >= 60")
    if update_cfg.pre_update_docker_dump and not update_cfg.backup_before_update:
        raise ConfigError("update.pre_update_docker_dump requires update.backup_before_update: true")

    src_map = data.get("sources") or {}
    if not isinstance(src_map, dict) or not src_map:
        raise ConfigError("sources section is required and must be non-empty")

    sources: dict[str, SourceEntry] = {}
    for name, raw in src_map.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"sources.{name} must be a mapping")
        sources[name] = SourceEntry(
            enabled=bool(raw.get("enabled", False)),
            type=str(raw.get("type") or "http"),
            url=str(raw.get("url") or ""),
            filename=str(raw.get("filename") or f"{name}_data.bin"),
            api_key=str(raw.get("api_key") or ""),
            note=str(raw.get("note") or ""),
            archive_member=str(raw.get("archive_member") or ""),
            remove_archive_after_extract=bool(raw.get("remove_archive_after_extract", True)),
            timeout_seconds=int(raw.get("timeout_seconds") or 0),
            fallback_urls=_parse_string_list(raw.get("fallback_urls")),
            extra={k: v for k, v in raw.items() if k not in KNOWN_SOURCE_KEYS},
        )

    return LoadedConfig(
        project_root=project_root,
        data_root=data_root,
        log_root=log_root,
        backup_root=backup_root,
        state_root=state_root,
        neo4j=neo4j_cfg,
        update=update_cfg,
        sources=sources,
        raw=data,
    )


def load_config(path: Path) -> LoadedConfig:
    """Load YAML from ``path`` and return ``LoadedConfig``."""
    data = load_yaml_config(path)
    cfg = parse_config_dict(data, path)
    cfg.raw["_config_path"] = str(path.resolve())
    return cfg


def apply_cli_overrides(
    cfg: LoadedConfig,
    *,
    dry_run: bool | None = None,
    mode: str | None = None,
    sources: list[str] | None = None,
) -> LoadedConfig:
    """Deep-copy config and apply CLI overrides (does not mutate ``cfg``)."""
    new_raw = copy.deepcopy(cfg.raw)
    if dry_run is not None:
        new_raw.setdefault("update", {})["dry_run"] = dry_run
    if mode is not None:
        new_raw.setdefault("update", {})["mode"] = mode
    if sources:
        sm = new_raw.setdefault("sources", {})
        for name in list(sm.keys()):
            sm[name] = dict(sm[name])
            sm[name]["enabled"] = name in sources
    path_str = cfg.raw.get("_config_path", "inline")
    cp = Path(path_str)
    if not cp.is_file():
        cp = Path.cwd() / ".kg_update_config_placeholder.yaml"
    out = parse_config_dict(new_raw, cp)
    out.raw["_config_path"] = path_str
    return out


def attach_config_path(cfg: LoadedConfig, path: Path) -> LoadedConfig:
    """Store config path in ``cfg.raw`` for manifests."""
    cfg.raw["_config_path"] = str(path.resolve())
    return cfg
