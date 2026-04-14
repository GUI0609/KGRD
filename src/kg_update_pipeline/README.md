# KG Update Pipeline

**Versioned downloads → unified parse model → local snapshot diff → idempotent Neo4j MERGE** for rare-disease and ontology-style knowledge graphs. Raw files and exports are kept per run under `data_root`; the database is updated with `MERGE` only (no full wipe in core paths).

**中文文档：** [README.zh.md](README.zh.md)

---

## Table of contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [Neo4j integration](#neo4j-integration)
7. [Local diff & audit files](#local-diff--audit-files)
8. [Command-line interface](#command-line-interface)
9. [Run outputs & directory layout](#run-outputs--directory-layout)
10. [Incremental behaviour](#incremental-behaviour)
11. [Performance](#performance)
12. [Optional Docker backup](#optional-docker-backup)
13. [Testing](#testing)
14. [Package layout](#package-layout)
15. [Security & Git](#security--git)
16. [Troubleshooting](#troubleshooting)

---

## Overview

| Topic | Behaviour |
|-------|-----------|
| **End-to-end flow** | Download (optional) → parse sources into `NodeRecord` / `EdgeRecord` → export current Neo4j state to JSONL → compute deltas in Python → batch `MERGE` only what changed (nodes by `entity_id` fingerprint; edges by `(source_id, target_id, rel)` key). |
| **Writes** | `MERGE` on nodes (`entity_id` = parser `primary_id`) and relationships. No `DELETE` / `DETACH DELETE` in core modules. |
| **Relationship types** | With `neo4j.restrict_to_existing_relationship_types: true` (default), only types from `CALL db.relationshipTypes()` are merged; others are skipped (see logs). |
| **Node labels** | Parser `category` → Neo4j label via built-in defaults + `neo4j.category_label_map`. Optional `restrict_to_existing_node_labels` skips labels not in `db.labels()`. |
| **Refresh vs create-only** | Default `refresh_existing_nodes` / `refresh_existing_relationships`: **false** → node/rel props mainly on **`ON CREATE`**. Set **true** to refresh existing entities each run (slower). |
| **Extra labels** | `neo4j.additional_node_labels` (e.g. `entity_id`) adds secondary labels after `MERGE` (e.g. `(:disease:entity_id)`). |
| **Downloads** | `update.skip_download: true` avoids HTTP downloads every run; reuse `versions/.../raw/`. Override once with `--force-download`. |

Supported parsers: **HPO**, **MONDO**, **HGNC**, **Monarch** (nodes TSV), placeholders for **OMIM**, **NCBI**, **BioMart**, **Orphanet**. Monarch **edges** are not loaded unless you extend the parser.

For one-off huge bulk loads, consider `neo4j-admin import` or `LOAD CSV`; this pipeline optimises for **repeatable incremental MERGE** with versioning and audit trails.

---

## Requirements

- **Python** 3.10+
- **Neo4j** 4.4+ or 5.x (Bolt)
- **Packages** (this folder):

```bash
pip install -r kg_update_pipeline/requirements_kg_update.txt
```

Dependencies: `neo4j`, `PyYAML`, `tqdm`.

---

## Installation

The import path is `kg_update_pipeline`. Run commands from the **parent directory** of `kg_update_pipeline/` (e.g. repository root):

```bash
cd /path/to/parent-of-kg_update_pipeline
pip install -r kg_update_pipeline/requirements_kg_update.txt
```

---

## Quick start

1. Copy [`templates/my_kg_update.yaml`](templates/my_kg_update.yaml) to **`my_kg_update.yaml`** next to the template or under the package root (see [.gitignore](.gitignore); do not commit secrets).
2. Edit `neo4j.uri`, `neo4j.user`, `neo4j.password`, and paths (`data_root`, `log_root`, `backup_root`, `state_root`).
3. Run **[`scripts/ensure_entity_id_indexes.cypher`](scripts/ensure_entity_id_indexes.cypher)** once per database; wait until indexes are **ONLINE**.
4. Run:

```bash
python kg_update_pipeline/scripts/run_kg_update.py -c kg_update_pipeline/my_kg_update.yaml
```

Short flags: `-c` / `--config`. Typo tolerance: `--congfig`.

---

## Configuration

Paths in YAML are **relative to the config file’s directory** unless absolute.

| Block | Role |
|-------|------|
| `project_root` | Base for resolving other relative paths when needed. |
| `data_root` | `versions/<YYYY-MM-DD>/`, `latest.json`, `update_history.jsonl`, `archives/`. |
| `log_root` | `kg_update_*.log` files. |
| `backup_root` | Backup metadata and optional dump paths. |
| `state_root` | Per-source download hashes, `last_success.json`, optional snapshots. |
| `neo4j.*` | Bolt settings, `category_label_map`, `merge_batch_size`, `restrict_to_existing_*`, `refresh_existing_*`, `additional_node_labels`, `validate_category_label_targets`. |
| `update.*` | `mode` (`incremental` \| `full`), `dry_run`, **`skip_download`**, backups, `skip_parse_if_unchanged`, `export_parsed_jsonl`, HTTP timeouts, `pre_update_docker_dump`, `archive_format`, `keep_last_n_versions`. |
| `sources.<name>` | `enabled`, `type` (`http`, `http_tar_gz`, `manual_*`, …), `url`, `filename`, `fallback_urls`, `archive_member`, `timeout_seconds`. |

**Label mapping**

- `use_legacy_entity_labels: true` — merge defaults with `category_label_map` (YAML overrides on conflict).
- `use_legacy_entity_labels: false` — **only** `category_label_map`; every emitted parser `category` must be listed and must match `CALL db.labels()`.

---

## Neo4j integration

### Indexes

Create indexes on `(n:Label {entity_id})` for every label you merge. Script: [`scripts/ensure_entity_id_indexes.cypher`](scripts/ensure_entity_id_indexes.cypher). If you use `additional_node_labels: [entity_id]`, include the `(:entity_id)` index line in that script.

### Secondary label example

```yaml
neo4j:
  additional_node_labels:
    - entity_id
```

### Tuning

- `merge_batch_size`: default `2000`, allowed `50`–`20000`.
- Tune Neo4j **heap** and **page cache** for large batches or heavy `kg_properties_json`.

### Relationship merge and `restrict_to_existing_relationship_types`

If **`relationships_merged`** stays **0** while many edges are planned, check logs for skipped types. Parser relations (e.g. OBO `relationship:` lines) must normalise to a type that exists in your graph when this flag is `true`.

---

## Local diff & audit files

Each run (with Neo4j enabled) typically produces under `data_root/versions/<date>/`:

| Path | Description |
|------|-------------|
| `neo4j_snapshot/nodes.jsonl` | Streamed export of nodes with `entity_id` (+ `fp` fingerprint for refresh logic). |
| `neo4j_snapshot/edges.jsonl` | Export of relationships between nodes that have `entity_id`. |
| `local_diff_summary.json` | Counts: snapshot size, parsed size, planned node/edge merges, skipped counts. |
| `update_audit.jsonl` | Append-only audit: `local_diff_export_*`, `local_diff_plan`, `neo4j_merge_complete`. |

With `refresh_existing_relationships: true`, the **edge** phase sends the **full** parsed edge list (node phase stays delta-based via fingerprints / id set).

---

## Command-line interface

| Flag | Description |
|------|-------------|
| `-c`, `--config`, `--congfig` PATH | YAML config (**required**). |
| `--dry-run` | No Neo4j writes; still may connect and log planned work. |
| `--mode full` | Override `update.mode` to `full`. |
| `--skip-download` | Skip HTTP download this run. |
| `--force-download` | Download even if `update.skip_download: true` in YAML. |
| `--skip-neo4j` | Parse / version only; no Bolt. |
| `--source NAME` | Repeatable; restrict to these sources. |
| `-v`, `--verbose` | Debug-level logging. |

Entry point: [`scripts/run_kg_update.py`](scripts/run_kg_update.py).

---

## Run outputs & directory layout

```
data_root/
  latest.json                 # Pointer to last version_dir
  update_history.jsonl        # One JSON object per run
  archives/                   # Packed old version dirs (tar.gz or zip)
  versions/
    YYYY-MM-DD/
      manifest.json
      run_summary.json
      neo4j_merge_delta.json  # Counts + paths to delta JSONL when present
      local_diff_summary.json
      update_audit.jsonl
      neo4j_created_nodes.jsonl
      neo4j_created_relationships.jsonl
      neo4j_snapshot/
        nodes.jsonl
        edges.jsonl
      raw/                    # Downloaded upstream files
      parsed/                 # Optional JSONL/CSV exports per source
```

`backup_root` and `log_root` mirror the paths set in YAML.

---

## Incremental behaviour

- **`update.mode: incremental`** with **`skip_parse_if_unchanged: true`**: if a source’s downloaded file **SHA-256** matches the previous run, that source **skips parsing** and contributes **no new merge work** for that run.
- Force full re-parse: **`--mode full`** or **`skip_parse_if_unchanged: false`**.

---

## Performance

1. **Indexes** on `entity_id` per label (and secondary labels as needed).  
2. **Local diff** reduces Neo4j `MERGE` row counts when most data is unchanged (`refresh_existing_nodes: false`).  
3. **`merge_batch_size`** trade-off: fewer round-trips vs. larger transactions.  
4. **`refresh_existing_*: false`** (default) minimises property rewrites.  
5. **Export** still scans the DB once per run; very large graphs spend noticeable time in `neo4j_snapshot/`.

---

## Optional Docker backup

- `update.pre_update_docker_dump: true` and `update.backup_before_update: true`
- `update.docker_container` or **`KG_NEO4J_DOCKER_CONTAINER`**

See [`neo4j_db/backup.py`](neo4j_db/backup.py) and [`scripts/kg_neo4j_dump.py`](scripts/kg_neo4j_dump.py).

---

## Testing

```bash
python kg_update_pipeline/e2e_full_verify.py --help
```

---

## Package layout

| Path | Role |
|------|------|
| `cli.py` | Orchestration: download → parse → backup → snapshot → diff → MERGE → manifest. |
| `config_loader.py` | YAML → `LoadedConfig`. |
| `downloader.py` | HTTP / tar.gz with retries and hashing. |
| `manifest.py` | Run manifest assembly. |
| `version_manager.py` | Version dirs, `latest.json`, archives. |
| `parsers/` | Source parsers; `base_parser.py` (OBO, TSV streaming). |
| `schema/` | `NodeRecord`, `EdgeRecord`, normaliser. |
| `neo4j_db/` | Client, export, local diff, introspection, `GraphUpdater`. |
| `utils/` | I/O, logging, audit JSONL, hashing, time, parsed export. |
| `scripts/` | `run_kg_update.py`, Cypher helpers, Docker dump helper. |
| `templates/` | Example YAML only (no secrets). |

The internal package directory is named **`neo4j_db`** to avoid clashing with the PyPI **`neo4j`** driver package name on `sys.path`.

---

## Security & Git

- **Never commit** real passwords. This repo’s [`.gitignore`](.gitignore) ignores `my_kg_update.yaml`, `.env*`, runtime dirs (`kg_data/`, `logs/`, `state/`, `backups/`), dumps, and common Python artefacts.
- Use **`templates/my_kg_update.yaml`** as the committed template; keep private config outside Git or in a secret store.
- `update.allow_destructive_ops` is a guard for future destructive Cypher; the default pipeline only **MERGE**s.

---

## Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| `relationships_merged: 0` with many planned edges | `restrict_to_existing_relationship_types`; do parsed `relation` strings map to existing DB types? Endpoint nodes must exist with expected labels and `entity_id`. |
| Very slow MERGE | Missing or non-**ONLINE** `entity_id` indexes; reduce `merge_batch_size` if OOM. |
| Long phase before MERGE | Full-graph export to `neo4j_snapshot/`; expected on large DBs. |
| `skip_parse_if_unchanged` skipped work | Incremental mode + unchanged hash; use `--mode full` to re-parse. |
| `Config error` / argparse | Use `-c` or `--config` (not `--congfig` only if your shell strips it—both work). |

---

## License

If this package is published as part of a larger repository, follow that repository’s root **LICENSE**. This subdirectory does not ship a separate license file by default.
