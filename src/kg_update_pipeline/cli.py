"""
CLI entry and orchestration: download → parse → backup → Neo4j MERGE → manifest.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_update_pipeline.config_loader import (
    ConfigError,
    LoadedConfig,
    apply_cli_overrides,
    load_config,
)
import kg_update_pipeline as kgp

from kg_update_pipeline.downloader import (
    DownloadResult,
    DownloadStatus,
    download_source,
    make_download_result,
)
from kg_update_pipeline.manifest import build_manifest_skeleton, finalize_manifest, new_run_id
from kg_update_pipeline.neo4j_db.backup import (
    POST_DOCKER_DUMP_SLEEP_SEC,
    record_backup_meta,
    run_pre_update_docker_offline_dump,
    try_external_backup,
)
from kg_update_pipeline.neo4j_db.client import Neo4jClient
from kg_update_pipeline.neo4j_db.local_diff import (
    load_kg_category_map,
    plan_edge_delta,
    plan_node_delta,
)
from kg_update_pipeline.neo4j_db.snapshot_export import export_graph_snapshot
from kg_update_pipeline.neo4j_db.updater import GraphUpdater
from kg_update_pipeline.parsers.base_parser import BaseParser
from kg_update_pipeline.parsers.biomart_parser import BiomartParser
from kg_update_pipeline.parsers.hgnc_parser import HgncParser
from kg_update_pipeline.parsers.hpo_parser import HpoParser
from kg_update_pipeline.parsers.monarch_parser import MonarchParser
from kg_update_pipeline.parsers.mondo_parser import MondoParser
from kg_update_pipeline.parsers.ncbi_parser import NcbiParser
from kg_update_pipeline.parsers.omim_parser import OmimParser
from kg_update_pipeline.parsers.orpha_parser import OrphaParser
from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord
from kg_update_pipeline.utils.audit_log import audit_event
from kg_update_pipeline.utils.io_utils import ensure_dir, read_json, write_json
from kg_update_pipeline.utils.parsed_export import export_source_parsed
from kg_update_pipeline.utils.logger import setup_run_logger
from kg_update_pipeline.utils.time_utils import run_stamp_str, version_date_str
from kg_update_pipeline.version_manager import (
    append_update_history,
    init_version_layout,
    maybe_archive_old_versions,
    version_dir_for_date,
    write_last_success_state,
    write_latest_pointer,
)

logger = logging.getLogger(__name__)


def _should_run_pre_docker_dump(cfg: LoadedConfig, skip_neo4j: bool) -> bool:
    return (
        not skip_neo4j
        and not cfg.update.dry_run
        and cfg.update.backup_before_update
        and cfg.update.pre_update_docker_dump
    )


def _parser_for(name: str) -> BaseParser:
    m: dict[str, BaseParser] = {
        "hpo": HpoParser(),
        "mondo": MondoParser(),
        "hgnc": HgncParser(),
        "monarch": MonarchParser(),
        "omim": OmimParser(),
        "ncbi": NcbiParser(),
        "biomart": BiomartParser(),
        "orpha": OrphaParser(),
    }
    if name not in m:
        raise KeyError(f"No parser registered for source '{name}'")
    return m[name]


def _selected_sources(cfg: LoadedConfig) -> list[str]:
    return [n for n, e in cfg.sources.items() if e.enabled]


def run_pipeline(
    cfg: LoadedConfig,
    *,
    skip_download: bool = False,
    skip_neo4j: bool = False,
    cli_sources: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run the full update pipeline.

    Returns:
        Final manifest dict.
    """
    run_id = new_run_id()
    run_started = datetime.now(timezone.utc)
    version_label = version_date_str(run_started)
    run_stamp = run_stamp_str(run_started)
    version_path = version_dir_for_date(cfg, version_label)
    ensure_dir(version_path.parent)
    raw_dir, parsed_dir = init_version_layout(version_path)

    log_file = cfg.log_root / f"kg_update_{run_stamp}.log"
    if log_file.exists():
        log_file = cfg.log_root / f"kg_update_{run_stamp}_{run_id[:8]}.log"
    setup_run_logger("kg_update", log_file=log_file, verbose=verbose)

    selected = cli_sources if cli_sources else _selected_sources(cfg)
    manifest = build_manifest_skeleton(
        run_id=run_id,
        config_path=str(cfg.raw.get("_config_path", "")),
        selected_sources=selected,
        update_mode=cfg.update.mode,
        dry_run=cfg.update.dry_run,
    )
    # Record package version on the manifest for traceability.
    manifest["pipeline"] = {
        "package": "kg_update_pipeline",
        "version": kgp.__version__,
        "skip_download": skip_download,
        "skip_download_config": cfg.update.skip_download,
    }

    backup_dir = cfg.backup_root / run_stamp
    if backup_dir.exists():
        backup_dir = cfg.backup_root / f"{run_stamp}_{run_id[:8]}"
    early_backup: dict[str, Any] | None = None
    if _should_run_pre_docker_dump(cfg, skip_neo4j):
        try:
            early_backup = run_pre_update_docker_offline_dump(cfg, backup_dir, run_id=run_id)
            if POST_DOCKER_DUMP_SLEEP_SEC > 0:
                logger.info(
                    "Waiting %.1fs after Docker dump before downloads",
                    POST_DOCKER_DUMP_SLEEP_SEC,
                )
                time.sleep(POST_DOCKER_DUMP_SLEEP_SEC)
        except Exception as e:
            logger.exception("pre_update_docker_dump aborted: %s", e)
            manifest.setdefault("errors", []).append(f"pre_update_docker_dump: {e}")
            manifest = finalize_manifest(manifest)
            write_json(version_path / "manifest.json", manifest)
            write_json(
                version_path / "run_summary.json",
                {
                    "run_id": run_id,
                    "version_dir": str(version_path),
                    "log_file": str(log_file),
                    "manifest": str(version_path / "manifest.json"),
                    "failed_stage": "pre_update_docker_dump",
                    "errors": manifest.get("errors", []),
                },
            )
            append_update_history(
                cfg,
                {
                    "run_id": run_id,
                    "version_dir": str(version_path),
                    "sources": selected,
                    "dry_run": cfg.update.dry_run,
                    "ok": False,
                },
            )
            return manifest

    dl_results: dict[str, DownloadResult] = {}
    if skip_download:
        logger.info(
            "Skipping HTTP downloads (use files under %s; from config skip_download=%s)",
            raw_dir,
            cfg.update.skip_download,
        )
    if not skip_download:
        for name in selected:
            entry = cfg.sources.get(name)
            if not entry:
                continue
            dl_results[name] = download_source(
                name,
                entry,
                raw_dir,
                cfg.state_root,
                incremental=(cfg.update.mode == "incremental"),
                http_timeout_seconds=cfg.update.http_timeout_seconds,
                http_retries=cfg.update.http_retries,
                tar_gz_floor_timeout_seconds=cfg.update.tar_gz_floor_timeout_seconds,
            )
    else:
        logger.info("skip-download: using existing files under %s", raw_dir)
        latest = read_json(cfg.data_root / "latest.json", default=None)
        prev_raw: Path | None = None
        if isinstance(latest, dict) and latest.get("version_dir"):
            prev_raw = Path(latest["version_dir"]) / "raw"
        for name in selected:
            entry = cfg.sources.get(name)
            if not entry:
                continue
            p = raw_dir / entry.filename
            if not (p.is_file() and p.stat().st_size > 0) and prev_raw:
                alt = prev_raw / entry.filename
                if alt.is_file() and alt.stat().st_size > 0:
                    try:
                        ensure_dir(p.parent)
                        shutil.copy2(alt, p)
                        manifest["warnings"].append(
                            f"skip-download: copied {alt} -> {p} from latest version raw"
                        )
                    except OSError as e:
                        manifest["warnings"].append(f"skip-download copy failed: {e}")
            if p.is_file() and p.stat().st_size > 0:
                from kg_update_pipeline.utils.hashing import file_sha256

                dl_results[name] = make_download_result(
                    source=name,
                    status=DownloadStatus.SUCCESS,
                    path=p,
                    sha256=file_sha256(p),
                    message="skip-download",
                )
            else:
                dl_results[name] = make_download_result(
                    source=name,
                    status=DownloadStatus.SKIPPED,
                    message="file missing in raw dir (and no fallback from latest)",
                )

    manifest["files"] = {
        k: {
            "status": v.status.value,
            "path": str(v.path) if v.path else None,
            "sha256": v.sha256,
            "message": v.message,
            "downloaded_at": v.downloaded_at,
            "size_bytes": v.size_bytes,
        }
        for k, v in dl_results.items()
    }

    all_nodes: list[NodeRecord] = []
    all_edges: list[EdgeRecord] = []
    parse_stats: dict[str, Any] = {}

    for name in selected:
        entry = cfg.sources[name]
        raw_path = raw_dir / entry.filename
        dr = dl_results.get(name)
        if (
            cfg.update.skip_parse_if_unchanged
            and cfg.update.mode == "incremental"
            and dr
            and dr.status == DownloadStatus.UNCHANGED
        ):
            logger.info("[%s] skip parse (content unchanged)", name)
            parse_stats[name] = {"skipped": True, "reason": "unchanged"}
            continue
        if dr and dr.status in (DownloadStatus.SKIPPED, DownloadStatus.FAILED):
            parse_stats[name] = {"skipped": True, "reason": dr.status.value, "msg": dr.message}
            continue

        parser = _parser_for(name)
        try:
            nodes, edges = parser.parse(
                raw_path,
                version=version_label,
                out_parsed_dir=parsed_dir,
            )
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            st = {
                "nodes": len(nodes),
                "edges": len(edges),
            }
            if cfg.update.export_parsed_jsonl:
                try:
                    st["parsed_export"] = export_source_parsed(
                        parsed_dir,
                        name,
                        nodes,
                        edges,
                        max_rows=cfg.update.parsed_export_max_rows_per_file,
                    )
                except OSError as ex:
                    logger.warning("[%s] parsed export failed: %s", name, ex)
                    manifest["warnings"].append(f"{name}: parsed export: {ex}")
            parse_stats[name] = st
        except Exception as e:
            logger.exception("[%s] parse failed (other sources continue): %s", name, e)
            manifest["errors"].append(f"{name}: parse: {e!s}")
            parse_stats[name] = {"error": str(e)}

    manifest["parse_stats"] = parse_stats

    id_to_category: dict[str, str] = {}
    for n in all_nodes:
        id_to_category[n.primary_id] = n.category

    neo_stats: dict[str, Any] = {"skipped": skip_neo4j}
    if not skip_neo4j and selected:
        backup_info: dict[str, Any] = {}
        if cfg.update.backup_before_update and not cfg.update.dry_run:
            if early_backup is not None:
                backup_info = dict(early_backup)
            else:
                meta_path = record_backup_meta(cfg, backup_dir, run_id=run_id)
                backup_info["backup_meta"] = str(meta_path)
                ext = try_external_backup(cfg, backup_dir)
                backup_info.update(ext)
                if ext.get("warnings"):
                    for w in ext["warnings"]:
                        manifest["warnings"].append(w)
        elif cfg.update.dry_run:
            backup_info["note"] = "dry_run: skipped filesystem/neo4j-admin backup"

        delta_nodes_path: Path | None = None
        delta_rels_path: Path | None = None
        dr = cfg.update.dry_run
        if not dr:
            delta_nodes_path = version_path / "neo4j_created_nodes.jsonl"
            delta_rels_path = version_path / "neo4j_created_relationships.jsonl"
            for p in (delta_nodes_path, delta_rels_path):
                if p.exists():
                    p.unlink()

        client = Neo4jClient(
            cfg.neo4j.uri,
            cfg.neo4j.user,
            cfg.neo4j.password,
            cfg.neo4j.database,
        )
        client.connect()
        try:
            updater = GraphUpdater(client, cfg.neo4j)
            nodes_to_merge = all_nodes
            edges_to_merge = all_edges
            local_summary: dict[str, Any] = {"mode": "local_diff_export_and_delta"}
            audit_path = version_path / "update_audit.jsonl"

            snap_dir = version_path / "neo4j_snapshot"
            audit_event(
                audit_path,
                "local_diff_export_start",
                dry_run=dr,
                run_id=run_id,
            )
            snap = export_graph_snapshot(client, snap_dir)
            audit_event(
                audit_path,
                "local_diff_export_done",
                run_id=run_id,
                **snap,
            )
            ns = snap_dir / "nodes.jsonl"
            es = snap_dir / "edges.jsonl"
            nodes_planned, nplan = plan_node_delta(
                all_nodes,
                nodes_jsonl=ns,
                refresh_existing_nodes=cfg.neo4j.refresh_existing_nodes,
            )
            if cfg.neo4j.refresh_existing_relationships:
                edges_planned = all_edges
                eplan = {
                    "edges_planned_merge": len(all_edges),
                    "note": "full_edge_list_because_refresh_existing_relationships",
                }
            else:
                edges_planned, eplan = plan_edge_delta(all_edges, edges_jsonl=es)
            audit_event(
                audit_path,
                "local_diff_plan",
                run_id=run_id,
                node_plan=nplan,
                edge_plan=eplan,
            )
            nodes_to_merge = nodes_planned
            edges_to_merge = edges_planned
            local_summary.update(
                {
                    "snapshot": snap,
                    "node_plan": nplan,
                    "edge_plan": eplan,
                }
            )
            snap_cat = load_kg_category_map(ns)
            merged_cat = {**snap_cat, **id_to_category}
            id_to_category = merged_cat
            write_json(version_path / "local_diff_summary.json", local_summary)
            manifest["local_diff_summary_path"] = str(version_path / "local_diff_summary.json")

            nstat = updater.merge_nodes(
                nodes_to_merge,
                dry_run=dr,
                nodes_delta_jsonl=None if dr else delta_nodes_path,
            )
            estat = updater.merge_edges(
                edges_to_merge,
                id_to_category,
                dry_run=dr,
                rels_delta_jsonl=None if dr else delta_rels_path,
            )
            neo_stats = {
                **{k: v for k, v in nstat.items() if k != "failed_records"},
                **{k: v for k, v in estat.items() if k != "failed_records"},
                "failed_records_nodes": nstat.get("failed_records", 0),
                "failed_records_edges": estat.get("failed_records", 0),
                "backup": backup_info,
                "dry_run": dr,
                "neo4j_update_mode": "local_diff",
            }
            neo_stats["local_diff"] = local_summary
            audit_event(
                audit_path,
                "neo4j_merge_complete",
                run_id=run_id,
                neo4j_update_mode="local_diff",
                nodes_merged=nstat.get("nodes_merged"),
                relationships_merged=estat.get("relationships_merged"),
                dry_run=dr,
            )
            if not dr:
                merge_delta: dict[str, Any] = {
                    "run_id": run_id,
                    "skipped_parse_unchanged": sorted(
                        k
                        for k, v in parse_stats.items()
                        if v.get("skipped") and v.get("reason") == "unchanged"
                    ),
                    "nodes_created": nstat.get("nodes_created", 0),
                    "nodes_updated": nstat.get("nodes_updated", 0),
                    "relationships_created": estat.get("relationships_created", 0),
                    "relationships_updated": estat.get("relationships_updated", 0),
                }
                if delta_nodes_path and delta_nodes_path.is_file():
                    merge_delta["created_nodes_jsonl"] = str(delta_nodes_path)
                if delta_rels_path and delta_rels_path.is_file():
                    merge_delta["created_relationships_jsonl"] = str(delta_rels_path)
                write_json(version_path / "neo4j_merge_delta.json", merge_delta)
                manifest["neo4j_merge_delta"] = merge_delta
        finally:
            client.close()

    manifest["neo4j_stats"] = neo_stats
    manifest = finalize_manifest(manifest)

    write_json(version_path / "manifest.json", manifest)
    summary_path = version_path / "run_summary.json"
    summary_payload: dict[str, Any] = {
        "run_id": run_id,
        "version_dir": str(version_path),
        "log_file": str(log_file),
        "manifest": str(version_path / "manifest.json"),
        "parse_stats": parse_stats,
        "neo4j_stats": neo_stats,
    }
    if manifest.get("neo4j_merge_delta"):
        summary_payload["neo4j_merge_delta"] = str(version_path / "neo4j_merge_delta.json")
    ap = version_path / "update_audit.jsonl"
    if ap.is_file():
        summary_payload["update_audit"] = str(ap)
    write_json(summary_path, summary_payload)

    write_latest_pointer(cfg, version_path, {"run_id": run_id})
    append_update_history(
        cfg,
        {
            "run_id": run_id,
            "version_dir": str(version_path),
            "sources": selected,
            "dry_run": cfg.update.dry_run,
        },
    )
    if not cfg.update.dry_run and not manifest.get("errors"):
        write_last_success_state(cfg, version_path, version_path / "manifest.json")

    maybe_archive_old_versions(cfg, version_path)

    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Rare disease KG auto-update pipeline (download, parse, MERGE into Neo4j).",
    )
    p.add_argument(
        "-c",
        "--config",
        "--congfig",
        required=True,
        dest="config",
        metavar="CONFIG",
        help="Path to YAML config",
    )
    p.add_argument("--dry-run", action="store_true", help="No Neo4j writes")
    p.add_argument("--mode", choices=["incremental", "full"], help="Override update mode")
    p.add_argument("--source", action="append", dest="sources", help="Only these sources (repeatable)")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HTTP download this run (also set update.skip_download in YAML)",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Download this run even when update.skip_download is true in YAML",
    )
    p.add_argument("--skip-neo4j", action="store_true", help="Parse only, no database")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = p.parse_args(argv)

    try:
        cfg_path = Path(args.config).resolve()
        cfg = load_config(cfg_path)
        if args.sources:
            for s in args.sources:
                if s not in cfg.sources:
                    raise ConfigError(f"Unknown source '{s}'. Valid: {sorted(cfg.sources.keys())}")
        cfg = apply_cli_overrides(
            cfg,
            dry_run=True if args.dry_run else None,
            mode=args.mode,
            sources=args.sources,
        )
        skip_dl = bool(cfg.update.skip_download)
        if args.skip_download:
            skip_dl = True
        if args.force_download:
            skip_dl = False
        manifest = run_pipeline(
            cfg,
            skip_download=skip_dl,
            skip_neo4j=args.skip_neo4j,
            cli_sources=args.sources,
            verbose=args.verbose,
        )
        if manifest.get("errors"):
            for err in manifest["errors"]:
                logger.error("%s", err)
            return 1
        return 0
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        logging.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
