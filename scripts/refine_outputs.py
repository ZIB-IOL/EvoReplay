#!/usr/bin/env python3
"""Refine a skydiscover run directory into a deduplicated, lossless layout.

Reads from a source run directory (read-only) and writes a refined copy to
a destination directory. The source is never modified.

Backends supported: openevolve_native, gepa_native, evox, shinkaevolve.

Output layout (per run):

    <dst>/
      meta.json                  backend, source path, refined timestamp, counts
      run_info.json              copied from source
      run_config.yaml            copied from source
      best/, logs/, analysis/    symlinked to source (no copy)
      programs.jsonl             unique program records, deduped by UUID
                                 (rows include status: accepted | rejected | failed)
      blobs/<sha[:2]>/<sha>.<ext>
                                 content-addressed solution sources (.txt) and
                                 prompts payloads (.json)
      iterations.jsonl           per-checkpoint role membership
                                 {iteration, role, slot_key, program_id, value}
      iter_scalars.jsonl         {iteration, key, value} (best_program_id,
                                 last_iteration, current_island, ...)
      feature_stats.jsonl            openevolve-only: feature_stats.values flattened
      evox_search.jsonl              evox-only: search/iteration_N/{code,metadata}
      evox_failed_attempts.jsonl     evox-only: search/iteration_N/failed_attempts.json
      shinka_events.jsonl            shinkaevolve-only: generation_event_log
      shinka_attempts.jsonl          shinkaevolve-only: attempt_log
      shinka_gen_metrics.jsonl       shinkaevolve-only: per-gen results/metrics.json
      shinka_private_lineage.jsonl   shinkaevolve-only: private_eval_lineage.json
                                     (held-out test re-evals along the public
                                      best-so-far chain)

For shinkaevolve, archive-table membership is folded into iterations.jsonl
under role="archive"; metadata_store keys are folded into iter_scalars.jsonl
with iteration=null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ---------- helpers ----------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_blob(blobs_dir: Path, content: bytes, ext: str) -> str:
    h = sha256_bytes(content)
    out = blobs_dir / h[:2] / f"{h}.{ext}"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.rename(out)
    return h


def jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8")
    def write(row: dict):
        f.write(json.dumps(row, default=str, ensure_ascii=False))
        f.write("\n")
    return f, write


def detect_backend(run: Path) -> str:
    """Decide backend from on-disk hints."""
    cks = run / "checkpoints"
    if cks.is_dir():
        for c in cks.iterdir():
            if (c / "openevolve_native_metadata.json").exists():
                return "openevolve_native"
            if (c / "gepa_metadata.json").exists():
                return "gepa_native"
    if (run / "search").is_dir():
        return "evox"
    if (run / "programs.sqlite").exists():
        return "shinkaevolve"
    raise RuntimeError(f"Unable to detect backend for {run}")


def list_checkpoints(run: Path) -> list[tuple[int, Path]]:
    cks = run / "checkpoints"
    if not cks.is_dir():
        return []
    out = []
    for d in cks.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint_"):
            try:
                out.append((int(d.name.split("_")[1]), d))
            except ValueError:
                pass
    out.sort()
    return out


# ---------- program record handling ----------

PROGRAM_FIELDS_PASSTHROUGH = (
    "id", "language", "metrics", "iteration_found", "parent_id",
    "other_context_ids", "parent_info", "context_info", "timestamp",
    "metadata", "generation",
)


@dataclass
class Refiner:
    src: Path
    dst: Path
    backend: str
    seen_program_ids: set[str] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=lambda: {
        "checkpoints": 0,
        "accepted_unique": 0,
        "accepted_dup_skipped": 0,
        "rejected_unique": 0,
        "rejected_dup_skipped": 0,
        "failed_unique": 0,
        "blobs_solution": 0,
        "blobs_prompts": 0,
    })

    def __post_init__(self) -> None:
        self.blobs = self.dst / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)

    def emit_program(self, write, rec: dict, status: str, source: str,
                     extras: dict | None = None) -> bool:
        """Write a program row if not already seen. Return True if written.

        ``extras`` are merged into the row after the canonical fields, so
        backend-specific columns can be preserved without breaking the
        shared schema.
        """
        pid = rec.get("id")
        if not pid:
            return False
        if pid in self.seen_program_ids:
            self.counts[f"{status}_dup_skipped"] = self.counts.get(
                f"{status}_dup_skipped", 0) + 1
            return False
        self.seen_program_ids.add(pid)

        row: dict[str, Any] = {k: rec.get(k) for k in PROGRAM_FIELDS_PASSTHROUGH}
        row["status"] = status
        row["source"] = source

        sol = rec.get("solution")
        if isinstance(sol, str) and sol:
            sha = write_blob(self.blobs, sol.encode("utf-8"), "txt")
            row["solution_sha256"] = sha
            self.counts["blobs_solution"] += 1
        else:
            row["solution_sha256"] = None

        prompts = rec.get("prompts")
        if prompts:
            sha = write_blob(self.blobs,
                             json.dumps(prompts, ensure_ascii=False,
                                        sort_keys=True).encode("utf-8"),
                             "json")
            row["prompts_sha256"] = sha
            self.counts["blobs_prompts"] += 1
        else:
            row["prompts_sha256"] = None

        artifacts = rec.get("artifacts")
        row["artifacts"] = artifacts if artifacts else None

        if extras:
            row.update(extras)

        write(row)
        self.counts[f"{status}_unique"] = self.counts.get(
            f"{status}_unique", 0) + 1
        return True


# ---------- backend dispatch ----------

def refine_openevolve(r: Refiner, prog_w, iter_w, scal_w, feat_w) -> None:
    for it, ck in list_checkpoints(r.src):
        r.counts["checkpoints"] += 1
        # accepted programs in population
        prog_dir = ck / "programs"
        if prog_dir.is_dir():
            for jf in sorted(prog_dir.iterdir()):
                if jf.suffix != ".json":
                    continue
                rec = json.loads(jf.read_text())
                r.emit_program(prog_w, rec, status="accepted",
                               source="checkpoint.programs")

        meta = json.loads((ck / "metadata.json").read_text())
        scal_w({"iteration": it, "key": "best_program_id",
                "value": meta.get("best_program_id")})
        scal_w({"iteration": it, "key": "last_iteration",
                "value": meta.get("last_iteration")})

        oe_path = ck / "openevolve_native_metadata.json"
        if oe_path.exists():
            oe = json.loads(oe_path.read_text())
            scal_w({"iteration": it, "key": "current_island",
                    "value": oe.get("current_island")})
            scal_w({"iteration": it, "key": "last_migration_generation",
                    "value": oe.get("last_migration_generation")})
            for i, gen in enumerate(oe.get("island_generations", [])):
                scal_w({"iteration": it, "key": f"island_generation[{i}]",
                        "value": gen})

            for i, members in enumerate(oe.get("islands", [])):
                for slot, pid in enumerate(members):
                    iter_w({"iteration": it, "role": "island_member",
                            "island_idx": i, "slot_key": slot,
                            "program_id": pid, "value": None})
            for i, pid in enumerate(oe.get("island_best_programs", [])):
                iter_w({"iteration": it, "role": "island_best",
                        "island_idx": i, "slot_key": None,
                        "program_id": pid, "value": None})
            for slot, pid in enumerate(oe.get("archive", [])):
                iter_w({"iteration": it, "role": "archive",
                        "island_idx": None, "slot_key": slot,
                        "program_id": pid, "value": None})
            for i, fmap in enumerate(oe.get("island_feature_maps", [])):
                for cell, pid in (fmap or {}).items():
                    iter_w({"iteration": it, "role": "feature_map",
                            "island_idx": i, "slot_key": cell,
                            "program_id": pid, "value": None})

            for fname, stats in (oe.get("feature_stats") or {}).items():
                for stat_key in ("min", "max"):
                    if stat_key in stats:
                        feat_w({"iteration": it, "feature": fname,
                                "stat": stat_key, "value_idx": None,
                                "value": stats[stat_key]})
                for idx, v in enumerate(stats.get("values", [])):
                    feat_w({"iteration": it, "feature": fname,
                            "stat": "value", "value_idx": idx, "value": v})


def refine_gepa(r: Refiner, prog_w, iter_w, scal_w) -> None:
    for it, ck in list_checkpoints(r.src):
        r.counts["checkpoints"] += 1
        prog_dir = ck / "programs"
        if prog_dir.is_dir():
            for jf in sorted(prog_dir.iterdir()):
                if jf.suffix != ".json":
                    continue
                rec = json.loads(jf.read_text())
                r.emit_program(prog_w, rec, status="accepted",
                               source="checkpoint.programs")

        meta = json.loads((ck / "metadata.json").read_text())
        scal_w({"iteration": it, "key": "best_program_id",
                "value": meta.get("best_program_id")})
        scal_w({"iteration": it, "key": "last_iteration",
                "value": meta.get("last_iteration")})

        gp = ck / "gepa_metadata.json"
        if gp.exists():
            g = json.loads(gp.read_text())
            scal_w({"iteration": it, "key": "initial_program_id",
                    "value": g.get("initial_program_id")})
            for slot, pid in enumerate(g.get("elite_pool", [])):
                iter_w({"iteration": it, "role": "elite_pool",
                        "slot_key": slot, "program_id": pid, "value": None})
            for metric, payload in (g.get("metric_best") or {}).items():
                pid, val = payload if isinstance(payload, list) else (None, payload)
                iter_w({"iteration": it, "role": "metric_best",
                        "slot_key": metric, "program_id": pid, "value": val})
            for metric, pids in (g.get("program_at_metric_front") or {}).items():
                for slot, pid in enumerate(pids or []):
                    iter_w({"iteration": it, "role": "metric_front",
                            "slot_key": f"{metric}[{slot}]",
                            "program_id": pid, "value": metric})
            # rejection_history: snapshot the 20-entry sliding window AND merge
            # the program records (deduped by UUID).
            for slot, rec in enumerate(g.get("rejection_history") or []):
                pid = rec.get("id")
                iter_w({"iteration": it, "role": "rejection_history",
                        "slot_key": slot, "program_id": pid, "value": None})
                r.emit_program(prog_w, rec, status="rejected",
                               source="gepa.rejection_history")


def refine_evox(r: Refiner, prog_w, iter_w, scal_w,
                evox_search_w, evox_failed_w) -> None:
    for it, ck in list_checkpoints(r.src):
        r.counts["checkpoints"] += 1
        prog_dir = ck / "programs"
        if prog_dir.is_dir():
            for jf in sorted(prog_dir.iterdir()):
                if jf.suffix != ".json":
                    continue
                rec = json.loads(jf.read_text())
                r.emit_program(prog_w, rec, status="accepted",
                               source="checkpoint.programs")
                iter_w({"iteration": it, "role": "population_member",
                        "slot_key": rec.get("id"),
                        "program_id": rec.get("id"), "value": None})

        meta = json.loads((ck / "metadata.json").read_text())
        scal_w({"iteration": it, "key": "best_program_id",
                "value": meta.get("best_program_id")})
        scal_w({"iteration": it, "key": "last_iteration",
                "value": meta.get("last_iteration")})

    # the search/ tree (per iteration, independent of checkpoints)
    search = r.src / "search"
    if search.is_dir():
        for d in sorted(search.iterdir(),
                        key=lambda p: int(p.name.split("_")[1])
                        if p.name.startswith("iteration_") else -1):
            if not d.name.startswith("iteration_"):
                continue
            it = int(d.name.split("_")[1])
            code_path = d / "code.py"
            code_sha = None
            if code_path.exists():
                code_sha = write_blob(r.blobs, code_path.read_bytes(), "txt")
            md = {}
            md_path = d / "metadata.json"
            if md_path.exists():
                md = json.loads(md_path.read_text())
            evox_search_w({"iteration": it,
                           "is_fallback": md.get("is_fallback"),
                           "code_sha256": code_sha})

            fa_path = d / "failed_attempts.json"
            if fa_path.exists():
                fa = json.loads(fa_path.read_text())
                for entry in fa.get("failed_attempts", []):
                    sol = entry.get("solution") or ""
                    sol_sha = (write_blob(r.blobs, sol.encode("utf-8"), "txt")
                               if sol else None)
                    evox_failed_w({
                        "iteration": it,
                        "attempt_number": entry.get("attempt_number"),
                        "solution_iter": entry.get("solution_iter"),
                        "stage": entry.get("stage"),
                        "error": entry.get("error"),
                        "program_id": entry.get("program_id"),
                        "solution_sha256": sol_sha,
                    })


SHINKA_PROG_COLS = (
    "id", "language", "parent_id", "archive_inspiration_ids",
    "top_k_inspiration_ids", "generation", "timestamp",
    "combined_score", "public_metrics", "private_metrics",
    "text_feedback", "complexity", "embedding", "embedding_pca_2d",
    "embedding_pca_3d", "embedding_cluster_id", "correct",
    "children_count", "metadata", "migration_history", "island_idx",
    "system_prompt_id",
)
SHINKA_JSON_COLS = {
    "archive_inspiration_ids", "top_k_inspiration_ids",
    "public_metrics", "private_metrics", "embedding",
    "embedding_pca_2d", "embedding_pca_3d", "metadata",
    "migration_history",
}


def _maybe_json(v: Any) -> Any:
    if not isinstance(v, str) or not v:
        return v
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


def _shinka_to_canonical(row: sqlite3.Row) -> tuple[dict, dict]:
    """Translate a shinka programs-table row into (canonical_rec, extras).

    The canonical record matches PROGRAM_FIELDS_PASSTHROUGH so emit_program
    can route it through the shared blob/dedup pipeline. Extras carry
    shinka-only columns (complexity, embeddings, code_diff, etc).
    """
    public_metrics = _maybe_json(row["public_metrics"])
    private_metrics = _maybe_json(row["private_metrics"])
    archive_insp = _maybe_json(row["archive_inspiration_ids"])
    top_k_insp = _maybe_json(row["top_k_inspiration_ids"])
    metadata = _maybe_json(row["metadata"])

    metrics: dict[str, Any] = {}
    if isinstance(public_metrics, dict):
        metrics.update(public_metrics)
    if row["combined_score"] is not None:
        metrics.setdefault("combined_score", row["combined_score"])
    if isinstance(private_metrics, dict) and private_metrics:
        metrics["_private"] = private_metrics

    other_ctx: list[Any] = []
    if isinstance(archive_insp, list):
        other_ctx.extend(archive_insp)
    if isinstance(top_k_insp, list):
        other_ctx.extend(top_k_insp)

    canonical = {
        "id": row["id"],
        "language": row["language"],
        "metrics": metrics,
        "iteration_found": row["generation"],
        "parent_id": row["parent_id"] or None,
        "other_context_ids": other_ctx,
        "parent_info": None,
        "context_info": None,
        "timestamp": row["timestamp"],
        "metadata": metadata if isinstance(metadata, dict) else {},
        "generation": row["generation"],
        "solution": row["code"] if isinstance(row["code"], str) else None,
        "prompts": None,
        "artifacts": None,
    }

    extras = {
        "complexity": row["complexity"],
        "correct": bool(row["correct"]) if row["correct"] is not None else None,
        "children_count": row["children_count"],
        "island_idx": row["island_idx"],
        "system_prompt_id": row["system_prompt_id"],
        "text_feedback": row["text_feedback"] or None,
        "embedding": _maybe_json(row["embedding"]),
        "embedding_pca_2d": _maybe_json(row["embedding_pca_2d"]),
        "embedding_pca_3d": _maybe_json(row["embedding_pca_3d"]),
        "embedding_cluster_id": row["embedding_cluster_id"],
        "migration_history": _maybe_json(row["migration_history"]),
        "archive_inspiration_ids": archive_insp,
        "top_k_inspiration_ids": top_k_insp,
        "public_metrics": public_metrics,
        "private_metrics": private_metrics,
    }
    return canonical, extras


def refine_shinkaevolve(r: Refiner, prog_w, iter_w, scal_w,
                        events_w, attempts_w, gen_metrics_w, priv_w) -> None:
    db_path = r.src / "programs.sqlite"
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    cols = ", ".join(SHINKA_PROG_COLS)
    for row in con.execute(
            f"SELECT {cols}, code, code_diff FROM programs "
            "ORDER BY generation, timestamp"):
        canonical, extras = _shinka_to_canonical(row)
        diff = row["code_diff"]
        if isinstance(diff, str) and diff:
            extras["code_diff_sha256"] = write_blob(
                r.blobs, diff.encode("utf-8"), "txt")
        else:
            extras["code_diff_sha256"] = None
        r.emit_program(prog_w, canonical, status="accepted",
                       source="sqlite.programs", extras=extras)

    archive_n = 0
    for slot, row in enumerate(con.execute(
            "SELECT program_id FROM archive")):
        iter_w({"iteration": None, "role": "archive",
                "slot_key": slot, "program_id": row["program_id"],
                "value": None})
        archive_n += 1
    r.counts["archive"] = archive_n

    events_n = 0
    for row in con.execute(
            "SELECT id, generation, status, source_job_id, details, "
            "created_at FROM generation_event_log"):
        d = dict(row)
        d["details"] = _maybe_json(d.get("details"))
        events_w(d)
        events_n += 1
    r.counts["events"] = events_n

    attempts_n = 0
    for row in con.execute(
            "SELECT id, generation, stage, status, details, created_at "
            "FROM attempt_log"):
        d = dict(row)
        d["details"] = _maybe_json(d.get("details"))
        attempts_w(d)
        attempts_n += 1
    r.counts["attempts"] = attempts_n

    for row in con.execute("SELECT key, value FROM metadata_store"):
        scal_w({"iteration": None, "key": row["key"],
                "value": _maybe_json(row["value"])})

    con.close()

    gen_metrics_n = 0
    gen_dirs = []
    for d in r.src.iterdir():
        if not d.is_dir() or not d.name.startswith("gen_"):
            continue
        try:
            gen_dirs.append((int(d.name.split("_", 1)[1]), d))
        except ValueError:
            continue
    gen_dirs.sort()
    for gen, d in gen_dirs:
        m = d / "results" / "metrics.json"
        if not m.exists():
            continue
        try:
            metrics = json.loads(m.read_text())
        except (ValueError, OSError):
            metrics = {"_parse_error": True}
        gen_metrics_w({"generation": gen, "metrics": metrics})
        gen_metrics_n += 1
    r.counts["gen_metrics"] = gen_metrics_n

    priv_path = r.src / "private_eval_lineage.json"
    priv_n = 0
    if priv_path.exists():
        try:
            data = json.loads(priv_path.read_text())
        except (ValueError, OSError):
            data = None
        if isinstance(data, dict):
            for entry in data.get("lineage") or []:
                priv_w(entry)
                priv_n += 1
    r.counts["private_lineage"] = priv_n


# ---------- top-level ----------

def refine(src: Path, dst: Path) -> dict:
    if dst.exists():
        raise SystemExit(f"refusing to overwrite existing {dst}")
    backend = detect_backend(src)
    dst.mkdir(parents=True)

    refiner = Refiner(src=src, dst=dst, backend=backend)

    f_prog, prog_w = jsonl_writer(dst / "programs.jsonl")
    f_iter, iter_w = jsonl_writer(dst / "iterations.jsonl")
    f_scal, scal_w = jsonl_writer(dst / "iter_scalars.jsonl")
    extra_files = [f_prog, f_iter, f_scal]

    try:
        if backend in ("openevolve_native", "gepa_native", "evox"):
            for name in ("run_info.json", "run_config.yaml"):
                p = src / name
                if p.exists():
                    shutil.copy2(p, dst / name)
            for name in ("best", "logs", "analysis"):
                p = src / name
                if p.is_dir():
                    os.symlink(p.resolve(), dst / name)

            if backend == "openevolve_native":
                f_feat, feat_w = jsonl_writer(dst / "feature_stats.jsonl")
                extra_files.append(f_feat)
                refine_openevolve(refiner, prog_w, iter_w, scal_w, feat_w)
            elif backend == "gepa_native":
                refine_gepa(refiner, prog_w, iter_w, scal_w)
            else:  # evox
                f_es, es_w = jsonl_writer(dst / "evox_search.jsonl")
                f_ef, ef_w = jsonl_writer(dst / "evox_failed_attempts.jsonl")
                extra_files.extend([f_es, f_ef])
                refine_evox(refiner, prog_w, iter_w, scal_w, es_w, ef_w)
        elif backend == "shinkaevolve":
            for name in ("evaluate.py", "init_program.cpp",
                         "private_eval_lineage.json"):
                p = src / name
                if p.exists():
                    shutil.copy2(p, dst / name)
            for name in ("logs", "meta", "evolution_run.log",
                         "bandit_state.pkl"):
                p = src / name
                if p.exists():
                    os.symlink(p.resolve(), dst / name)

            f_evt, evt_w = jsonl_writer(dst / "shinka_events.jsonl")
            f_att, att_w = jsonl_writer(dst / "shinka_attempts.jsonl")
            f_gm, gm_w = jsonl_writer(dst / "shinka_gen_metrics.jsonl")
            f_pl, pl_w = jsonl_writer(dst / "shinka_private_lineage.jsonl")
            extra_files.extend([f_evt, f_att, f_gm, f_pl])
            refine_shinkaevolve(refiner, prog_w, iter_w, scal_w,
                                evt_w, att_w, gm_w, pl_w)
        else:
            raise RuntimeError(f"unknown backend {backend}")
    finally:
        for f in extra_files:
            f.close()

    meta = {
        "backend": backend,
        "source": str(src.resolve()),
        "refined_at": None,
        "counts": refiner.counts,
        "schema_version": 1,
    }
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="source run directory")
    ap.add_argument("dst", type=Path, help="destination (must not exist)")
    args = ap.parse_args(argv)
    meta = refine(args.src.resolve(), args.dst.resolve())
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
