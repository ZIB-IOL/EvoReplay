"""Shared checkpoint loading + lineage utilities.

Two on-disk layouts are supported transparently:

A) **Refined** (preferred, produced by skydiscover/scripts/refine_outputs.py):

    <run_dir>/
        meta.json
        run_config.yaml          (canonical 3 backends only)
        programs.jsonl           one row per unique program; canonical fields
                                 (incl. solution_sha256, prompts_sha256)
        iterations.jsonl
        iter_scalars.jsonl
        blobs/<sha[:2]>/<sha>.{txt,json}   content-addressed code & prompts
        best/, logs/, analysis/  (canonical 3 backends; symlinks or copies)

B) **Raw** (legacy skydiscover output):

    <run_dir>/
        run_config.yaml
        run_info.json
        checkpoints/checkpoint_<N>/
            programs/<uuid>.json     # one per program seen up to checkpoint N
            best_program.{cpp,py}
            best_program_info.json
            metadata.json
        search/iteration_<N>/
            code.py | code.cpp
            metadata.json
            failed_attempts.json
        best/best_program.{cpp,py}
        logs/

`load_programs(run_dir)` returns the same {pid: program_record} shape from
either layout. For the refined layout it dereferences the content-addressed
blobs and re-injects them as `program["solution"]` / `program["prompts"]`,
so downstream callers do not need to know which layout they are reading.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml  # type: ignore[import-untyped]


_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)$")

CANONICAL_ANALYSIS_DIRNAME = "solution_length"


def detect_language(run_dir: Path) -> str:
    """Return the program language ('python' / 'cpp').

    Checks `run_config.yaml` first, then the refined `programs.jsonl`'s first
    row, then `best/best_program.*` as a last resort.
    """
    cfg_path = run_dir / "run_config.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            lang = cfg.get("language", "")
            if isinstance(lang, str) and lang:
                return lang.lower()
        except Exception:
            pass
    # Refined-layout fallback: peek at first programs.jsonl row.
    pj = run_dir / "programs.jsonl"
    if pj.exists():
        try:
            with pj.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    lang = rec.get("language", "")
                    if isinstance(lang, str) and lang:
                        return lang.lower()
                    break
        except Exception:
            pass
    # Final fallback: peek at best/best_program.*
    best = run_dir / "best"
    if best.exists():
        if (best / "best_program.cpp").exists():
            return "cpp"
        if (best / "best_program.py").exists():
            return "python"
    return ""


def iter_checkpoints(run_dir: Path) -> Iterator[Tuple[int, Path]]:
    """Yield (checkpoint_number, checkpoint_dir) sorted ascending. Raw layout only."""
    ck_root = run_dir / "checkpoints"
    if not ck_root.exists():
        return
    items: List[Tuple[int, Path]] = []
    for child in ck_root.iterdir():
        m = _CHECKPOINT_RE.match(child.name)
        if m and child.is_dir():
            items.append((int(m.group(1)), child))
    items.sort(key=lambda x: x[0])
    yield from items


def _resolve_blob(blobs_dir: Path, sha: str, ext: str) -> Optional[str]:
    """Read a content-addressed blob; return None if missing/unreadable."""
    if not sha or not isinstance(sha, str):
        return None
    blob = blobs_dir / sha[:2] / f"{sha}.{ext}"
    if not blob.exists():
        return None
    try:
        return blob.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_programs_refined(run_dir: Path, jsonl: Path) -> Dict[str, Dict[str, Any]]:
    blobs_dir = run_dir / "blobs"
    out: Dict[str, Dict[str, Any]] = {}
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id")
            if not pid:
                continue
            pid = str(pid)

            # Resolve solution blob → inject as 'solution' for downstream compat.
            if not rec.get("solution"):
                sol = _resolve_blob(blobs_dir, rec.get("solution_sha256"), "txt")
                if sol is not None:
                    rec["solution"] = sol

            # Resolve prompts blob → parse JSON → inject as 'prompts'.
            if not rec.get("prompts"):
                raw = _resolve_blob(blobs_dir, rec.get("prompts_sha256"), "json")
                if raw is not None:
                    try:
                        rec["prompts"] = json.loads(raw)
                    except json.JSONDecodeError:
                        pass

            out[pid] = rec
    return out


def _load_programs_raw(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for _, ck_dir in iter_checkpoints(run_dir):
        prog_dir = ck_dir / "programs"
        if not prog_dir.exists():
            continue
        for path in sorted(prog_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            pid = str(data.get("id"))
            if not pid:
                continue
            out[pid] = data
    return out


def load_programs(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return {program_id: program_record}, deduped.

    Auto-detects the run layout: refined (preferred, via programs.jsonl) or
    raw (via checkpoints/checkpoint_*/programs/*.json).
    """
    refined = run_dir / "programs.jsonl"
    if refined.exists():
        return _load_programs_refined(run_dir, refined)
    return _load_programs_raw(run_dir)


def primary_score(program: Dict[str, Any]) -> Optional[float]:
    """Pick the score we treat as canonical (combined_score → overall_score → score)."""
    metrics = program.get("metrics") or {}
    for key in ("combined_score", "overall_score", "score"):
        v = metrics.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def is_valid_program(program: Dict[str, Any]) -> bool:
    """Mirror of static_analysis/analyze_hyperparameter_counts.is_valid_program."""
    metrics = program.get("metrics") or {}
    if not metrics:
        return False
    judge = metrics.get("judge_result")
    if isinstance(judge, str) and judge.upper() not in {"ACCEPTED", "OK", "PASSED", "WRONG_ANSWER"}:
        # WRONG_ANSWER programs still ran and are scored; only hard-fail ones are dropped
        if judge.upper() in {"COMPILE_ERROR", "RUNTIME_ERROR", "TIME_LIMIT_EXCEEDED",
                              "MEMORY_LIMIT_EXCEEDED", "INTERNAL_ERROR"}:
            return False
    if metrics.get("error"):
        return False
    if primary_score(program) is None:
        return False
    return True


@dataclass
class LineageNode:
    program_id: str
    iteration: int
    parent_id: Optional[str]
    score: Optional[float]
    program: Dict[str, Any] = field(repr=False, default_factory=dict)


def lineage(programs: Dict[str, Dict[str, Any]], target_id: str) -> List[LineageNode]:
    """Walk parent_id chain from `target_id` back to a root, cycle-safe."""
    chain: List[LineageNode] = []
    seen: set[str] = set()
    cur: Optional[str] = target_id
    while cur and cur in programs and cur not in seen:
        seen.add(cur)
        prog = programs[cur]
        chain.append(LineageNode(
            program_id=cur,
            iteration=int(prog.get("iteration_found") or 0),
            parent_id=prog.get("parent_id"),
            score=primary_score(prog),
            program=prog,
        ))
        cur = prog.get("parent_id")
    return list(reversed(chain))


def best_so_far_targets(
    programs: Dict[str, Dict[str, Any]],
    *,
    require_valid: bool = True,
) -> List[Dict[str, Any]]:
    """Programs whose score strictly improved the running max, in iteration order."""
    ordered = sorted(
        programs.values(),
        key=lambda p: int(p.get("iteration_found") or 0),
    )
    targets: List[Dict[str, Any]] = []
    best = float("-inf")
    for prog in ordered:
        if require_valid and not is_valid_program(prog):
            continue
        s = primary_score(prog)
        if s is None:
            continue
        if s > best:
            best = s
            targets.append(prog)
    return targets


__all__ = [
    "detect_language",
    "iter_checkpoints",
    "load_programs",
    "primary_score",
    "is_valid_program",
    "LineageNode",
    "lineage",
    "best_so_far_targets",
]
