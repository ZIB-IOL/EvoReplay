"""Classify each parent→child edit into tuning / structural / comment.

Strict-regex cycle detection (detect_cycling.py) only catches very simple
``name = NUM;`` lines as hparam tuning, so it under-counts the real tuning
share of edits — most evolution runs tune coefficients inside arithmetic
formulas like ``double A = 15.0 + 2.2 * turn + (turn > 75 ? ... : 0.0);``
which the regex flags as structural.

This module uses a different signal: pair a removed line with an added
line whose **number-collapsed skeleton** matches. If both `_NUMBERS_RE`
substitute to the same string, they're "same line, different numbers" —
i.e. pure coefficient tuning.

Usage:
    uv run python -m evo_replay.cycling.classify_edits <run_dir>
        [--csv <prefix>]   # default: <run_dir>/analysis/edit_classification

Outputs (under <run_dir>/analysis/ by default):
    edit_classification.json            full record per edit
    edit_classification.csv             one row per edit
    edit_classification.summary.json    aggregate counts + per-edit buckets
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evo_replay.cycling.detect_cycling import (
    _NUMBERS_RE,
    _is_numeric_only,
    _is_trivial,
    _normalise,
    load_programs,
)


def _collapsed(line: str) -> str:
    return _NUMBERS_RE.sub("#", line)


def _is_comment(stripped: str) -> bool:
    """Treat C/C++ comment lines as their own bucket."""
    return stripped.startswith(("/*", "*/", "*", "//"))


@dataclass
class EditClassification:
    program_id: str
    parent_id: str
    iteration: int
    parent_iteration: int
    score: Optional[float]
    parent_score: Optional[float]
    score_delta: Optional[float]
    tuning_lines: int
    structural_lines: int
    comment_lines: int
    n_added: int
    n_removed: int
    tuning_pairs: List[Tuple[str, str]] = field(default_factory=list)
    structural_added: List[str] = field(default_factory=list)
    structural_removed: List[str] = field(default_factory=list)

    @property
    def total_code(self) -> int:
        return self.tuning_lines + self.structural_lines

    @property
    def tuning_fraction(self) -> float:
        return self.tuning_lines / self.total_code if self.total_code else 0.0


def classify_edit(
    parent_code: str,
    child_code: str,
) -> Dict[str, Any]:
    """Return the bucket counts + paired tuning examples for one edit."""
    rem_norm: List[str] = []
    add_norm: List[str] = []
    for line in difflib.unified_diff(
        parent_code.splitlines(), child_code.splitlines(), n=0,
    ):
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            n = _normalise(line[1:])
            if n and not _is_trivial(n) and not _is_numeric_only(n):
                rem_norm.append(n)
        elif line.startswith("+"):
            n = _normalise(line[1:])
            if n and not _is_trivial(n) and not _is_numeric_only(n):
                add_norm.append(n)

    rem_code = [s for s in rem_norm if not _is_comment(s)]
    add_code = [s for s in add_norm if not _is_comment(s)]
    rem_comments = [s for s in rem_norm if _is_comment(s)]
    add_comments = [s for s in add_norm if _is_comment(s)]

    rem_skel = [_collapsed(s) for s in rem_code]
    add_skel = [_collapsed(s) for s in add_code]

    used_r = [False] * len(rem_code)
    used_a = [False] * len(add_code)
    pairs: List[Tuple[str, str]] = []

    # Greedy pairing: same skeleton, different actual content (otherwise it's
    # not an edit at all — it would be in a context line).
    for i, sa in enumerate(add_skel):
        for j, sr in enumerate(rem_skel):
            if used_r[j]:
                continue
            if sa == sr and rem_code[j] != add_code[i]:
                used_r[j] = True
                used_a[i] = True
                pairs.append((rem_code[j], add_code[i]))
                break

    structural_removed = [rem_code[j] for j in range(len(rem_code)) if not used_r[j]]
    structural_added = [add_code[i] for i in range(len(add_code)) if not used_a[i]]

    return {
        "tuning_lines": 2 * len(pairs),
        "structural_lines": len(structural_removed) + len(structural_added),
        "comment_lines": len(rem_comments) + len(add_comments),
        "tuning_pairs": pairs,
        "structural_removed": structural_removed,
        "structural_added": structural_added,
        "n_removed": len(rem_norm),
        "n_added": len(add_norm),
    }


def classify_run(programs: Dict[str, Dict[str, Any]]) -> List[EditClassification]:
    out: List[EditClassification] = []
    for pid, child in programs.items():
        par_id = child.get("parent_id")
        if not par_id or par_id not in programs:
            continue
        parent = programs[par_id]
        if not (parent.get("solution") and child.get("solution")):
            continue
        r = classify_edit(parent["solution"], child["solution"])

        def _score(p: Dict[str, Any]) -> Optional[float]:
            v = (p.get("metrics") or {}).get("combined_score")
            return float(v) if isinstance(v, (int, float)) else None

        s_c, s_p = _score(child), _score(parent)
        delta = (s_c - s_p) if (s_c is not None and s_p is not None) else None
        out.append(EditClassification(
            program_id=pid,
            parent_id=par_id,
            iteration=int(child.get("iteration_found") or 0),
            parent_iteration=int(parent.get("iteration_found") or 0),
            score=s_c,
            parent_score=s_p,
            score_delta=delta,
            tuning_lines=r["tuning_lines"],
            structural_lines=r["structural_lines"],
            comment_lines=r["comment_lines"],
            n_added=r["n_added"],
            n_removed=r["n_removed"],
            tuning_pairs=r["tuning_pairs"],
            structural_added=r["structural_added"],
            structural_removed=r["structural_removed"],
        ))
    out.sort(key=lambda e: e.iteration)
    return out


def _bucket(fraction: float) -> str:
    if fraction >= 0.9: return "pure-tuning"
    if fraction >= 0.6: return "mostly-tuning"
    if fraction <= 0.1: return "pure-structural"
    if fraction <= 0.4: return "mostly-structural"
    return "mixed"


def summarize(edits: List[EditClassification]) -> Dict[str, Any]:
    tot_t = sum(e.tuning_lines for e in edits)
    tot_s = sum(e.structural_lines for e in edits)
    tot_c = sum(e.comment_lines for e in edits)
    tot = tot_t + tot_s + tot_c
    code = tot_t + tot_s

    bucket_counts = {k: 0 for k in
                     ("pure-tuning", "mostly-tuning", "mixed",
                      "mostly-structural", "pure-structural")}
    fractions = []
    for e in edits:
        if e.total_code == 0:
            continue
        fractions.append(e.tuning_fraction)
        bucket_counts[_bucket(e.tuning_fraction)] += 1

    score_up = [e for e in edits if e.score_delta is not None and e.score_delta > 0]
    score_dn = [e for e in edits if e.score_delta is not None and e.score_delta < 0]

    def _frac(es):
        t = sum(e.tuning_lines for e in es)
        s = sum(e.structural_lines for e in es)
        return (t / (t + s)) if (t + s) else None

    return {
        "edits": len(edits),
        "lines_total": tot,
        "tuning_lines": tot_t,
        "structural_lines": tot_s,
        "comment_lines": tot_c,
        "tuning_share_of_all_lines": (tot_t / tot) if tot else 0.0,
        "tuning_share_of_code_lines": (tot_t / code) if code else 0.0,
        "structural_share_of_code_lines": (tot_s / code) if code else 0.0,
        "median_per_edit_tuning_fraction":
            statistics.median(fractions) if fractions else 0.0,
        "buckets": bucket_counts,
        "score_up_tuning_fraction": _frac(score_up),
        "score_down_tuning_fraction": _frac(score_dn),
        "n_score_up": len(score_up),
        "n_score_down": len(score_dn),
    }


def write_outputs(
    edits: List[EditClassification],
    summary: Dict[str, Any],
    out_prefix: Path,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(
        [asdict(e) for e in edits], indent=2, default=str,
    ))

    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iteration", "parent_iteration", "program_id", "parent_id",
            "score", "parent_score", "score_delta",
            "tuning_lines", "structural_lines", "comment_lines",
            "tuning_fraction", "bucket",
        ])
        for e in edits:
            w.writerow([
                e.iteration, e.parent_iteration, e.program_id[:12], e.parent_id[:12],
                e.score if e.score is not None else "",
                e.parent_score if e.parent_score is not None else "",
                e.score_delta if e.score_delta is not None else "",
                e.tuning_lines, e.structural_lines, e.comment_lines,
                f"{e.tuning_fraction:.4f}",
                _bucket(e.tuning_fraction) if e.total_code else "trivial",
            ])

    summary_path = out_prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


def print_report(summary: Dict[str, Any]) -> None:
    print(f"edits: {summary['edits']}  lines: {summary['lines_total']}")
    print(f"  tuning     : {summary['tuning_lines']:>5} ({summary['tuning_share_of_all_lines']:>5.1%} of all, "
          f"{summary['tuning_share_of_code_lines']:>5.1%} of code)")
    print(f"  structural : {summary['structural_lines']:>5} ({summary['structural_share_of_code_lines']:>5.1%} of code)")
    print(f"  comment    : {summary['comment_lines']:>5}")
    print(f"  median per-edit tuning fraction (code only): "
          f"{summary['median_per_edit_tuning_fraction']:.1%}")
    print(f"  per-edit buckets:")
    for k, v in summary["buckets"].items():
        print(f"    {k:<18} {v:>3}")
    if summary["score_up_tuning_fraction"] is not None:
        print(f"  score-up edits   ({summary['n_score_up']}): "
              f"tuning fraction = {summary['score_up_tuning_fraction']:.1%}")
    if summary["score_down_tuning_fraction"] is not None:
        print(f"  score-down edits ({summary['n_score_down']}): "
              f"tuning fraction = {summary['score_down_tuning_fraction']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output prefix (without extension). "
             "Defaults to <run_dir>/analysis/edit_classification",
    )
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    out_prefix = args.csv or (run_dir / "analysis" / "edit_classification")

    programs = load_programs(run_dir)
    if not programs:
        sys.exit("No programs found in checkpoints.")
    edits = classify_run(programs)
    if not edits:
        sys.exit("No parent→child edits found.")
    summary = summarize(edits)

    print_report(summary)
    write_outputs(edits, summary, out_prefix)


if __name__ == "__main__":
    main()
