"""Three-way classification of recycled lines: literal / tuning / trivial.

`detect_cycling.py` reports cycling rate but conflates two things:
  - literal recycling: a removed line is later re-added byte-for-byte
  - tuning recycling: a removed line's *skeleton* is later re-added with
    different numeric constants (e.g. `A = 15.0 + 1.8*turn` ↔
    `A = 14.0 + 2.0*turn`).

The `--collapse-numbers` flag in `detect_cycling.py` makes the second case
match too, but it doesn't separate them. This module does the split:

    literal_recycling  — same skeleton AND same numbers
    tuning_recycling   — same skeleton, different numbers
    trivial_recycling  — comment / whitespace-only differences

Output (under <run_dir>/analysis/):
    cycle_classes.json           per-event records
    cycle_classes.programs.csv   per-program counts + ratios
    cycle_classes.summary.json   aggregate counts + recycling-by-class

Usage:
    uv run python -m evo_replay.cycling.classify_cycles <run_dir> [--csv <prefix>]
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import statistics
import sys
from collections import defaultdict
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


def _is_comment(s: str) -> bool:
    return s.startswith(("/*", "*/", "*", "//", "#"))


@dataclass
class CycleEvent:
    """One recycled-line event."""
    line: str
    klass: str  # "literal" | "tuning" | "trivial"
    removed_iter: int
    removed_program: str
    added_iter: int
    added_program: str
    span: int  # added_iter - removed_iter


def _classify_pair(removed: str, added: str) -> str:
    """Return 'literal' | 'tuning' | 'trivial'.

    Caller has already checked the collapsed-skeletons match.
    """
    if removed == added:
        return "literal"
    # Skeletons match but content differs → look at why.
    if _is_comment(removed) or _is_comment(added):
        return "trivial"
    # Strip whitespace; if only whitespace differs, it's trivial.
    if "".join(removed.split()) == "".join(added.split()):
        return "trivial"
    return "tuning"


def classify_cycles(programs: Dict[str, Dict[str, Any]],
                    *,
                    min_line_len: int = 5) -> Tuple[List[CycleEvent], Dict[str, Dict[str, int]]]:
    """Walk every parent→child diff, build line-event histories, classify cycles.

    Returns:
        events: list of CycleEvent
        per_program: {program_id: {literal_added, tuning_added, trivial_added,
                                   total_added, recycle_ratio_*}}
    """
    # Order programs by iteration so "earlier" is well-defined.
    ordered = sorted(
        programs.values(),
        key=lambda p: int(p.get("iteration_found") or 0),
    )

    # Map skeleton → list of (kind, full_line, iter, program_id)
    # 'kind' is "removed" or "added"; we need full_line to classify pairs.
    history: Dict[str, List[Tuple[str, str, int, str]]] = defaultdict(list)

    # Per-program added-line tracking, for ratios
    per_prog: Dict[str, Dict[str, int]] = {}
    # Track which (skeleton, added_program) we've already credited
    counted: set[Tuple[str, str]] = set()

    events: List[CycleEvent] = []

    for child in ordered:
        par_id = child.get("parent_id")
        if not par_id or par_id not in programs:
            continue
        parent = programs[par_id]
        parent_code = parent.get("solution") or ""
        child_code = child.get("solution") or ""
        if not (parent_code and child_code):
            continue

        cit = int(child.get("iteration_found") or 0)
        cid = str(child.get("id") or "")

        diff = list(difflib.unified_diff(
            parent_code.splitlines(), child_code.splitlines(), n=0,
        ))

        rem_lines: List[str] = []
        add_lines: List[str] = []
        for L in diff:
            if L.startswith(("---", "+++", "@@")):
                continue
            if L.startswith("-"):
                n = _normalise(L[1:])
                if n and not _is_trivial(n) and not _is_numeric_only(n) and len(n) >= min_line_len:
                    rem_lines.append(n)
            elif L.startswith("+"):
                n = _normalise(L[1:])
                if n and not _is_trivial(n) and not _is_numeric_only(n) and len(n) >= min_line_len:
                    add_lines.append(n)

        # Record removals (no classification needed yet)
        for line in rem_lines:
            history[_collapsed(line)].append(("removed", line, cit, cid))

        # Per-program slot for this child
        per_prog[cid] = {
            "iteration": cit,
            "total_added": len(add_lines),
            "literal_added": 0,
            "tuning_added": 0,
            "trivial_added": 0,
            "model": (child.get("metadata") or {}).get("model_name") or "?",
            "score": (child.get("metrics") or {}).get("combined_score"),
        }

        # Classify each added line against the most recent prior removal
        # with the same skeleton.
        for line in add_lines:
            skel = _collapsed(line)
            prior_rem = [h for h in history[skel] if h[0] == "removed" and h[2] < cit]
            if not prior_rem:
                # Also record this as an addition (so future removals can pair)
                history[skel].append(("added", line, cit, cid))
                continue
            # Most recent prior removal:
            kind, prev_line, prev_iter, prev_pid = max(prior_rem, key=lambda h: h[2])
            klass = _classify_pair(prev_line, line)
            key = (skel, cid)
            if key not in counted:
                counted.add(key)
                if klass == "literal":
                    per_prog[cid]["literal_added"] += 1
                elif klass == "tuning":
                    per_prog[cid]["tuning_added"] += 1
                else:
                    per_prog[cid]["trivial_added"] += 1
                events.append(CycleEvent(
                    line=line,
                    klass=klass,
                    removed_iter=prev_iter,
                    removed_program=prev_pid,
                    added_iter=cit,
                    added_program=cid,
                    span=cit - prev_iter,
                ))
            # Always record the addition
            history[skel].append(("added", line, cit, cid))

    # Compute ratios
    for pid, rec in per_prog.items():
        tot = rec["total_added"] or 1
        rec["literal_ratio"] = rec["literal_added"] / tot
        rec["tuning_ratio"] = rec["tuning_added"] / tot
        rec["trivial_ratio"] = rec["trivial_added"] / tot
        rec["any_ratio"] = (rec["literal_added"] + rec["tuning_added"] + rec["trivial_added"]) / tot

    return events, per_prog


def summarize(events: List[CycleEvent], per_prog: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    n_lit = sum(1 for e in events if e.klass == "literal")
    n_tun = sum(1 for e in events if e.klass == "tuning")
    n_triv = sum(1 for e in events if e.klass == "trivial")
    total_added = sum(p["total_added"] for p in per_prog.values())
    spans = [e.span for e in events]

    return {
        "n_edits": len(per_prog),
        "n_recycle_events": len(events),
        "events_by_class": {"literal": n_lit, "tuning": n_tun, "trivial": n_triv},
        "total_added_lines": total_added,
        "share_of_added_lines": {
            "literal": n_lit / max(total_added, 1),
            "tuning": n_tun / max(total_added, 1),
            "trivial": n_triv / max(total_added, 1),
            "any": (n_lit + n_tun + n_triv) / max(total_added, 1),
        },
        "median_per_edit_ratios": {
            "literal": statistics.median(p["literal_ratio"] for p in per_prog.values()),
            "tuning": statistics.median(p["tuning_ratio"] for p in per_prog.values()),
            "trivial": statistics.median(p["trivial_ratio"] for p in per_prog.values()),
            "any": statistics.median(p["any_ratio"] for p in per_prog.values()),
        },
        "span_stats": {
            "median": statistics.median(spans) if spans else 0,
            "mean": statistics.mean(spans) if spans else 0.0,
            "max": max(spans) if spans else 0,
        },
    }


def write_outputs(events: List[CycleEvent],
                  per_prog: Dict[str, Dict[str, int]],
                  summary: Dict[str, Any],
                  out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Per-event JSON
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps([asdict(e) for e in events], indent=2))

    # Per-program CSV
    csv_path = out_prefix.with_suffix(".programs.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iteration", "program_id", "model", "score",
            "total_added", "literal_added", "tuning_added", "trivial_added",
            "literal_ratio", "tuning_ratio", "trivial_ratio", "any_ratio",
        ])
        for pid, rec in sorted(per_prog.items(), key=lambda kv: kv[1]["iteration"]):
            w.writerow([
                rec["iteration"], pid[:12], rec["model"],
                f"{rec['score']:.6f}" if isinstance(rec["score"], (int, float)) else "",
                rec["total_added"],
                rec["literal_added"], rec["tuning_added"], rec["trivial_added"],
                f"{rec['literal_ratio']:.4f}",
                f"{rec['tuning_ratio']:.4f}",
                f"{rec['trivial_ratio']:.4f}",
                f"{rec['any_ratio']:.4f}",
            ])

    summary_path = out_prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


def print_report(summary: Dict[str, Any]) -> None:
    s = summary["share_of_added_lines"]
    m = summary["median_per_edit_ratios"]
    print(f"edits: {summary['n_edits']}  recycle events: {summary['n_recycle_events']}  "
          f"total added lines: {summary['total_added_lines']}")
    print(f"  share of all added lines:")
    print(f"    literal recycling: {s['literal']:>5.1%}  "
          f"(real backtracking — same line, same numbers)")
    print(f"    tuning recycling : {s['tuning']:>5.1%}  "
          f"(same skeleton, different numbers)")
    print(f"    trivial recycling: {s['trivial']:>5.1%}  "
          f"(comments/whitespace)")
    print(f"    any recycling    : {s['any']:>5.1%}")
    print(f"  median per-edit ratios:")
    print(f"    literal {m['literal']:>5.1%}  tuning {m['tuning']:>5.1%}  "
          f"trivial {m['trivial']:>5.1%}  any {m['any']:>5.1%}")
    print(f"  recycling span (iterations between removal and re-add):")
    sp = summary["span_stats"]
    print(f"    median {sp['median']}  mean {sp['mean']:.1f}  max {sp['max']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--csv", type=Path, default=None,
                    help="Output prefix (no extension). "
                         "Defaults to <run_dir>/analysis/cycle_classes")
    ap.add_argument("--min-line-len", type=int, default=5)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    out_prefix = args.csv or (run_dir / "analysis" / "cycle_classes")

    progs = load_programs(run_dir)
    if not progs:
        sys.exit("No programs found.")

    events, per_prog = classify_cycles(progs, min_line_len=args.min_line_len)
    summary = summarize(events, per_prog)

    print_report(summary)
    write_outputs(events, per_prog, summary, out_prefix)


if __name__ == "__main__":
    main()
