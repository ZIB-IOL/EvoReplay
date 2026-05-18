"""detect_cycling.py — Detect line-level cycling in evolution runs.

Cycling occurs when a line is removed in one iteration and re-introduced
in a later iteration (possibly by a different program on a different branch).

Algorithm:
  1. For every program with a parent, compute the unified diff.
  2. Record each removed line and each added line (normalised).
  3. A line that was removed at iteration i and added at iteration j > i
     has "cycled back". We report the cycle with both iterations.
  4. Aggregate per-program: fraction of added lines that are recycled
     from earlier removals, and fraction of removed lines that get
     re-introduced later.

Usage:
    uv run python -m evo_replay.cycling.detect_cycling <run_dir> [options]
    uv run python -m evo_replay.cycling.detect_cycling <run_dir> --min-cycles 2
    uv run python -m evo_replay.cycling.detect_cycling <run_dir> --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evo_replay.core.checkpoints import load_programs

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"//.*$")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/")
_NUMBERS_RE = re.compile(r"[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?")


def _normalise(line: str, collapse_numbers: bool = False) -> str:
    """Strip whitespace and single-line comments for comparison.

    If *collapse_numbers* is True, all numeric literals are replaced with ``#``
    so that lines differing only in a constant (e.g. ``< 10`` vs ``< 8``) are
    treated as the same line.
    """
    line = _COMMENT_RE.sub("", line)
    line = _BLOCK_COMMENT_RE.sub("", line)
    line = line.strip()
    if collapse_numbers:
        line = _NUMBERS_RE.sub("#", line)
    return line


def _is_trivial(line: str) -> bool:
    """Skip lines that are too short or purely structural."""
    if len(line) <= 2:
        return True
    if line in ("{", "}", "};", "});", "else {", "else{", "return;"):
        return True
    if line.startswith("#include"):
        return True
    return False


def _is_numeric_only(line: str) -> bool:
    """True if the line is purely numeric content — no meaningful code structure.

    After stripping all numeric literals, if only punctuation/operators/whitespace
    remain (e.g. ``=``, ``(``, ``,``, ``;``), the line is just a number assignment
    or literal and should be excluded from cycling detection.
    """
    stripped = _NUMBERS_RE.sub("", line)
    # Remove common C/C++ punctuation/operators that surround numbers
    stripped = re.sub(r"[=+\-*/()<>,;:\s{}.|&!~^%?\[\]]", "", stripped)
    return len(stripped) == 0


_HYPERPARAM_PATTERNS: list[re.Pattern[str]] = [
    # [const] type name = NUM [* ident]; — simple numeric assignment
    re.compile(
        r"^(const\s+)?\w+\s+\w+\s*=\s*"
        r"[\d.eE+\-]+(\s*[*+\-/]\s*(\w+|[\d.eE+\-]+))*\s*;$"
    ),
    # name = NUM [* ident]; — reassignment
    re.compile(
        r"^\w+\s*=\s*"
        r"[\d.eE+\-]+(\s*[*+\-/]\s*(\w+|[\d.eE+\-]+))*\s*;$"
    ),
    # [else] if (ident op NUM) ident = NUM; — threshold dispatch
    re.compile(
        r"^(else\s+)?if\s*\(\s*\(?\s*\w+\s*[<>=!&|]+\s*[\d.eE+\-]+\s*\)?\s*\)"
        r"\s*(\w+\s*=\s*[\d.eE+\-]+\s*;|\s*\{?\s*)$"
    ),
    # if((ident & NUM) == NUM){ — bitmask check frequency
    re.compile(
        r"^(else\s+)?if\s*\(\s*\(\s*\w+\s*&\s*[\d]+\s*\)\s*==\s*[\d]+\s*\)\s*\{?\s*$"
    ),
]


def _is_hyperparam(line: str) -> bool:
    """True if the line is a simple numeric constant / threshold assignment.

    Matches patterns like:
      ``double end_temp = 0.01;``
      ``if (op_choice_val < 10) operation_type = 0;``
      ``const int MAX_LEN = 30;``

    Does NOT match complex expressions that merely contain a number:
      ``lookahead += 0.3 * (double)D[nnnr][nnnc] * pow(wait3, alpha);``
      ``for(int d2=0; d2<4; ++d2)``
    """
    return any(p.match(line) for p in _HYPERPARAM_PATTERNS)


# ---------------------------------------------------------------------------
# Diff extraction
# ---------------------------------------------------------------------------

@dataclass
class DiffLines:
    removed: list[str]  # normalised non-trivial lines
    added: list[str]


def compute_diff(
    parent_code: str,
    child_code: str,
    *,
    collapse_numbers: bool = False,
    exclude_hyperparams: bool = False,
) -> DiffLines:
    """Extract normalised removed/added lines from a unified diff."""
    parent_lines = parent_code.splitlines()
    child_lines = child_code.splitlines()
    diff = difflib.unified_diff(parent_lines, child_lines, n=0)

    removed: list[str] = []
    added: list[str] = []

    for line in diff:
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            norm = _normalise(line[1:], collapse_numbers=collapse_numbers)
            if norm and not _is_trivial(norm) and not _is_numeric_only(norm):
                if exclude_hyperparams and _is_hyperparam(norm):
                    continue
                removed.append(norm)
        elif line.startswith("+"):
            norm = _normalise(line[1:], collapse_numbers=collapse_numbers)
            if norm and not _is_trivial(norm) and not _is_numeric_only(norm):
                if exclude_hyperparams and _is_hyperparam(norm):
                    continue
                added.append(norm)

    return DiffLines(removed=removed, added=added)


# ---------------------------------------------------------------------------
# Cycling detection
# ---------------------------------------------------------------------------

@dataclass
class LineEvent:
    """A single occurrence of a normalised line being added or removed."""
    iteration: int
    program_id: str
    model: str
    action: str  # "added" or "removed"


@dataclass
class Cycle:
    """A detected cycle: a line removed at iteration i, re-added at iteration j."""
    line: str
    removed_iter: int
    removed_program: str
    removed_model: str
    added_iter: int
    added_program: str
    added_model: str


@dataclass
class ProgramCyclingSummary:
    """Per-program cycling statistics."""
    program_id: str
    iteration: int
    model: str
    score: float
    total_added: int
    recycled_added: int  # added lines that were previously removed
    total_removed: int
    recycled_removed: int  # removed lines that get re-added later (filled in second pass)
    cycles: list[Cycle] = field(default_factory=list)

    @property
    def recycle_ratio(self) -> float:
        if self.total_added == 0:
            return 0.0
        return self.recycled_added / self.total_added


def detect_cycles(
    programs: dict[str, dict[str, Any]],
    *,
    min_line_len: int = 5,
    collapse_numbers: bool = False,
    exclude_hyperparams: bool = False,
) -> tuple[list[Cycle], dict[str, ProgramCyclingSummary]]:
    """Detect line-level cycling across all programs in a run.

    Returns:
        cycles: list of individual line-cycle events
        summaries: per-program cycling statistics keyed by program id
    """
    # Build per-program diffs sorted by iteration
    ordered: list[tuple[int, str, dict[str, Any]]] = []
    for pid, prog in programs.items():
        it = prog.get("iteration_found") or 0
        ordered.append((it, pid, prog))
    ordered.sort(key=lambda x: x[0])

    # Track: normalised line -> list of (iteration, program_id, model, "added"/"removed")
    line_events: dict[str, list[LineEvent]] = defaultdict(list)

    # Per-program diff info
    program_diffs: dict[str, DiffLines] = {}
    summaries: dict[str, ProgramCyclingSummary] = {}

    for iteration, pid, prog in ordered:
        parent_id = prog.get("parent_id")
        if not parent_id or parent_id not in programs:
            continue
        parent = programs[parent_id]
        child_code = prog.get("solution", "")
        parent_code = parent.get("solution", "")
        if not child_code or not parent_code:
            continue

        diff = compute_diff(
            parent_code, child_code,
            collapse_numbers=collapse_numbers,
            exclude_hyperparams=exclude_hyperparams,
        )
        program_diffs[pid] = diff

        model = prog.get("metadata", {}).get("model_name", "unknown")
        score = (prog.get("metrics") or {}).get("combined_score", float("nan"))

        for line in diff.removed:
            if len(line) >= min_line_len:
                line_events[line].append(
                    LineEvent(iteration=iteration, program_id=pid, model=model, action="removed")
                )
        for line in diff.added:
            if len(line) >= min_line_len:
                line_events[line].append(
                    LineEvent(iteration=iteration, program_id=pid, model=model, action="added")
                )

        summaries[pid] = ProgramCyclingSummary(
            program_id=pid,
            iteration=iteration,
            model=model,
            score=score,
            total_added=len(diff.added),
            recycled_added=0,
            total_removed=len(diff.removed),
            recycled_removed=0,
        )

    # Detect cycles: line removed at i, added at j > i
    all_cycles: list[Cycle] = []
    # Track which (line, added_program) pairs we've already counted
    counted_added: set[tuple[str, str]] = set()
    for line, events in line_events.items():
        removals = [e for e in events if e.action == "removed"]
        additions = [e for e in events if e.action == "added"]
        if not removals or not additions:
            continue

        for add_event in additions:
            # Find the most recent removal before this addition
            prior_removals = [r for r in removals if r.iteration < add_event.iteration]
            if not prior_removals:
                continue
            # Take the closest prior removal
            removal = max(prior_removals, key=lambda r: r.iteration)
            cycle = Cycle(
                line=line,
                removed_iter=removal.iteration,
                removed_program=removal.program_id,
                removed_model=removal.model,
                added_iter=add_event.iteration,
                added_program=add_event.program_id,
                added_model=add_event.model,
            )
            all_cycles.append(cycle)

            # Update per-program recycled counts
            key = (line, add_event.program_id)
            if key not in counted_added:
                counted_added.add(key)
                if add_event.program_id in summaries:
                    summaries[add_event.program_id].recycled_added += 1
                    summaries[add_event.program_id].cycles.append(cycle)

    # Second pass: mark removed lines that get re-added later
    counted_removed: set[tuple[str, str]] = set()
    for cycle in all_cycles:
        key = (cycle.line, cycle.removed_program)
        if key not in counted_removed:
            counted_removed.add(key)
            if cycle.removed_program in summaries:
                summaries[cycle.removed_program].recycled_removed += 1

    return all_cycles, summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    cycles: list[Cycle],
    summaries: dict[str, ProgramCyclingSummary],
    *,
    min_cycles: int = 1,
    show_lines: bool = False,
    top_n: int = 20,
) -> None:
    if not cycles:
        print("No cycling detected.")
        return

    # Unique cycled lines
    unique_lines = {c.line for c in cycles}
    print(f"Detected {len(cycles)} cycle events across {len(unique_lines)} unique lines.\n")

    # Most frequently cycled lines
    line_counts: dict[str, int] = defaultdict(int)
    for c in cycles:
        line_counts[c.line] += 1

    print(f"{'Top cycled lines':─<70}")
    for line, count in sorted(line_counts.items(), key=lambda x: -x[1])[:top_n]:
        display = line[:80] + "…" if len(line) > 80 else line
        print(f"  {count:>3}x  {display}")

    # Per-model cycling contribution
    model_removed: dict[str, int] = defaultdict(int)
    model_readded: dict[str, int] = defaultdict(int)
    for c in cycles:
        model_removed[c.removed_model] += 1
        model_readded[c.added_model] += 1

    print(f"\n{'Cycling by model':─<70}")
    all_models = sorted(set(model_removed) | set(model_readded))
    for m in all_models:
        print(f"  {m}: removed {model_removed[m]} lines that cycled back, "
              f"re-added {model_readded[m]} previously removed lines")

    # Programs with highest recycle ratio
    cycling_programs = [
        s for s in summaries.values()
        if s.recycled_added >= min_cycles and s.total_added > 0
    ]
    cycling_programs.sort(key=lambda s: s.recycle_ratio, reverse=True)

    print(f"\n{'Programs with most recycled lines (min {min_cycles} cycles)':─<70}")
    print(f"{'iter':>4}  {'model':<28s}  {'score':>14s}  {'recycled/added':>14s}  {'ratio':>6s}")
    print("─" * 72)
    for s in cycling_programs[:top_n]:
        score_str = f"{s.score:.2f}" if s.score == s.score else "FAILED"
        print(f"{s.iteration:>4}  {s.model:<28s}  {score_str:>14s}  "
              f"{s.recycled_added:>6d}/{s.total_added:<6d}  {s.recycle_ratio:>6.1%}")

    # Cycle chains: lines that cycle 3+ times
    lines_with_many = [(l, c) for l, c in line_counts.items() if c >= 3]
    if lines_with_many:
        print(f"\n{'Lines cycling 3+ times (strong cycling signal)':─<70}")
        for line, count in sorted(lines_with_many, key=lambda x: -x[1]):
            events = sorted(
                [(c.removed_iter, "−", c.removed_model[:10]) for c in cycles if c.line == line]
                + [(c.added_iter, "+", c.added_model[:10]) for c in cycles if c.line == line],
                key=lambda x: x[0],
            )
            display = line[:60] + "…" if len(line) > 60 else line
            timeline = " → ".join(f"{a}{it}({m})" for it, a, m in events)
            print(f"\n  [{count}x] {display}")
            print(f"        {timeline}")

    if show_lines:
        print(f"\n{'All cycle events':─<70}")
        for c in sorted(cycles, key=lambda c: (c.removed_iter, c.added_iter)):
            display = c.line[:60] + "…" if len(c.line) > 60 else c.line
            print(f"  iter {c.removed_iter:>3} ({c.removed_model[:10]}) −  "
                  f"→  iter {c.added_iter:>3} ({c.added_model[:10]}) +  │ {display}")


def write_csv(
    cycles: list[Cycle],
    summaries: dict[str, ProgramCyclingSummary],
    output_path: Path,
) -> None:
    # Per-program summary
    summary_path = output_path.with_suffix(".programs.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iteration", "program_id", "model", "score",
            "total_added", "recycled_added", "total_removed", "recycled_removed",
            "recycle_ratio",
        ])
        for s in sorted(summaries.values(), key=lambda s: s.iteration):
            w.writerow([
                s.iteration, s.program_id[:12], s.model, s.score,
                s.total_added, s.recycled_added, s.total_removed, s.recycled_removed,
                f"{s.recycle_ratio:.4f}",
            ])

    # Individual cycles
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "line", "removed_iter", "removed_program", "removed_model",
            "added_iter", "added_program", "added_model",
        ])
        for c in sorted(cycles, key=lambda c: (c.removed_iter, c.added_iter)):
            w.writerow([
                c.line, c.removed_iter, c.removed_program[:12], c.removed_model,
                c.added_iter, c.added_program[:12], c.added_model,
            ])

    print(f"Wrote {output_path} ({len(cycles)} cycles)")
    print(f"Wrote {summary_path} ({len(summaries)} programs)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path, help="Run output directory")
    parser.add_argument(
        "--min-cycles", type=int, default=1,
        help="Min recycled lines to include a program in the report (default: 1)",
    )
    parser.add_argument(
        "--min-line-len", type=int, default=5,
        help="Min normalised line length to track (default: 5)",
    )
    parser.add_argument(
        "--show-lines", action="store_true",
        help="Print every individual cycle event",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Write cycle events to CSV",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top items to show in each section (default: 20)",
    )
    parser.add_argument(
        "--collapse-numbers", action="store_true",
        help="Replace numeric literals with '#' before comparing lines, "
             "so lines differing only in constants are treated as identical",
    )
    parser.add_argument(
        "--exclude-hyperparams", action="store_true",
        help="Exclude lines containing numeric literals (hyperparameter tuning). "
             "Shows only structural code cycling.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        sys.exit(f"Directory not found: {run_dir}")

    programs = load_programs(run_dir)
    if not programs:
        sys.exit(f"No programs found in {run_dir}")

    print(f"Loaded {len(programs)} programs from {run_dir.name}\n")

    cycles, summaries = detect_cycles(
        programs, min_line_len=args.min_line_len,
        collapse_numbers=args.collapse_numbers,
        exclude_hyperparams=args.exclude_hyperparams,
    )

    print_report(
        cycles, summaries,
        min_cycles=args.min_cycles,
        show_lines=args.show_lines,
        top_n=args.top,
    )

    if args.csv:
        write_csv(cycles, summaries, args.csv)


if __name__ == "__main__":
    main()
