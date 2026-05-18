"""Walk a run directory, classify every parent->child edit, write JSONL+CSV+summary.

Usage:
    uv run python -m evo_replay.edit_taxonomy.run_classify <run_dir> [options]

Outputs (under <run_dir>/analysis/ by default):
    llm_edit_taxonomy.jsonl          one record per edit with full LLM output
    llm_edit_taxonomy.csv            flat per-edit table (one row per edit)
    llm_edit_taxonomy.summary.json   per-category counts, co-occurrence, score deltas
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

from evo_replay.core.checkpoints import load_programs
from evo_replay.edit_taxonomy.judge import (
    DEFAULT_MODEL,
    JudgeResult,
    classify_diff,
    make_unified_diff,
)
from evo_replay.edit_taxonomy.rubric import CATEGORY_NAMES


def _score(p: Dict[str, Any], key: str = "combined_score") -> Optional[float]:
    v = (p.get("metrics") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _detect_language(solution: str) -> str:
    s = solution[:2000]
    if "#include" in s or "::" in s or "int main(" in s:
        return "cpp"
    return "python"


def iter_parent_child(programs: Dict[str, Dict[str, Any]]):
    """Yield (parent, child) dicts for every edit in iteration order."""
    children = []
    for pid, child in programs.items():
        par_id = child.get("parent_id")
        if not par_id or par_id not in programs:
            continue
        par = programs[par_id]
        if not (par.get("solution") and child.get("solution")):
            continue
        if par["solution"] == child["solution"]:
            continue
        children.append((int(child.get("iteration_found") or 0), par, child))
    children.sort(key=lambda t: t[0])
    for _, par, child in children:
        yield par, child


def classify_run(
    run_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    diff_context: int = 3,
    max_edits: Optional[int] = None,
    score_metric: str = "combined_score",
    progress: bool = True,
) -> List[Dict[str, Any]]:
    programs = load_programs(run_dir)
    if not programs:
        raise SystemExit(f"No programs in {run_dir}")

    pairs = list(iter_parent_child(programs))
    if max_edits is not None:
        pairs = pairs[:max_edits]
    iterator: Iterable = pairs
    if progress:
        iterator = tqdm(pairs, desc=run_dir.name, unit="edit")

    out: List[Dict[str, Any]] = []
    for parent, child in iterator:
        diff = make_unified_diff(parent["solution"], child["solution"], n=diff_context)
        if not diff.strip():
            continue
        language = _detect_language(child["solution"])
        result: JudgeResult = classify_diff(diff, language=language, model=model)
        s_c, s_p = _score(child, score_metric), _score(parent, score_metric)
        delta = (s_c - s_p) if (s_c is not None and s_p is not None) else None
        rec = {
            "iteration": int(child.get("iteration_found") or 0),
            "parent_iteration": int(parent.get("iteration_found") or 0),
            "program_id": child["id"],
            "parent_id": parent["id"],
            "language": language,
            "score": s_c,
            "parent_score": s_p,
            "score_delta": delta,
            "diff_chars": len(diff),
            "labels": result.labels,
            "rationale": result.rationale,
            "diff_sha": result.diff_sha,
            "cache_hit": result.cache_hit,
            "model": result.model,
            "parse_error": result.parse_error,
        }
        out.append(rec)
    return out


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    label_counts = Counter()
    n_labels_per_edit = Counter()
    cooc: Counter = Counter()
    by_label_delta: Dict[str, List[float]] = defaultdict(list)
    parse_errors = 0

    for r in records:
        labels = r.get("labels") or []
        n_labels_per_edit[len(labels)] += 1
        for l in labels:
            label_counts[l] += 1
        for a, b in combinations(sorted(set(labels)), 2):
            cooc[(a, b)] += 1
        d = r.get("score_delta")
        if isinstance(d, (int, float)):
            for l in labels:
                by_label_delta[l].append(float(d))
        if r.get("parse_error"):
            parse_errors += 1

    def _avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    return {
        "n_edits": n,
        "n_parse_errors": parse_errors,
        "label_counts": {k: label_counts.get(k, 0) for k in CATEGORY_NAMES},
        "label_share": {
            k: (label_counts.get(k, 0) / n if n else 0.0) for k in CATEGORY_NAMES
        },
        "labels_per_edit_distribution": dict(n_labels_per_edit),
        "cooccurrence_top": [
            {"a": a, "b": b, "n": v}
            for (a, b), v in cooc.most_common(20)
        ],
        "mean_score_delta_by_label": {
            k: _avg(by_label_delta.get(k, [])) for k in CATEGORY_NAMES
        },
        "n_score_up_by_label": {
            k: sum(1 for d in by_label_delta.get(k, []) if d > 0)
            for k in CATEGORY_NAMES
        },
        "n_score_down_by_label": {
            k: sum(1 for d in by_label_delta.get(k, []) if d < 0)
            for k in CATEGORY_NAMES
        },
    }


def write_outputs(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    out_prefix: Path,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_prefix.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iteration",
                "parent_iteration",
                "program_id",
                "parent_id",
                "score",
                "parent_score",
                "score_delta",
                "diff_chars",
                "labels",
                "rationale",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r["iteration"],
                    r["parent_iteration"],
                    r["program_id"][:12],
                    r["parent_id"][:12],
                    r["score"] if r["score"] is not None else "",
                    r["parent_score"] if r["parent_score"] is not None else "",
                    r["score_delta"] if r["score_delta"] is not None else "",
                    r["diff_chars"],
                    "|".join(r["labels"]),
                    r["rationale"],
                ]
            )

    summary_path = out_prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


def print_report(summary: Dict[str, Any]) -> None:
    n = summary["n_edits"]
    print(f"edits classified: {n}  (parse errors: {summary['n_parse_errors']})")
    print("  label counts (share):")
    for k in CATEGORY_NAMES:
        c = summary["label_counts"][k]
        s = summary["label_share"][k]
        print(f"    {k:<24} {c:>4}  ({s:>5.1%})")
    print("  labels-per-edit:")
    for k in sorted(summary["labels_per_edit_distribution"]):
        print(
            f"    {k} labels: {summary['labels_per_edit_distribution'][k]}"
        )
    print("  top co-occurrences:")
    for row in summary["cooccurrence_top"][:10]:
        print(f"    {row['a']:<22} + {row['b']:<22} {row['n']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output prefix (without extension). "
                         "Defaults to <run_dir>/analysis/llm_edit_taxonomy")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--diff-context", type=int, default=3)
    ap.add_argument("--max-edits", type=int, default=None,
                    help="Cap how many edits to classify (smoke testing).")
    ap.add_argument("--score-metric", default="combined_score")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    out_prefix = args.out or (run_dir / "analysis" / "llm_edit_taxonomy")

    records = classify_run(
        run_dir,
        model=args.model,
        diff_context=args.diff_context,
        max_edits=args.max_edits,
        score_metric=args.score_metric,
    )
    if not records:
        sys.exit("No edits to classify.")
    summary = summarise(records)
    print_report(summary)
    write_outputs(records, summary, out_prefix)


if __name__ == "__main__":
    main()
