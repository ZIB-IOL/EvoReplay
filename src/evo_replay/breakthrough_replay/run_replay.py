"""Thin convenience wrapper around `counterfactuals.py`.

Picks the top-N best-so-far events automatically, sweeps a comma-separated
list of models and prompt variants, then post-processes
`<output>/events.json` into a flat `replay_matrix.{json,csv}` for paper
plots.

For full power (custom condition specs, retry semantics, etc.) call
`python -m evo_replay.breakthrough_replay.counterfactuals` directly.

Usage:
    uv run python -m evo_replay.breakthrough_replay.run_replay <run_dir> \
        --top-events 3 \
        --models gpt-oss:120b,gemini-3-flash-preview \
        --prompts exact,strict_diff,no_history \
        --repeats 1 --attempts 2
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from evo_replay.core.checkpoints import (
    best_so_far_targets,
    load_programs,
    primary_score,
)


def _select_events(run_dir: Path, top_n: int) -> List[int]:
    """Return iteration numbers of the top-N best-so-far updates."""
    progs = load_programs(run_dir)
    bsf = best_so_far_targets(progs)
    if not bsf:
        sys.exit("No best-so-far events found in this run.")
    if top_n <= 0 or top_n >= len(bsf):
        chosen = bsf
    else:
        chosen = sorted(bsf, key=lambda p: primary_score(p) or 0.0, reverse=True)[:top_n]
    iters = sorted({int(p.get("iteration_found") or 0) for p in chosen})
    print(f"Selected {len(iters)} event iterations: {iters}")
    return iters


def _flatten_events(events_path: Path) -> List[Dict[str, Any]]:
    """Read counterfactuals events.json and emit one row per (event,cond,repeat)."""
    raw = json.loads(events_path.read_text())
    rows: List[Dict[str, Any]] = []
    for event in raw:
        ev_iter = event.get("iteration")
        ev_id = event.get("program_id")
        parent_score = event.get("parent_score")
        original_score = event.get("original_score")
        for cond in event.get("conditions", []):
            cond_name = cond.get("name")
            model = cond.get("model")
            prompt = cond.get("prompt") or cond.get("prompt_variant")
            for r_idx, repeat in enumerate(cond.get("repeats", [])):
                rows.append({
                    "event_iteration": ev_iter,
                    "event_program": ev_id,
                    "parent_score": parent_score,
                    "original_score": original_score,
                    "condition": cond_name,
                    "model": model,
                    "prompt": prompt,
                    "repeat": r_idx,
                    "score": repeat.get("score"),
                    "classification": repeat.get("classification"),
                    "attempts": repeat.get("attempts"),
                })
    return rows


def _write_matrix(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "replay_matrix.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str))
    csv_path = out_dir / "replay_matrix.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--top-events", type=int, default=3,
                    help="Replay this many top-scoring best-so-far events")
    ap.add_argument("--event-iter", action="append", default=[],
                    help="Explicit iteration number(s); repeat to add. "
                         "Overrides --top-events.")
    ap.add_argument("--models", required=True,
                    help="Comma-separated model labels (passed straight to counterfactuals)")
    ap.add_argument("--prompts", default="exact,strict_diff,no_history,no_other_context",
                    help="Comma-separated prompt variants")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--output", type=Path, default=None,
                    help="Defaults to <run_dir>/analysis/replay/")
    ap.add_argument("--evaluator", type=Path, default=None,
                    help="Override evaluator path (defaults to run's recorded evaluator)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    out_dir = args.output or (run_dir / "analysis" / "replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.event_iter:
        iters = sorted({int(x) for x in args.event_iter})
    else:
        iters = _select_events(run_dir, args.top_events)

    cf_argv: List[str] = [str(run_dir)]
    for it in iters:
        cf_argv += ["--event", f"iter:{it}"]
    cf_argv += [
        "--model", args.models,
        "--prompt", args.prompts,
        "--repeats", str(args.repeats),
        "--max-events", str(max(len(iters), 1)),
        "--output", str(out_dir),
    ]
    if args.evaluator:
        # counterfactuals takes evaluator from the run config; surface a clear error.
        sys.exit("counterfactuals.py reads the evaluator from the run's config; "
                 "--evaluator is not currently supported here. "
                 "Edit run_config.yaml or call counterfactuals.py directly.")

    print(f"\n>>> python -m evo_replay.breakthrough_replay.counterfactuals {' '.join(cf_argv)}\n",
          flush=True)
    from evo_replay.breakthrough_replay import counterfactuals as cf
    rc = asyncio.run(cf.main_async(cf_argv))
    if rc != 0:
        sys.exit(rc)

    events_path = out_dir / "events.json"
    if not events_path.exists():
        sys.exit(f"counterfactuals did not produce {events_path}")

    rows = _flatten_events(events_path)
    _write_matrix(rows, out_dir)


if __name__ == "__main__":
    main()
