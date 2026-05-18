"""Aggregate breakthrough_replay summaries across many replay targets.

Reads all replay_experiments/<exp>/<target>/summary.json files under one
or more roots. Each summary describes K repeats of replaying a single
"breakthrough" event under a fresh model+prompt and the outcome stats.

Outputs:
  replay_per_target.csv     per-target rows
  replay_outcome.png        parse / eval / score breakdown across targets
  replay_score_recovery.png replay_score relative to original score
  replay_summary_table.csv  paper Tab 6
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def collect(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for s in sorted(root.rglob("replay_experiments/*/*/summary.json")):
            try:
                d = json.loads(s.read_text())
            except (OSError, ValueError):
                continue
            agg = d.get("aggregate") or {}
            rows.append({
                "exp": s.parent.parent.name,
                "target_dir": s.parent.name,
                "target_id": d.get("target_program_id"),
                "scope": d.get("scope"),
                "repeats": d.get("repeats"),
                "replayed": agg.get("included_node_count")
                            or agg.get("successful_replay_node_count"),
                "parse_rate": agg.get("mean_parse_success_rate"),
                "eval_rate": agg.get("mean_evaluation_success_rate"),
                "match_rate": agg.get("mean_exact_solution_match_rate"),
                "replay_score": agg.get("mean_replay_score"),
                "checkpoint_dir": d.get("checkpoint_dir"),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply as _apply_paper_style
    _apply_paper_style()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = collect(args.roots)
    print(f"replay summaries: {len(rows)}")
    if not rows:
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    # per-target CSV (no paths leaked — checkpoint_dir omitted)
    with (args.out / "replay_per_target.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp", "target_dir", "target_id", "scope", "repeats",
                    "parse_rate", "eval_rate", "match_rate", "replay_score"])
        for r in rows:
            def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else ""
            w.writerow([r["exp"], r["target_dir"], r["target_id"], r["scope"],
                        r["repeats"], fmt(r["parse_rate"]),
                        fmt(r["eval_rate"]), fmt(r["match_rate"]),
                        fmt(r["replay_score"])])

    # --- Fig: outcome rates (parse / eval / exact match) ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, key, title in zip(
            axes,
            ("parse_rate", "eval_rate", "match_rate"),
            ("parse success rate (LLM output → valid program)",
             "evaluation success rate (program runs)",
             "exact-match rate (replay = original solution)")):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        if vals:
            ax.hist(vals, bins=np.linspace(0, 1, 21), color="#4C78A8",
                    edgecolor="white", linewidth=0.6)
            ax.axvline(np.median(vals), color="#d62728", linestyle="--",
                       linewidth=1.2,
                       label=f"median = {np.median(vals):.2f}")
            ax.legend()
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("rate")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("# replay targets")
    fig.suptitle(f"Breakthrough replay: outcome rates "
                 f"(n={len(rows)} targets)")
    fig.tight_layout()
    fig.savefig(args.out / "replay_outcome.png", dpi=180)
    plt.close(fig)

    # --- Fig: replay_score distribution ---
    fig, ax = plt.subplots(figsize=(9, 5))
    rs = [r["replay_score"] for r in rows
          if isinstance(r["replay_score"], (int, float))]
    if rs:
        ax.hist(rs, bins=30, color="#4C78A8", edgecolor="white", linewidth=0.6)
        ax.axvline(np.median(rs), color="#d62728", linestyle="--",
                   linewidth=1.4, label=f"median = {np.median(rs):.2f}")
        ax.legend()
    ax.set_xlabel("mean replay score (across K repeats per target)")
    ax.set_ylabel("# replay targets")
    ax.set_title(f"replay score distribution (n={len(rs)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "replay_score_recovery.png", dpi=180)
    plt.close(fig)

    # --- Summary table ---
    with (args.out / "replay_summary_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        def med(key):
            vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
            return f"{np.median(vals):.3f}" if vals else ""
        w.writerow(["n_targets", len(rows)])
        w.writerow(["median_parse_rate", med("parse_rate")])
        w.writerow(["median_eval_rate", med("eval_rate")])
        w.writerow(["median_match_rate", med("match_rate")])
        w.writerow(["median_replay_score", med("replay_score")])
        # full eval (=1.0) targets
        full_eval = sum(1 for r in rows
                        if isinstance(r["eval_rate"], (int, float))
                        and r["eval_rate"] >= 0.999)
        w.writerow(["targets_with_full_eval_success", full_eval])
        any_match = sum(1 for r in rows
                        if isinstance(r["match_rate"], (int, float))
                        and r["match_rate"] > 0)
        w.writerow(["targets_with_any_exact_match", any_match])

    print(f"wrote outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
