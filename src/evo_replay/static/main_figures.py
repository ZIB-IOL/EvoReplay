"""Compact main-paper figures.

Produces small, single-figure PDFs sized for paper main-text inclusion:

  main_fig_loc_hparam.pdf   1x2 panel: LOC trajectory + hparam trajectory,
                            ALE vs math overlay (median + IQR).
  main_fig_bo_outcome.pdf   single-panel bar: BO improves / no-change / regresses.

The full breakdowns (per-backend grids, per-iteration histograms, etc.) are
already in paper_stats/ and serve as appendix material; this module is the
"slim main-paper" variant.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evo_replay.core.checkpoints import load_programs
from evo_replay.static.aggregate_loc import (
    aggregate, backend_of, find_runs, domain_of,
)
from evo_replay.static.hyperparameter_counts import count_hyperparameter_candidates
from evo_replay.core.checkpoints import detect_language

DOMAIN_COLOR = {"ale": "#1f77b4", "math": "#d62728"}


def loc_hparam_trajectories(roots: list[Path]):
    """Return list of {domain, x_norm, bsf_lines_norm, bsf_hparams_norm} rows.

    One row per (run, iteration) along the public best-so-far chain.
    """
    runs = find_runs(roots)
    out = []
    for run in runs:
        progs = load_programs(run)
        if not progs:
            continue
        lang = detect_language(run) or "python"
        rows = []
        for p in progs.values():
            sol = p.get("solution")
            if not isinstance(sol, str) or not sol:
                continue
            metrics = p.get("metrics") or {}
            score = metrics.get("combined_score")
            if not isinstance(score, (int, float)):
                continue
            try:
                n_h, _, _ = count_hyperparameter_candidates(sol, lang)
            except Exception:
                n_h = None
            rows.append({
                "iteration": int(p.get("iteration_found") or 0),
                "lines": len(sol.splitlines()),
                "hparams": n_h,
                "score": float(score),
            })
        if not rows:
            continue
        rows.sort(key=lambda r: r["iteration"])
        iters = sorted({r["iteration"] for r in rows})
        if len(iters) < 2:
            continue
        max_iter = max(iters)
        seed_rows = [r for r in rows if r["iteration"] == iters[0]]
        seed_lines = float(np.median([r["lines"] for r in seed_rows]))
        seed_h_vals = [r["hparams"] for r in seed_rows
                       if r["hparams"] is not None]
        seed_h = float(np.median(seed_h_vals)) if seed_h_vals else 0.0
        denom_h = max(seed_h, 1.0)
        if seed_lines <= 0:
            continue
        bsf_lines = seed_lines
        bsf_h = seed_h
        bsf_score = -float("inf")
        d = domain_of(run)
        for r in rows:
            if r["score"] > bsf_score:
                bsf_score = r["score"]
                bsf_lines = float(r["lines"])
                if r["hparams"] is not None:
                    bsf_h = float(r["hparams"])
            out.append({
                "run": run.name,
                "domain": d,
                "x_norm": r["iteration"] / max_iter if max_iter else 0.0,
                "bsf_lines_norm": bsf_lines / seed_lines,
                "bsf_hparams_norm": bsf_h / denom_h,
            })
    return out


def _draw_overlay(ax, points, metric, n_bins=50, log_y=True,
                  ylabel="", xlabel="normalised iteration"):
    by_dom = defaultdict(list)
    for p in points:
        by_dom[p["domain"]].append(p)

    # for cross-run aggregate we need (run -> list of (x,y))
    for dom in ("ale", "math"):
        sub = by_dom[dom]
        if not sub:
            continue
        # group by run
        runs = defaultdict(list)
        for p in sub:
            runs[p["run"]].append((p["x_norm"], p[metric]))
        # rebuild aggregate-compatible structure
        traj_like = [{"rows": [{"x_norm": x, metric: y}
                               for x, y in sorted(rs)]}
                     for rs in runs.values()]
        agg = aggregate(traj_like, metric=metric, n_bins=n_bins)
        color = DOMAIN_COLOR[dom]
        ax.fill_between(agg["x"], agg["q25"], agg["q75"],
                        color=color, alpha=0.22)
        ax.plot(agg["x"], agg["q50"], color=color, linewidth=2.6,
                label=f"{dom} (n={len(runs)})")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=0.7,
               alpha=0.55)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left")


def fig_loc_hparam(out_path: Path, points):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    _draw_overlay(axes[0], points, "bsf_lines_norm",
                  ylabel=r"best LOC $\div$ seed LOC")
    _draw_overlay(axes[1], points, "bsf_hparams_norm",
                  ylabel=r"best hparams $\div$ seed hparams")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_bo_outcome(out_path: Path, bo_csv: Path):
    pos = zero = neg = 0
    rel_gains = []
    with bo_csv.open() as f:
        for r in csv.DictReader(f):
            try:
                gain = float(r["gain"]) if r["gain"] else None
                rel = float(r["relative_gain"]) if r["relative_gain"] else None
            except (TypeError, ValueError):
                continue
            if gain is None:
                continue
            if gain > 0: pos += 1
            elif gain == 0: zero += 1
            else: neg += 1
            if rel is not None:
                rel_gains.append(rel)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    cats = ["BO improves", "no change", "BO regresses"]
    counts = [pos, zero, neg]
    colors = ["#2ca02c", "#bdbdbd", "#d62728"]
    bars = ax.bar(cats, counts, color=colors, edgecolor="white", linewidth=0.8)
    n = sum(counts)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v}",
                ha="center", va="bottom")
    ax.set_ylabel(f"# BO targets  (n={n})")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply
    apply()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Refined dataset root (e.g. evo_trace_anon)")
    ap.add_argument("--bo-csv", type=Path, required=True,
                    help="paper_stats/bo/bo_per_target.csv")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for main_fig_*.pdf")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print("computing LOC + hparam trajectories...")
    points = loc_hparam_trajectories([args.data_root])
    print(f"  collected {len(points):,} points across "
          f"{len({p['run'] for p in points})} runs")
    fig_loc_hparam(args.out / "main_fig_loc_hparam.pdf", points)
    print(f"  wrote {args.out/'main_fig_loc_hparam.pdf'}")

    fig_bo_outcome(args.out / "main_fig_bo_outcome.pdf", args.bo_csv)
    print(f"  wrote {args.out/'main_fig_bo_outcome.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
