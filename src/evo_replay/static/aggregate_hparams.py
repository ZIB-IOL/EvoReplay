"""Cross-run aggregate: numeric-literal "hyperparameter" count vs iteration.

For every run in the dataset, count tunable numeric literals in each
program's source via `static.hyperparameter_counts.count_hyperparameter_candidates`,
then build per-iteration trajectories:

    median_count          — median hparam count across programs at that iter
    best_count            — hparam count of the best-scoring program seen so far
    median_norm           — median_count / max(seed_count, 1)
    best_norm             — best_count   / max(seed_count, 1)

Cross-run aggregation: per x_norm bin, compute IQR over runs (forward-filling
runs that ended early in the bin so they don't pull the band down).

Outputs:
    <out>/aggregate_hparams.{json,csv}
    <out>/aggregate_hparams.png
    <out>/aggregate_hparams_by_domain.{png,csv}   (with --by-domain)
    <out>/aggregate_hparams_by_backend.png        (with --by-backend)

Usage:
    uv run python -m evo_replay.static.aggregate_hparams \\
        /path/to/evo_trace_anon --out /tmp/hp --by-domain
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evo_replay.core.checkpoints import detect_language, load_programs
from evo_replay.static.aggregate_loc import (
    aggregate, backend_of, domain_of, find_runs,
)
from evo_replay.static.hyperparameter_counts import count_hyperparameter_candidates


def trajectory(run_dir: Path) -> dict[str, Any] | None:
    progs = load_programs(run_dir)
    lang = detect_language(run_dir) or "python"

    by_iter: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for p in progs.values():
        sol = p.get("solution")
        if not isinstance(sol, str) or not sol:
            continue
        score = (p.get("metrics") or {}).get("combined_score")
        if not isinstance(score, (int, float)):
            continue
        try:
            n_h, _, _ = count_hyperparameter_candidates(sol, lang)
        except Exception:
            continue
        it = int(p.get("iteration_found") or 0)
        by_iter[it].append((n_h, float(score)))

    if not by_iter:
        return None
    iters = sorted(by_iter)
    if len(iters) < 2:
        return None

    seed_count = float(np.median([n for n, _ in by_iter[iters[0]]]))
    norm_denom = max(seed_count, 1.0)
    max_iter = max(iters)
    rows = []
    best_count = seed_count
    best_score = -float("inf")
    for it in iters:
        counts_here = [n for n, _ in by_iter[it]]
        for n, s in by_iter[it]:
            if s > best_score:
                best_score = s
                best_count = float(n)
        med = float(np.median(counts_here))
        rows.append({
            "iteration": it,
            "x_norm": it / max_iter if max_iter > 0 else 0.0,
            "median_count": med,
            "median_norm": med / norm_denom,
            "best_count": best_count,
            "best_norm": best_count / norm_denom,
            "n_programs": len(counts_here),
        })
    return {
        "run_dir": str(run_dir),
        "backend": backend_of(run_dir),
        "domain": domain_of(run_dir),
        "language": lang,
        "seed_count": seed_count,
        "max_iter": max_iter,
        "rows": rows,
    }


def plot_panel(ax, trajs, metric, title, color, n_bins, *, log_y=False):
    for t in trajs:
        xs = [r["x_norm"] for r in t["rows"]]
        ys = [r[metric] for r in t["rows"]]
        ax.plot(xs, ys, color=color, alpha=0.18, linewidth=0.7)
    agg = aggregate(trajs, metric=metric, n_bins=n_bins)
    ax.fill_between(agg["x"], agg["q25"], agg["q75"], color=color, alpha=0.30,
                    label=f"IQR (n={int(np.median(agg['n']))} runs)")
    ax.plot(agg["x"], agg["q50"], color=color, linewidth=2.4, label="median")
    if log_y:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("normalised iteration  (iter / max_iter)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=9)
    return agg


def write_csv(path: Path, trajs, metrics, n_bins, group_label=None):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        head = (["group"] if group_label else []) + \
               ["metric", "x_norm", "n_runs", "q25", "q50", "q75"]
        w.writerow(head)
        groups = {group_label: trajs} if group_label \
            else {None: trajs}
        for g, ts in groups.items():
            for metric in metrics:
                agg = aggregate(ts, metric=metric, n_bins=n_bins)
                for x, n, q25, q50, q75 in zip(
                        agg["x"], agg["n"], agg["q25"], agg["q50"], agg["q75"]):
                    row = ([g] if g is not None else []) + \
                          [metric, f"{x:.4f}", int(n),
                           f"{q25:.4f}", f"{q50:.4f}", f"{q75:.4f}"]
                    w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply as _apply_paper_style
    _apply_paper_style()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bins", type=int, default=50)
    ap.add_argument("--by-backend", action="store_true")
    ap.add_argument("--by-domain", action="store_true")
    args = ap.parse_args(argv)

    runs = find_runs(args.roots)
    print(f"discovered {len(runs)} runs")
    trajs = []
    skipped = []
    for run in runs:
        t = trajectory(run)
        if t is None:
            skipped.append(str(run))
            continue
        trajs.append(t)
    print(f"usable trajectories: {len(trajs)}  (skipped {len(skipped)})")
    if not trajs:
        sys.exit("No usable runs.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aggregate_hparams.json").write_text(
        json.dumps({"runs": trajs, "skipped": skipped}, indent=2))

    METRICS_ABS = ("median_count", "best_count")
    METRICS_NORM = ("median_norm", "best_norm")

    write_csv(args.out / "aggregate_hparams.csv", trajs,
              METRICS_ABS + METRICS_NORM, args.n_bins)

    # Combined plot: 2x2 panels (abs + norm) × (median + best)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_panel(axes[0][0], trajs, "median_count",
               "median hparam count per iter", "#1f77b4", args.n_bins)
    plot_panel(axes[0][1], trajs, "best_count",
               "best-so-far hparam count", "#d62728", args.n_bins)
    plot_panel(axes[1][0], trajs, "median_norm",
               "median hparam count (× seed)", "#1f77b4", args.n_bins,
               log_y=True)
    plot_panel(axes[1][1], trajs, "best_norm",
               "best-so-far hparam count (× seed)", "#d62728", args.n_bins,
               log_y=True)
    fig.suptitle(f"hyperparameter count vs iteration "
                 f"(n={len(trajs)} runs, denom = max(seed_count, 1))")
    fig.tight_layout()
    fig.savefig(args.out / "aggregate_hparams.png", dpi=180)
    plt.close(fig)

    if args.by_backend:
        by_b: dict[str, list] = defaultdict(list)
        for t in trajs:
            by_b[t["backend"]].append(t)
        bs = sorted(by_b)
        fig, axes = plt.subplots(len(bs), 2, figsize=(14, 4 * len(bs)),
                                 sharex=True, squeeze=False)
        for i, b in enumerate(bs):
            plot_panel(axes[i][0], by_b[b], "median_count",
                       f"{b}: median hparam count "
                       f"(n={len(by_b[b])})", "#1f77b4", args.n_bins)
            plot_panel(axes[i][1], by_b[b], "best_count",
                       f"{b}: best-so-far hparam count "
                       f"(n={len(by_b[b])})", "#d62728", args.n_bins)
        fig.tight_layout()
        fig.savefig(args.out / "aggregate_hparams_by_backend.png", dpi=180)
        plt.close(fig)

    if args.by_domain:
        by_d: dict[str, list] = defaultdict(list)
        for t in trajs:
            by_d[t["domain"]].append(t)
        ds = sorted(by_d)
        fig, axes = plt.subplots(len(ds), 2, figsize=(14, 4 * len(ds)),
                                 sharex=True, squeeze=False)
        for i, d in enumerate(ds):
            plot_panel(axes[i][0], by_d[d], "median_count",
                       f"{d}: median hparam count "
                       f"(n={len(by_d[d])})", "#1f77b4", args.n_bins)
            plot_panel(axes[i][1], by_d[d], "best_count",
                       f"{d}: best-so-far hparam count "
                       f"(n={len(by_d[d])})", "#d62728", args.n_bins)
        fig.tight_layout()
        fig.savefig(args.out / "aggregate_hparams_by_domain.png", dpi=180)
        plt.close(fig)

        with (args.out / "aggregate_hparams_by_domain.csv").open(
                "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "metric", "x_norm", "n_runs",
                        "q25", "q50", "q75"])
            for d in ds:
                for metric in METRICS_ABS + METRICS_NORM:
                    agg = aggregate(by_d[d], metric=metric, n_bins=args.n_bins)
                    for x, n, q25, q50, q75 in zip(
                            agg["x"], agg["n"], agg["q25"],
                            agg["q50"], agg["q75"]):
                        w.writerow([d, metric, f"{x:.4f}", int(n),
                                    f"{q25:.4f}", f"{q50:.4f}",
                                    f"{q75:.4f}"])

    print(f"wrote outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
