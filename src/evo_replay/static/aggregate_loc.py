"""Cross-run aggregate: normalised lines-of-code vs. iteration.

Walks every run directory under one or more roots, loads its programs
through `core.checkpoints.load_programs` (which auto-detects refined vs.
raw layout), and computes a normalised LOC trajectory per run:

    x = iteration_found / max_iteration_in_run
    y = lines_at_iter / seed_lines

Two y-aggregates per run, computed at each iteration bucket:

    median_lines  — median LOC across all programs proposed at that iter
    best_lines    — LOC of the highest-scoring program seen up to that iter

The aggregator produces:

    <out_dir>/aggregate_loc.{json,csv}     per-run trajectories + summary
    <out_dir>/aggregate_loc.png            overlay + cross-run median+IQR

Usage:

    uv run python -m evo_replay.static.aggregate_loc \
        /path/to/evo_trace_anon --out /tmp/loc_agg --by-backend
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evo_replay.core.checkpoints import load_programs


def domain_of(run_dir: Path) -> str:
    """Classify a run as 'ale' (AtCoder Heuristic Contest / ALE bench)
    vs 'math' (everything else: circle_packing, heilbronn, autocorr_ineq, ...).
    """
    n = run_dir.name.lower()
    if n.startswith("ale_bench_") or re.match(r"ahc\d", n):
        return "ale"
    return "math"


def find_runs(roots: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for r in roots:
        for meta in r.rglob("meta.json"):
            run_dir = meta.parent
            if (run_dir / "programs.jsonl").exists() \
                    or (run_dir / "checkpoints").exists():
                runs.append(run_dir)
    # also accept raw-only dirs without meta.json
    for r in roots:
        for ck in r.rglob("checkpoints"):
            if ck.is_dir() and ck.parent not in runs:
                runs.append(ck.parent)
    return sorted(set(runs))


def backend_of(run_dir: Path) -> str:
    meta = run_dir / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("backend", "unknown")
        except Exception:
            pass
    return "unknown"


def trajectory(run_dir: Path) -> dict[str, Any] | None:
    """Return per-iteration LOC stats for one run, or None if too sparse."""
    progs = load_programs(run_dir)
    by_iter: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for p in progs.values():
        sol = p.get("solution")
        if not isinstance(sol, str) or not sol:
            continue
        score = (p.get("metrics") or {}).get("combined_score")
        if not isinstance(score, (int, float)):
            continue
        it = int(p.get("iteration_found") or 0)
        by_iter[it].append((len(sol.splitlines()), float(score)))

    if not by_iter:
        return None
    iters = sorted(by_iter)
    # require at least a seed iteration AND >= 2 iters with data
    if len(iters) < 2:
        return None

    seed_lines = float(np.median([n for n, _ in by_iter[iters[0]]]))
    if seed_lines <= 0:
        return None

    max_iter = max(iters)
    rows = []
    best_lines = seed_lines
    best_score = -float("inf")
    for it in iters:
        lines_here = [n for n, _ in by_iter[it]]
        for n, s in by_iter[it]:
            if s > best_score:
                best_score = s
                best_lines = float(n)
        rows.append({
            "iteration": it,
            "x_norm": it / max_iter if max_iter > 0 else 0.0,
            "median_lines": float(np.median(lines_here)),
            "median_norm": float(np.median(lines_here)) / seed_lines,
            "best_lines": best_lines,
            "best_norm": best_lines / seed_lines,
            "n_programs": len(lines_here),
        })
    return {
        "run_dir": str(run_dir),
        "backend": backend_of(run_dir),
        "domain": domain_of(run_dir),
        "seed_lines": seed_lines,
        "max_iter": max_iter,
        "rows": rows,
    }


def aggregate(trajs: list[dict[str, Any]],
              metric: str,  # "median_norm" or "best_norm"
              n_bins: int = 50) -> dict[str, Any]:
    """Bin x_norm into n_bins and report cross-run quantiles of y."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # collect per-run, per-bin: take the LAST y in the bin (the iter closest
    # to the bin's right edge represents the bin's value for that run)
    per_bin: list[list[float]] = [[] for _ in range(n_bins)]
    for t in trajs:
        last_y_in_bin: list[float | None] = [None] * n_bins
        for row in t["rows"]:
            x = row["x_norm"]
            y = row[metric]
            idx = min(int(x * n_bins), n_bins - 1)
            last_y_in_bin[idx] = y
        # forward-fill: a bin with no programs in this run inherits the prior
        # bin's value (so a run that ended earlier doesn't pull the median up).
        prev = None
        for i in range(n_bins):
            if last_y_in_bin[i] is None:
                last_y_in_bin[i] = prev
            else:
                prev = last_y_in_bin[i]
        for i, y in enumerate(last_y_in_bin):
            if y is not None:
                per_bin[i].append(y)

    q25 = np.array([np.quantile(b, 0.25) if b else np.nan for b in per_bin])
    q50 = np.array([np.quantile(b, 0.50) if b else np.nan for b in per_bin])
    q75 = np.array([np.quantile(b, 0.75) if b else np.nan for b in per_bin])
    n = np.array([len(b) for b in per_bin])
    return {"x": centers, "q25": q25, "q50": q50, "q75": q75, "n": n}


def plot_panel(ax, trajs, metric, title, color, n_bins, *, log_y=True):
    for t in trajs:
        xs = [r["x_norm"] for r in t["rows"]]
        ys = [r[metric] for r in t["rows"]]
        ax.plot(xs, ys, color=color, alpha=0.18, linewidth=0.7)

    agg = aggregate(trajs, metric=metric, n_bins=n_bins)
    ax.fill_between(agg["x"], agg["q25"], agg["q75"], color=color, alpha=0.30,
                    label=f"IQR (n={int(np.median(agg['n']))} runs)")
    ax.plot(agg["x"], agg["q50"], color=color, linewidth=2.4, label="median")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=0.8, alpha=0.6,
               label="seed length")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("normalised iteration  (iter / max_iter)")
    ax.set_ylabel("normalised LOC  (lines / seed_lines)"
                  + ("  [log scale]" if log_y else ""))
    ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=9)
    return agg


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply as _apply_paper_style
    _apply_paper_style()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", type=Path, nargs="+",
                    help="One or more dataset roots to walk.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for {json,csv,png}.")
    ap.add_argument("--n-bins", type=int, default=50)
    ap.add_argument("--by-backend", action="store_true",
                    help="Also produce one panel per backend.")
    ap.add_argument("--by-domain", action="store_true",
                    help="Also produce one panel per domain (ALE vs math).")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Restrict to these backends (default: all).")
    args = ap.parse_args(argv)

    runs = find_runs(args.roots)
    print(f"discovered {len(runs)} runs under {args.roots}")

    trajs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for run in runs:
        t = trajectory(run)
        if t is None:
            skipped.append(str(run))
            continue
        if args.include and t["backend"] not in args.include:
            continue
        trajs.append(t)
    print(f"usable trajectories: {len(trajs)}  (skipped {len(skipped)})")

    if not trajs:
        sys.exit("No usable runs.")

    args.out.mkdir(parents=True, exist_ok=True)

    # JSON: per-run trajectories
    (args.out / "aggregate_loc.json").write_text(
        json.dumps({"runs": trajs, "skipped": skipped}, indent=2))

    # CSV: cross-run aggregate per metric
    csv_path = args.out / "aggregate_loc.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "x_norm", "n_runs", "q25", "q50", "q75"])
        for metric in ("median_norm", "best_norm"):
            agg = aggregate(trajs, metric=metric, n_bins=args.n_bins)
            for x, n, q25, q50, q75 in zip(
                    agg["x"], agg["n"], agg["q25"], agg["q50"], agg["q75"]):
                w.writerow([metric, f"{x:.4f}", int(n),
                            f"{q25:.4f}", f"{q50:.4f}", f"{q75:.4f}"])

    # Plot: combined (all backends) — two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_panel(axes[0], trajs, "median_norm",
               "median LOC per iteration (normalised)",
               "#1f77b4", args.n_bins)
    plot_panel(axes[1], trajs, "best_norm",
               "best-so-far LOC (normalised)",
               "#d62728", args.n_bins)
    fig.suptitle(f"LOC vs. iteration — aggregate across {len(trajs)} runs")
    fig.tight_layout()
    fig.savefig(args.out / "aggregate_loc.png", dpi=180)
    plt.close(fig)

    # Optional: per-backend panels
    if args.by_backend:
        by_backend: dict[str, list] = defaultdict(list)
        for t in trajs:
            by_backend[t["backend"]].append(t)
        backends = sorted(by_backend)
        fig, axes = plt.subplots(len(backends), 2,
                                 figsize=(14, 4 * len(backends)),
                                 sharex=True, sharey=True, squeeze=False)
        for row_i, b in enumerate(backends):
            plot_panel(axes[row_i][0], by_backend[b], "median_norm",
                       f"{b}: median LOC", "#1f77b4", args.n_bins)
            plot_panel(axes[row_i][1], by_backend[b], "best_norm",
                       f"{b}: best-so-far LOC", "#d62728", args.n_bins)
        fig.tight_layout()
        fig.savefig(args.out / "aggregate_loc_by_backend.png", dpi=180)
        plt.close(fig)

    # Optional: per-domain panels (ALE vs math)
    if args.by_domain:
        by_domain: dict[str, list] = defaultdict(list)
        for t in trajs:
            by_domain[t["domain"]].append(t)
        domains = sorted(by_domain)
        fig, axes = plt.subplots(len(domains), 2,
                                 figsize=(14, 4 * len(domains)),
                                 sharex=True, sharey=True, squeeze=False)
        for row_i, d in enumerate(domains):
            plot_panel(axes[row_i][0], by_domain[d], "median_norm",
                       f"{d}: median LOC (n={len(by_domain[d])})",
                       "#1f77b4", args.n_bins)
            plot_panel(axes[row_i][1], by_domain[d], "best_norm",
                       f"{d}: best-so-far LOC (n={len(by_domain[d])})",
                       "#d62728", args.n_bins)
        fig.tight_layout()
        fig.savefig(args.out / "aggregate_loc_by_domain.png", dpi=180)
        plt.close(fig)

        # Domain CSV: per-bin q25/q50/q75 per domain
        with (args.out / "aggregate_loc_by_domain.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "metric", "x_norm", "n_runs",
                        "q25", "q50", "q75"])
            for d in domains:
                for metric in ("median_norm", "best_norm"):
                    agg = aggregate(by_domain[d], metric=metric,
                                    n_bins=args.n_bins)
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
