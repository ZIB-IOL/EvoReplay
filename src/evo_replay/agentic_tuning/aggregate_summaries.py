"""Aggregate Bayesian-optimisation summaries across many BO targets.

Reads all bo_runs/<exp>/<target>/summary.json files under one or more roots
and produces:

  bo_per_target.csv        per-target rows (target_id, original, baseline, bo_best, gain)
  bo_gain_distribution.png distribution of (bo_best - original_score), with sign breakdown
  bo_regret_curves.png     overlay of normalised regret curves + median+IQR
  bo_summary_table.csv     paper Tab 5 (n, fraction_positive, median_gain_pct, ...)

Score normalisation: BO operates within one program's parameter space, so
absolute deltas are problem-scale-dependent. We report:
  - sign of gain (positive / zero / negative)
  - relative gain = (bo_best - original) / max(|original|, eps)
  - regret_norm[t] = (regret_curve[t] - regret_curve[0]) / (bo_best - regret_curve[0])
    so each curve runs 0 → 1 over its 24 calls, and the cross-target median
    is a "normalised search trajectory".
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
        for s in sorted(root.rglob("bo_runs/*/*/summary.json")):
            try:
                d = json.loads(s.read_text())
            except (OSError, ValueError):
                continue
            tgt = d.get("target") or {}
            origin = tgt.get("original_score")
            baseline = tgt.get("baseline_score")
            best = d.get("best_score")
            regret = d.get("regret_curve") or []
            if not isinstance(best, (int, float)) or not regret:
                continue
            gain = (best - origin) if isinstance(origin, (int, float)) else None
            rel = (gain / max(abs(origin), 1e-9)
                   if isinstance(origin, (int, float)) and gain is not None
                   else None)
            rows.append({
                "exp": s.parent.parent.name,
                "target_dir": s.parent.name,
                "target_id": tgt.get("id"),
                "iteration": tgt.get("iteration"),
                "original_score": origin,
                "baseline_score": baseline,
                "bo_best": best,
                "gain": gain,
                "relative_gain": rel,
                "n_calls": d.get("n_calls", len(regret)),
                "n_specs": len(d.get("applied_specs") or []),
                "regret_curve": list(regret),
            })
    return rows


def normalise_regret(regret: list[float]) -> list[float] | None:
    if len(regret) < 2:
        return None
    final = regret[-1]
    start = regret[0]
    span = final - start
    if abs(span) < 1e-12:
        return [0.0] * len(regret)
    return [(r - start) / span for r in regret]


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply as _apply_paper_style
    _apply_paper_style()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip-rel-gain", type=float, default=2.0,
                    help="Clip relative_gain to ±this for the histogram only "
                         "(extreme problem-specific scales). Default: 2.0")
    args = ap.parse_args(argv)

    rows = collect(args.roots)
    print(f"BO summaries: {len(rows)}")
    if not rows:
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    # per-target CSV
    with (args.out / "bo_per_target.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp", "target_dir", "target_id", "iteration",
                    "original_score", "baseline_score", "bo_best",
                    "gain", "relative_gain", "n_calls", "n_specs"])
        for r in rows:
            w.writerow([r["exp"], r["target_dir"], r["target_id"],
                        r["iteration"], r["original_score"],
                        r["baseline_score"], r["bo_best"], r["gain"],
                        f"{r['relative_gain']:.6g}"
                        if r["relative_gain"] is not None else "",
                        r["n_calls"], r["n_specs"]])

    # --- Fig 1: gain distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rel_gains = [r["relative_gain"] for r in rows
                 if r["relative_gain"] is not None]
    rel_clipped = [max(min(v, args.clip_rel_gain), -args.clip_rel_gain)
                   for v in rel_gains]
    axes[0].hist(rel_clipped, bins=20, color="#4C78A8",
                 edgecolor="white", linewidth=0.6)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel(f"relative gain  (BO_best - original) / |original|"
                       f"  [clipped at ±{args.clip_rel_gain}]")
    axes[0].set_ylabel("# BO targets")
    axes[0].set_title(f"BO ceiling: gain over original program "
                      f"(median={np.median(rel_gains):+.3f}, "
                      f"n={len(rel_gains)})")
    axes[0].grid(True, alpha=0.3)

    # sign breakdown
    pos = sum(1 for r in rows if r["gain"] is not None and r["gain"] > 0)
    zero = sum(1 for r in rows if r["gain"] is not None and r["gain"] == 0)
    neg = sum(1 for r in rows if r["gain"] is not None and r["gain"] < 0)
    axes[1].bar(["BO improves", "no change", "BO regresses"],
                [pos, zero, neg],
                color=["#2ca02c", "#cccccc", "#d62728"])
    for i, v in enumerate([pos, zero, neg]):
        axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=10)
    axes[1].set_ylabel("# BO targets")
    axes[1].set_title("did BO find a better program than skydiscover's original?")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.out / "bo_gain_distribution.png", dpi=180)
    plt.close(fig)

    # --- Fig 2: normalised regret curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    norms: list[list[float]] = []
    for r in rows:
        n = normalise_regret(r["regret_curve"])
        if n is None:
            continue
        norms.append(n)
        ax.plot(range(len(n)), n, color="#aab7d4", alpha=0.45,
                linewidth=0.9)
    if norms:
        L = max(len(n) for n in norms)
        # align lengths (BO is fixed n_calls=24 so this is usually trivial)
        arr = np.array([n + [n[-1]] * (L - len(n)) for n in norms])
        med = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        xs = np.arange(L)
        ax.fill_between(xs, q25, q75, color="#3b5b9a", alpha=0.25,
                        label=f"IQR (n={len(norms)} targets)")
        ax.plot(xs, med, color="#1f3b73", linewidth=2.6, label="median")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=0.8,
               label="final BO best")
    ax.axhline(0.0, linestyle="--", color="grey", linewidth=0.8,
               label="initial baseline")
    ax.set_xlabel("BO call (0..n_calls-1)")
    ax.set_ylabel("normalised best-so-far  (0 = initial, 1 = final)")
    ax.set_title("BO regret curves — normalised per target")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "bo_regret_curves.png", dpi=180)
    plt.close(fig)

    # --- Summary table ---
    with (args.out / "bo_summary_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        n = len(rows)
        gains_pos = [r["relative_gain"] for r in rows
                     if r["gain"] is not None and r["gain"] > 0
                     and r["relative_gain"] is not None]
        rel_all = [r["relative_gain"] for r in rows
                   if r["relative_gain"] is not None]
        # how many calls until 90% of the final gain on average
        until_90 = []
        for r in rows:
            n_curve = normalise_regret(r["regret_curve"])
            if n_curve is None:
                continue
            for i, v in enumerate(n_curve):
                if v >= 0.9:
                    until_90.append(i)
                    break
        w.writerow(["n_targets", n])
        w.writerow(["fraction_positive", f"{pos/n:.3f}" if n else ""])
        w.writerow(["fraction_zero", f"{zero/n:.3f}" if n else ""])
        w.writerow(["fraction_negative", f"{neg/n:.3f}" if n else ""])
        w.writerow(["median_relative_gain_all",
                    f"{np.median(rel_all):+.4f}" if rel_all else ""])
        w.writerow(["median_relative_gain_positive",
                    f"{np.median(gains_pos):+.4f}" if gains_pos else ""])
        w.writerow(["median_calls_to_90pct",
                    f"{np.median(until_90):.1f}" if until_90 else ""])
        w.writerow(["median_n_specs",
                    f"{np.median([r['n_specs'] for r in rows]):.1f}"])

    print(f"wrote outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
