"""Cycling time-series analysis.

Reads per-event + per-program output of `classify_cycles.py` and produces
trajectory-aware metrics that answer:

  1. Does cycling rate change over time within a run?
  2. What is the half-life of a recycled line — short churn vs long rediscovery?
  3. Are cycles concentrated around best-so-far events?

Outputs (under <run_dir>/analysis/cycle_timeseries/):
    per_iteration.csv        iteration, n_added, n_lit, n_tun, n_triv, frac_*
    rolling_window.csv       rolling-window-smoothed cycling rates
    span_distribution.csv    histogram of (added_iter - removed_iter) per recycled line
    summary.json             slope of cycling vs iteration, span median/quantiles,
                              cycling spike around best-so-far events
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from evo_replay.core.checkpoints import best_so_far_targets, load_programs


def per_iteration_rates(programs_csv: Path) -> List[Dict[str, Any]]:
    """Aggregate the per-program CSV by added-iteration."""
    by_iter: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"n_progs": 0, "added": 0, "lit": 0, "tun": 0, "triv": 0}
    )
    with programs_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                it = int(row["iteration"])
                d = by_iter[it]
                d["n_progs"] += 1
                d["added"] += int(row["total_added"])
                d["lit"] += int(row["literal_added"])
                d["tun"] += int(row["tuning_added"])
                d["triv"] += int(row["trivial_added"])
            except (KeyError, ValueError):
                continue

    rows: List[Dict[str, Any]] = []
    for it in sorted(by_iter):
        d = by_iter[it]
        a = d["added"] or 1
        rows.append({
            "iteration": it,
            "n_programs": d["n_progs"],
            "n_added": d["added"],
            "n_literal": d["lit"],
            "n_tuning": d["tun"],
            "n_trivial": d["triv"],
            "frac_literal": d["lit"] / a,
            "frac_tuning": d["tun"] / a,
            "frac_trivial": d["triv"] / a,
            "frac_any": (d["lit"] + d["tun"] + d["triv"]) / a,
        })
    return rows


def _rolling_mean(xs: List[float], window: int) -> List[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


def rolling_smooth(per_iter: List[Dict[str, Any]], window: int = 5) -> List[Dict[str, Any]]:
    if not per_iter:
        return []
    iters = [r["iteration"] for r in per_iter]
    series: Dict[str, List[float]] = {}
    for k in ("frac_literal", "frac_tuning", "frac_trivial", "frac_any"):
        smoothed = _rolling_mean([r[k] for r in per_iter], window)
        series[k] = smoothed
    return [
        {"iteration": it, **{k: series[k][i] for k in series}}
        for i, it in enumerate(iters)
    ]


def linear_slope(per_iter: List[Dict[str, Any]], key: str) -> float:
    """OLS slope of `frac_*` vs iteration."""
    xs = [r["iteration"] for r in per_iter]
    ys = [r[key] for r in per_iter]
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def span_distribution(events_json: Path) -> Dict[str, Any]:
    events = json.loads(events_json.read_text())
    spans = [e["span"] for e in events if isinstance(e.get("span"), int)]
    if not spans:
        return {"n": 0}
    spans.sort()
    return {
        "n": len(spans),
        "min": spans[0],
        "max": spans[-1],
        "mean": statistics.mean(spans),
        "median": statistics.median(spans),
        "p25": spans[len(spans) // 4],
        "p75": spans[3 * len(spans) // 4],
        "p90": spans[int(len(spans) * 0.9)],
        "p99": spans[int(len(spans) * 0.99)] if len(spans) > 100 else spans[-1],
        "histogram_bins": [
            ("1", sum(1 for s in spans if s <= 1)),
            ("2-5", sum(1 for s in spans if 2 <= s <= 5)),
            ("6-10", sum(1 for s in spans if 6 <= s <= 10)),
            ("11-20", sum(1 for s in spans if 11 <= s <= 20)),
            ("21-50", sum(1 for s in spans if 21 <= s <= 50)),
            ("51+", sum(1 for s in spans if s >= 51)),
        ],
    }


def post_breakthrough_spikes(
    per_iter: List[Dict[str, Any]],
    best_so_far_iters: List[int],
    *,
    window_before: int = 5,
    window_after: int = 5,
    key: str = "frac_any",
) -> Dict[str, Any]:
    """For each best-so-far iter, compare cycling rate before vs after."""
    by_iter = {r["iteration"]: r[key] for r in per_iter}
    deltas: List[Dict[str, Any]] = []
    for bsf in best_so_far_iters:
        before = [by_iter[i] for i in range(bsf - window_before, bsf)
                  if i in by_iter]
        after = [by_iter[i] for i in range(bsf + 1, bsf + 1 + window_after)
                 if i in by_iter]
        if not before or not after:
            continue
        deltas.append({
            "iteration": bsf,
            "mean_before": statistics.mean(before),
            "mean_after": statistics.mean(after),
            "delta": statistics.mean(after) - statistics.mean(before),
        })
    if not deltas:
        return {"n_events": 0}
    return {
        "n_events": len(deltas),
        "mean_before": statistics.mean(d["mean_before"] for d in deltas),
        "mean_after": statistics.mean(d["mean_after"] for d in deltas),
        "mean_delta": statistics.mean(d["delta"] for d in deltas),
        "events": deltas,
    }


def _bsf_iters(run_dir: Path) -> List[int]:
    progs = load_programs(run_dir)
    return [int(p.get("iteration_found") or 0)
            for p in best_so_far_targets(progs)]


def write_outputs(per_iter: List[Dict[str, Any]],
                  smoothed: List[Dict[str, Any]],
                  spans: Dict[str, Any],
                  bsf_summary: Dict[str, Any],
                  out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if per_iter:
        keys = list(per_iter[0].keys())
        with (out_dir / "per_iteration.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_iter)

    if smoothed:
        keys = list(smoothed[0].keys())
        with (out_dir / "rolling_window.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(smoothed)

    summary = {
        "n_iterations": len(per_iter),
        "slope_any": linear_slope(per_iter, "frac_any"),
        "slope_literal": linear_slope(per_iter, "frac_literal"),
        "slope_tuning": linear_slope(per_iter, "frac_tuning"),
        "spans": spans,
        "post_breakthrough": bsf_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def plot_timeseries(per_iter: List[Dict[str, Any]],
                    smoothed: List[Dict[str, Any]],
                    bsf_iters: List[int],
                    out_path: Path) -> None:
    if not per_iter:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    iters = [r["iteration"] for r in per_iter]
    ax.scatter(iters, [r["frac_any"] for r in per_iter],
               s=10, alpha=0.3, color="#888", label="raw")
    if smoothed:
        sm_iters = [r["iteration"] for r in smoothed]
        ax.plot(sm_iters, [r["frac_literal"] for r in smoothed],
                color="#1f77b4", linewidth=2, label="literal (rolling-5)")
        ax.plot(sm_iters, [r["frac_tuning"] for r in smoothed],
                color="#cc6677", linewidth=2, label="tuning (rolling-5)")
        ax.plot(sm_iters, [r["frac_any"] for r in smoothed],
                color="black", linewidth=1, linestyle="--", label="any (rolling-5)")
    for bsf in bsf_iters:
        ax.axvline(bsf, color="green", alpha=0.4, linestyle=":", linewidth=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("fraction of added lines that are recycled")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_title("Cycling rate over time (green: best-so-far events)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()

    programs_csv = run_dir / "analysis/cycle_classes.programs.csv"
    events_json = run_dir / "analysis/cycle_classes.json"
    if not programs_csv.exists() or not events_json.exists():
        sys.exit(f"Missing cycle_classes outputs in {run_dir}/analysis/. "
                 f"Run `evo_replay.cycling.classify_cycles` first.")

    out_dir = run_dir / "analysis/cycle_timeseries"
    per_iter = per_iteration_rates(programs_csv)
    smoothed = rolling_smooth(per_iter, window=args.window)
    spans = span_distribution(events_json)
    bsf_iters = _bsf_iters(run_dir)
    bsf_summary = post_breakthrough_spikes(per_iter, bsf_iters)
    write_outputs(per_iter, smoothed, spans, bsf_summary, out_dir)

    if not args.no_plot:
        try:
            plot_timeseries(per_iter, smoothed, bsf_iters,
                            out_dir / "timeseries.png")
        except Exception as exc:
            print(f"plot skipped: {exc}", file=sys.stderr)

    print(f"n_iterations: {len(per_iter)}")
    print(f"slope (any): {linear_slope(per_iter, 'frac_any'):+.5f}/iter")
    print(f"slope (lit): {linear_slope(per_iter, 'frac_literal'):+.5f}/iter")
    print(f"slope (tun): {linear_slope(per_iter, 'frac_tuning'):+.5f}/iter")
    print(f"span median: {spans.get('median', 'n/a')}  "
          f"p25/p75: {spans.get('p25','?')}/{spans.get('p75','?')}  "
          f"max: {spans.get('max','?')}")
    if bsf_summary.get("n_events"):
        print(f"post-bsf cycling Δ: {bsf_summary['mean_delta']:+.4f}  "
              f"(before {bsf_summary['mean_before']:.3f}, "
              f"after {bsf_summary['mean_after']:.3f}, n={bsf_summary['n_events']})")
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
