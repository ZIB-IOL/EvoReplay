"""Aggregate per-run cycle_classes + edit_classification summaries.

Reads pre-computed summaries that already live under <run_dir>/analysis/:
  cycle_classes.summary.json       — recycling rates per run
  edit_classification.summary.json — tuning vs structural edit buckets

Produces:
  cycling_rates.{png,csv}    distribution of recycle share by domain × backend
  edit_buckets.{png,csv}     stacked bars of edit composition per domain
  cycling_stats_table.csv    paper Tab 4
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evo_replay.static.aggregate_loc import backend_of, domain_of, find_runs

DOMAINS = ("ale", "math")
BACKENDS = ("openevolve_native", "evox", "gepa_native", "shinkaevolve")
COLOR_DOMAIN = {"ale": "#1f77b4", "math": "#d62728"}
EDIT_BUCKET_COLORS = {
    "pure-tuning":      "#9ecae1",
    "mostly-tuning":    "#3182bd",
    "mixed":            "#a1d99b",
    "mostly-structural": "#fd8d3c",
    "pure-structural":  "#a63603",
}


def collect(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run in find_runs(roots):
        cyc = run / "analysis" / "cycle_classes.summary.json"
        edt = run / "analysis" / "edit_classification.summary.json"
        if not cyc.exists() or not edt.exists():
            continue
        try:
            c = json.loads(cyc.read_text())
            e = json.loads(edt.read_text())
        except (OSError, ValueError):
            continue
        rows.append({
            "run": run.name,
            "backend": backend_of(run),
            "domain": domain_of(run),
            "n_edits": c.get("n_edits", 0),
            "n_recycle_events": c.get("n_recycle_events", 0),
            "recycle_share": (c.get("share_of_added_lines") or {}).get("any", 0.0),
            "recycle_literal": (c.get("share_of_added_lines") or {}).get("literal", 0.0),
            "recycle_tuning":  (c.get("share_of_added_lines") or {}).get("tuning", 0.0),
            "span_median": (c.get("span_stats") or {}).get("median", 0.0),
            "tuning_share_code": e.get("tuning_share_of_code_lines", 0.0),
            "buckets": e.get("buckets", {}),
            "n_score_up": e.get("n_score_up", 0),
            "n_score_down": e.get("n_score_down", 0),
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
    print(f"runs with cycling summaries: {len(rows)}")
    if not rows:
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    # --- raw per-run CSV ---
    with (args.out / "cycling_per_run.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "backend", "domain", "n_edits", "n_recycle_events",
                    "recycle_share", "recycle_literal", "recycle_tuning",
                    "span_median", "tuning_share_code",
                    "pure_tuning", "mostly_tuning", "mixed",
                    "mostly_structural", "pure_structural",
                    "n_score_up", "n_score_down"])
        for r in rows:
            b = r["buckets"] or {}
            w.writerow([r["run"], r["backend"], r["domain"], r["n_edits"],
                        r["n_recycle_events"],
                        f"{r['recycle_share']:.4f}",
                        f"{r['recycle_literal']:.4f}",
                        f"{r['recycle_tuning']:.4f}",
                        f"{r['span_median']:.2f}",
                        f"{r['tuning_share_code']:.4f}",
                        b.get("pure-tuning", 0), b.get("mostly-tuning", 0),
                        b.get("mixed", 0), b.get("mostly-structural", 0),
                        b.get("pure-structural", 0),
                        r["n_score_up"], r["n_score_down"]])

    # --- Fig: recycle share distribution by domain ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for d in DOMAINS:
        sub = [r for r in rows if r["domain"] == d]
        if not sub:
            continue
        shares = [r["recycle_share"] for r in sub]
        axes[0].hist(shares, bins=30, alpha=0.55, color=COLOR_DOMAIN[d],
                     label=f"{d} (n={len(shares)}, "
                           f"median={np.median(shares):.2f})")
        tunes = [r["tuning_share_code"] for r in sub]
        axes[1].hist(tunes, bins=30, alpha=0.55, color=COLOR_DOMAIN[d],
                     label=f"{d} (n={len(tunes)}, "
                           f"median={np.median(tunes):.2f})")
    axes[0].set_xlabel("recycle share of added lines")
    axes[0].set_ylabel("# runs")
    axes[0].set_title("recycling: fraction of new lines that re-introduce removed lines")
    axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("tuning-only share of edited code lines")
    axes[1].set_ylabel("# runs")
    axes[1].set_title("edit composition: fraction of edits that are pure tuning")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.suptitle("Cycling and edit composition (n="
                 f"{len(rows)} runs with pre-computed analyses)")
    fig.tight_layout()
    fig.savefig(args.out / "cycling_rates.png", dpi=180)
    plt.close(fig)

    # --- Fig: stacked bar of edit buckets per domain × backend ---
    cells = []  # (label, group)
    for d in DOMAINS:
        for b in BACKENDS:
            grp = [r for r in rows if r["domain"] == d and r["backend"] == b]
            if grp:
                cells.append((f"{b}\n×{d}", grp))
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.7
    keys = ["pure-tuning", "mostly-tuning", "mixed",
            "mostly-structural", "pure-structural"]
    bottoms = np.zeros(len(cells))
    for k in keys:
        heights = []
        for _, grp in cells:
            tot = sum((g["buckets"] or {}).get(k, 0) for g in grp)
            tot_all = sum(sum((g["buckets"] or {}).values()) for g in grp)
            heights.append(tot / max(tot_all, 1))
        ax.bar(range(len(cells)), heights, width=width, bottom=bottoms,
               color=EDIT_BUCKET_COLORS[k], label=k, edgecolor="white",
               linewidth=0.5)
        bottoms += np.array(heights)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells], rotation=0, fontsize=8)
    ax.set_ylabel("fraction of edits")
    ax.set_ylim(0, 1.05)
    ax.set_title("edit composition (pure-tuning ↔ pure-structural) per backend × domain")
    ax.legend(loc="lower right", fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(args.out / "edit_buckets.png", dpi=180)
    plt.close(fig)

    # --- Table: cycling stats by domain × backend ---
    with (args.out / "cycling_stats_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "domain", "n_runs",
                    "recycle_share_q25", "recycle_share_q50", "recycle_share_q75",
                    "tuning_share_q25", "tuning_share_q50", "tuning_share_q75",
                    "median_recycle_span"])
        for b in BACKENDS:
            for d in DOMAINS:
                grp = [r for r in rows if r["domain"] == d and r["backend"] == b]
                if not grp:
                    continue
                rs = [g["recycle_share"] for g in grp]
                ts = [g["tuning_share_code"] for g in grp]
                ss = [g["span_median"] for g in grp if g["span_median"] > 0]
                def q(arr, p): return np.quantile(arr, p) if arr else float("nan")
                w.writerow([b, d, len(grp),
                            f"{q(rs, 0.25):.3f}", f"{q(rs, 0.5):.3f}",
                            f"{q(rs, 0.75):.3f}",
                            f"{q(ts, 0.25):.3f}", f"{q(ts, 0.5):.3f}",
                            f"{q(ts, 0.75):.3f}",
                            f"{q(ss, 0.5):.2f}" if ss else ""])

    print(f"wrote outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
