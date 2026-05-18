"""Compute the 8 paper-ready figures/tables from static analysis alone.

Walks every run under one or more roots, extracts per-program features
(score, lines, hparam count, parent_id, validity, iteration_found), then
produces:

    1_loc_trajectory_grid.png        normalised LOC vs iter, backend × domain
    2_hparam_trajectory_grid.png     normalised hparam count vs iter, b × d
    3_endofrun_table.csv             per (backend, domain): final LOC & hparam
                                     IQR + counts (paper Tab 1 source)
    4_lineage_depth_cdf.png          CDF of lineage depth of final best,
       lineage_depth_table.csv         by domain (and overall)
    5_time_to_best.png               iteration_found(best)/max_iter histogram
    6_n_best_events.png              distribution of # best-so-far updates
    7_loc_delta_per_event.png        LOC delta per best-so-far event
    8_validity_rate.png              mean fraction of valid programs per iter
    per_run_features.csv             raw per-run features (for own analyses)

Usage:
    uv run python -m evo_replay.static.paper_stats \\
        /path/to/evo_trace_anon --out /tmp/paper_stats
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
from evo_replay.static.hyperparameter_counts import (
    count_hyperparameter_candidates, is_valid_program,
)


# Backends to include in figures; private_view is too sparse + different scope.
BACKENDS = ("openevolve_native", "evox", "gepa_native", "shinkaevolve")
DOMAINS = ("ale", "math")
COLOR_DOMAIN = {"ale": "#1f77b4", "math": "#d62728"}

# Tokens that mark the start of model/config tail in a run name; the problem
# id is everything before the first one of these.
_MODEL_HEAD_TOKENS = {
    "deepseek", "dpsk", "dschat", "claude", "gemini", "gpt", "local", "gflash",
}


def problem_of(name: str) -> str:
    """Strip model + iteration + timestamp tail and return the problem id.

    e.g. 'circle_packing_deepseek-deepseek-reasoner_100_056bfb'
         -> 'circle_packing'
         'ale_bench_ahc008_deepseek-...'  -> 'ale_bench_ahc008'
         'ahc015_deepseek-reasoner_nodiff_100_...' -> 'ahc015'
         'ahc008' (private_view) -> 'ahc008'
    """
    parts = name.split("_")
    out: list[str] = []
    for p in parts:
        head = p.split("-", 1)[0]
        if head in _MODEL_HEAD_TOKENS:
            break
        if p.isdigit() and out:
            break
        out.append(p)
    return "_".join(out) if out else name


# ---------- per-run feature extraction ------------------------------------

def extract_features(run_dir: Path) -> dict[str, Any] | None:
    progs = load_programs(run_dir)
    if not progs:
        return None
    lang = detect_language(run_dir) or "python"

    rows = []
    for pid, p in progs.items():
        sol = p.get("solution")
        if not isinstance(sol, str) or not sol:
            continue
        score = (p.get("metrics") or {}).get("combined_score")
        if not isinstance(score, (int, float)):
            score_f = None
        else:
            score_f = float(score)
        lines = len(sol.splitlines())
        try:
            n_h, _, _ = count_hyperparameter_candidates(sol, lang)
        except Exception:
            n_h = None
        rows.append({
            "id": str(pid),
            "iteration": int(p.get("iteration_found") or 0),
            "parent_id": p.get("parent_id"),
            "score": score_f,
            "lines": lines,
            "hparams": n_h,
            "valid": is_valid_program(p),
        })
    if not rows:
        return None
    rows.sort(key=lambda r: (r["iteration"], r["id"]))

    iters = sorted({r["iteration"] for r in rows})
    if len(iters) < 2:
        return None
    max_iter = max(iters)

    # seed: median lines/hparams over iter=0
    iter0 = [r for r in rows if r["iteration"] == iters[0]]
    seed_lines = float(np.median([r["lines"] for r in iter0]))
    seed_hparams = float(np.median(
        [r["hparams"] for r in iter0 if r["hparams"] is not None] or [0.0]))

    # by-iter aggregates + best-so-far trajectory (using only scored progs)
    by_iter: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_iter[r["iteration"]].append(r)

    bsf_events = []  # each event = (iter, score, lines, hparams, program_id)
    best_score = -float("inf")
    best_event = None
    for it in iters:
        for r in by_iter[it]:
            if r["score"] is None:
                continue
            if r["score"] > best_score:
                best_score = r["score"]
                evt = {
                    "iteration": it,
                    "score": r["score"],
                    "lines": r["lines"],
                    "hparams": r["hparams"],
                    "program_id": r["id"],
                }
                bsf_events.append(evt)
                best_event = evt

    # iteration trajectory (median lines, hparam, validity, best-so-far)
    traj = []
    bsf_lines = seed_lines
    bsf_hparams = seed_hparams
    bsf_score = -float("inf")
    for it in iters:
        rs = by_iter[it]
        for r in rs:
            if r["score"] is not None and r["score"] > bsf_score:
                bsf_score = r["score"]
                bsf_lines = float(r["lines"])
                if r["hparams"] is not None:
                    bsf_hparams = float(r["hparams"])
        scored = [r for r in rs if r["score"] is not None]
        valid = [r for r in rs if r["valid"]]
        med_lines = float(np.median([r["lines"] for r in scored])) \
            if scored else None
        med_h = (float(np.median([r["hparams"] for r in scored
                                  if r["hparams"] is not None]))
                 if scored else None)
        traj.append({
            "iteration": it,
            "x_norm": it / max_iter if max_iter > 0 else 0.0,
            "n_programs": len(rs),
            "n_scored": len(scored),
            "n_valid": len(valid),
            "validity_rate": (len(valid) / len(rs)) if rs else 0.0,
            "median_lines": med_lines,
            "median_lines_norm": (med_lines / seed_lines) if med_lines and seed_lines else None,
            "median_hparams": med_h,
            "median_hparams_norm": (med_h / max(seed_hparams, 1.0)) if med_h is not None else None,
            "bsf_lines": bsf_lines,
            "bsf_lines_norm": bsf_lines / seed_lines if seed_lines else None,
            "bsf_hparams": bsf_hparams,
            "bsf_hparams_norm": bsf_hparams / max(seed_hparams, 1.0),
            "bsf_score": bsf_score if bsf_score != -float("inf") else None,
        })

    # lineage of final best: walk parent_id chain
    progs_by_id = {r["id"]: r for r in rows}
    lineage_ids: list[str] = []
    if best_event is not None:
        cur = best_event["program_id"]
        seen: set[str] = set()
        while cur and cur in progs_by_id and cur not in seen:
            seen.add(cur)
            lineage_ids.append(cur)
            cur = progs_by_id[cur].get("parent_id")
    lineage_depth = len(lineage_ids)

    # "active" lineage events: only the bsf_events that share lineage with final best
    bsf_in_lineage = sum(1 for e in bsf_events if e["program_id"] in set(lineage_ids))

    # Per-best-so-far event LOC delta
    deltas = []
    if len(bsf_events) >= 2:
        for prev, cur in zip(bsf_events[:-1], bsf_events[1:]):
            deltas.append({
                "from_iter": prev["iteration"],
                "to_iter": cur["iteration"],
                "delta_lines": cur["lines"] - prev["lines"],
                "delta_hparams": (cur["hparams"] - prev["hparams"]
                                  if (cur["hparams"] is not None
                                      and prev["hparams"] is not None) else None),
                "delta_score": cur["score"] - prev["score"],
            })

    return {
        "run_dir": str(run_dir),
        "backend": backend_of(run_dir),
        "domain": domain_of(run_dir),
        "language": lang,
        "max_iter": max_iter,
        "n_programs": len(rows),
        "seed_lines": seed_lines,
        "seed_hparams": seed_hparams,
        "n_bsf_events": len(bsf_events),
        "n_bsf_events_in_lineage": bsf_in_lineage,
        "lineage_depth": lineage_depth,
        "final_best": best_event,
        "final_best_x_norm": (best_event["iteration"] / max_iter
                              if best_event and max_iter else None),
        "trajectory": traj,
        "bsf_events": bsf_events,
        "bsf_deltas": deltas,
    }


# ---------- shared helpers -----------------------------------------------

def _plot_traj_panel(ax, trajs, metric, title, color, n_bins=50, log_y=True):
    for t in trajs:
        xs = [r["x_norm"] for r in t["trajectory"]]
        ys = [r[metric] for r in t["trajectory"]]
        xs_y = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not xs_y:
            continue
        ax.plot(*zip(*xs_y), color=color, alpha=0.18, linewidth=0.7)
    # build same shape rows for aggregate()
    coerced = [{
        "rows": [{"x_norm": r["x_norm"], "metric": r[metric]}
                 for r in t["trajectory"] if r[metric] is not None],
    } for t in trajs]
    # remap key from "metric" → expected column
    for c in coerced:
        for r in c["rows"]:
            r[metric] = r.pop("metric")
    agg = aggregate(coerced, metric=metric, n_bins=n_bins)
    ax.fill_between(agg["x"], agg["q25"], agg["q75"], color=color, alpha=0.30,
                    label=f"IQR (n={len(trajs)})")
    ax.plot(agg["x"], agg["q50"], color=color, linewidth=2.4, label="median")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("iter / max_iter")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=8)


def _grid_by_backend_domain(features, metric, color_for_domain, n_bins=50,
                            log_y=True):
    by = {(b, d): [] for b in BACKENDS for d in DOMAINS}
    for f in features:
        if f["backend"] in BACKENDS and f["domain"] in DOMAINS:
            by[(f["backend"], f["domain"])].append(f)
    fig, axes = plt.subplots(len(BACKENDS), len(DOMAINS),
                             figsize=(11, 3.0 * len(BACKENDS)),
                             sharex=True, sharey=True, squeeze=False)
    for i, b in enumerate(BACKENDS):
        for j, d in enumerate(DOMAINS):
            cell = by[(b, d)]
            ax = axes[i][j]
            if not cell:
                ax.set_visible(False)
                continue
            color = color_for_domain.get(d, "#666666")
            _plot_traj_panel(ax, cell, metric,
                             f"{b} × {d}  (n={len(cell)})",
                             color, n_bins=n_bins, log_y=log_y)
    fig.tight_layout()
    return fig


# ---------- main ---------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply as _apply_paper_style
    _apply_paper_style()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bins", type=int, default=50)
    args = ap.parse_args(argv)

    runs = find_runs(args.roots)
    print(f"discovered {len(runs)} runs")
    features: list[dict[str, Any]] = []
    skipped = []
    for run in runs:
        f = extract_features(run)
        if f is None:
            skipped.append(str(run))
            continue
        features.append(f)
    print(f"runs with features: {len(features)} (skipped {len(skipped)})")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out

    # per_run_features.csv
    with (out / "per_run_features.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_dir", "backend", "domain", "language", "max_iter",
            "n_programs", "seed_lines", "seed_hparams",
            "n_bsf_events", "n_bsf_events_in_lineage",
            "lineage_depth",
            "final_best_iter", "final_best_x_norm",
            "final_best_score", "final_best_lines", "final_best_hparams",
            "loc_norm_at_end", "hparam_norm_at_end",
        ])
        for ft in features:
            fb = ft["final_best"] or {}
            last = ft["trajectory"][-1] if ft["trajectory"] else {}
            w.writerow([
                ft["run_dir"].rsplit("/", 1)[-1], ft["backend"], ft["domain"],
                ft["language"], ft["max_iter"], ft["n_programs"],
                ft["seed_lines"], ft["seed_hparams"],
                ft["n_bsf_events"], ft["n_bsf_events_in_lineage"],
                ft["lineage_depth"],
                fb.get("iteration"), ft.get("final_best_x_norm"),
                fb.get("score"), fb.get("lines"), fb.get("hparams"),
                last.get("bsf_lines_norm"), last.get("bsf_hparams_norm"),
            ])

    # ---- 1: LOC trajectory grid ----
    fig = _grid_by_backend_domain(features, "bsf_lines_norm",
                                  COLOR_DOMAIN, n_bins=args.n_bins, log_y=True)
    fig.suptitle("Fig 1: best-so-far LOC (× seed), by backend × domain", y=1.0)
    fig.savefig(out / "1_loc_trajectory_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- 2: hparam trajectory grid ----
    fig = _grid_by_backend_domain(features, "bsf_hparams_norm",
                                  COLOR_DOMAIN, n_bins=args.n_bins, log_y=True)
    fig.suptitle("Fig 2: best-so-far hparam count (× seed), by backend × domain",
                 y=1.0)
    fig.savefig(out / "2_hparam_trajectory_grid.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 3: end-of-run table ----
    with (out / "3_endofrun_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "backend", "domain", "n_runs",
            "loc_norm_q25", "loc_norm_q50", "loc_norm_q75",
            "hparam_norm_q25", "hparam_norm_q50", "hparam_norm_q75",
            "lineage_depth_q25", "lineage_depth_q50", "lineage_depth_q75",
        ])
        for b in BACKENDS:
            for d in DOMAINS:
                cell = [ft for ft in features
                        if ft["backend"] == b and ft["domain"] == d]
                if not cell:
                    continue
                loc = [ft["trajectory"][-1]["bsf_lines_norm"]
                       for ft in cell if ft["trajectory"]
                       and ft["trajectory"][-1]["bsf_lines_norm"] is not None]
                hp = [ft["trajectory"][-1]["bsf_hparams_norm"]
                      for ft in cell if ft["trajectory"]
                      and ft["trajectory"][-1]["bsf_hparams_norm"] is not None]
                ld = [ft["lineage_depth"] for ft in cell]
                def q(arr):
                    return ([np.quantile(arr, q) for q in (0.25, 0.5, 0.75)]
                            if arr else [np.nan]*3)
                w.writerow([b, d, len(cell),
                            *[f"{v:.3f}" if v == v else "" for v in q(loc)],
                            *[f"{v:.3f}" if v == v else "" for v in q(hp)],
                            *[f"{v:.1f}" if v == v else "" for v in q(ld)]])

    # ---- 4: lineage depth CDF + table ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in DOMAINS:
        depths = sorted(ft["lineage_depth"] for ft in features
                        if ft["domain"] == d and ft["lineage_depth"] > 0)
        if not depths:
            continue
        xs = depths
        ys = np.linspace(1.0/len(depths), 1.0, len(depths))
        ax.step(xs, ys, where="post", linewidth=2.0,
                color=COLOR_DOMAIN[d],
                label=f"{d} (n={len(depths)}, median={int(np.median(depths))})")
    ax.set_xlabel("lineage depth of final best program")
    ax.set_ylabel("cumulative fraction of runs")
    ax.set_title("Fig 4: lineage depth of final best (CDF, by domain)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "4_lineage_depth_cdf.png", dpi=180)
    plt.close(fig)

    with (out / "lineage_depth_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "min", "q25", "median", "q75", "max", "mean"])
        groups = [("all", features)] + \
                 [(d, [ft for ft in features if ft["domain"] == d]) for d in DOMAINS] + \
                 [(b, [ft for ft in features if ft["backend"] == b]) for b in BACKENDS]
        for name, grp in groups:
            ds = [ft["lineage_depth"] for ft in grp if ft["lineage_depth"] > 0]
            if not ds:
                continue
            w.writerow([name, len(ds), min(ds),
                        f"{np.quantile(ds, 0.25):.1f}",
                        f"{np.median(ds):.1f}",
                        f"{np.quantile(ds, 0.75):.1f}",
                        max(ds), f"{np.mean(ds):.2f}"])

    # ---- 5: time-to-best (iter_found(best) / max_iter) histogram ----
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)
    for d in DOMAINS:
        xs = [ft["final_best_x_norm"] for ft in features
              if ft["domain"] == d and ft["final_best_x_norm"] is not None]
        if not xs:
            continue
        ax.hist(xs, bins=bins, alpha=0.55, color=COLOR_DOMAIN[d],
                label=f"{d} (n={len(xs)}, median={np.median(xs):.2f})")
    ax.set_xlabel("iteration of final best / max iteration")
    ax.set_ylabel("# runs")
    ax.set_title("Fig 5: when in the run is the final best discovered?")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "5_time_to_best.png", dpi=180)
    plt.close(fig)

    # ---- 6: # best-so-far events per run ----
    fig, ax = plt.subplots(figsize=(8, 5))
    max_n = max((ft["n_bsf_events"] for ft in features), default=1)
    bins = np.linspace(0, max_n + 1, min(max_n + 2, 40))
    for d in DOMAINS:
        xs = [ft["n_bsf_events"] for ft in features if ft["domain"] == d]
        if not xs:
            continue
        ax.hist(xs, bins=bins, alpha=0.55, color=COLOR_DOMAIN[d],
                label=f"{d} (n={len(xs)}, median={int(np.median(xs))})")
    ax.set_xlabel("# best-so-far updates in run")
    ax.set_ylabel("# runs")
    ax.set_title("Fig 6: distribution of # best-so-far events")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "6_n_best_events.png", dpi=180)
    plt.close(fig)

    # ---- 7: per-event LOC delta distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in DOMAINS:
        deltas = [e["delta_lines"] for ft in features
                  if ft["domain"] == d for e in ft["bsf_deltas"]]
        if not deltas:
            continue
        ax.hist(deltas, bins=40, alpha=0.55, color=COLOR_DOMAIN[d],
                label=f"{d} (n_events={len(deltas)}, median={int(np.median(deltas))})")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Δ lines (best-so-far step → next step)")
    ax.set_ylabel("# events")
    ax.set_title("Fig 7: code-length change per best-so-far update")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "7_loc_delta_per_event.png", dpi=180)
    plt.close(fig)

    # ---- 8: seed distributions by domain (LOC + hparam) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for d in DOMAINS:
        sub = [ft for ft in features if ft["domain"] == d]
        if not sub:
            continue
        seeds_lines = [ft["seed_lines"] for ft in sub if ft["seed_lines"] > 0]
        seeds_h = [ft["seed_hparams"] for ft in sub]
        axes[0].hist(seeds_lines, bins=30, alpha=0.55, color=COLOR_DOMAIN[d],
                     label=f"{d} (n={len(seeds_lines)}, "
                           f"median={int(np.median(seeds_lines))})")
        axes[1].hist(seeds_h, bins=30, alpha=0.55, color=COLOR_DOMAIN[d],
                     label=f"{d} (n={len(seeds_h)}, "
                           f"median={int(np.median(seeds_h))})")
    axes[0].set_xlabel("seed program length (lines)")
    axes[0].set_ylabel("# runs")
    axes[0].set_title("seed LOC distribution")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("seed hyperparameter count")
    axes[1].set_ylabel("# runs")
    axes[1].set_title("seed hparam-count distribution")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle("Fig 8: seeds are the asymmetry — ALE seeds start long, "
                 "math seeds start short")
    fig.tight_layout()
    fig.savefig(out / "8_seed_distributions.png", dpi=180)
    plt.close(fig)

    # ---- 9: lineage depth vs final score (normalised within each problem) ---
    by_problem: dict[str, list] = defaultdict(list)
    for ft in features:
        run_name = ft["run_dir"].rsplit("/", 1)[-1]
        ft["_problem"] = problem_of(run_name)
        by_problem[ft["_problem"]].append(ft)

    # min-max normalise final_best.score within each problem (>=2 runs)
    for prob, grp in by_problem.items():
        scores = [ft["final_best"]["score"] for ft in grp
                  if ft["final_best"]
                  and isinstance(ft["final_best"].get("score"), (int, float))]
        if len(scores) < 2:
            for ft in grp:
                ft["_score_norm"] = None
            continue
        lo, hi = min(scores), max(scores)
        span = hi - lo
        for ft in grp:
            s = (ft["final_best"] or {}).get("score")
            if not isinstance(s, (int, float)) or span == 0:
                ft["_score_norm"] = None
            else:
                ft["_score_norm"] = (s - lo) / span

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # left: scatter colored by domain, with regression line per domain
    ax = axes[0]
    for d in DOMAINS:
        xs, ys = [], []
        for ft in features:
            if ft["domain"] != d:
                continue
            if ft["_score_norm"] is None:
                continue
            xs.append(ft["lineage_depth"])
            ys.append(ft["_score_norm"])
        if not xs:
            continue
        ax.scatter(xs, ys, alpha=0.55, s=36, color=COLOR_DOMAIN[d],
                   edgecolor="white", linewidth=0.5,
                   label=f"{d} (n={len(xs)}, "
                         f"r={np.corrcoef(xs, ys)[0,1]:.2f})")
        # least-squares fit
        if len(xs) >= 3:
            coef = np.polyfit(xs, ys, 1)
            xs_fit = np.linspace(min(xs), max(xs), 50)
            ax.plot(xs_fit, np.polyval(coef, xs_fit),
                    color=COLOR_DOMAIN[d], linewidth=1.5, alpha=0.7,
                    linestyle="--")
    ax.set_xlabel("lineage depth of final best")
    ax.set_ylabel("final score (min-max normalised within problem)")
    ax.set_title("scatter + per-domain linear fit")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # right: scatter colored by backend (controls for backend confound)
    ax = axes[1]
    backend_colors = {"openevolve_native": "#1f77b4", "evox": "#2ca02c",
                      "gepa_native": "#ff7f0e", "shinkaevolve": "#9467bd"}
    for b in BACKENDS:
        xs, ys = [], []
        for ft in features:
            if ft["backend"] != b:
                continue
            if ft["_score_norm"] is None:
                continue
            xs.append(ft["lineage_depth"])
            ys.append(ft["_score_norm"])
        if not xs:
            continue
        ax.scatter(xs, ys, alpha=0.55, s=36, color=backend_colors.get(b),
                   edgecolor="white", linewidth=0.5,
                   label=f"{b} (n={len(xs)})")
    ax.set_xlabel("lineage depth of final best")
    ax.set_ylabel("final score (min-max normalised within problem)")
    ax.set_title("same data, coloured by backend")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Fig 9: deeper lineage → better outcome? (per-problem score "
                 "normalisation removes the cross-task scale)")
    fig.tight_layout()
    fig.savefig(out / "9_lineage_depth_vs_score.png", dpi=180)
    plt.close(fig)

    # CSV with the underlying numbers
    with (out / "9_lineage_depth_vs_score.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "problem", "domain", "backend",
                    "lineage_depth", "final_score_raw", "final_score_norm"])
        for ft in features:
            run_name = ft["run_dir"].rsplit("/", 1)[-1]
            fb = ft["final_best"] or {}
            w.writerow([run_name, ft["_problem"], ft["domain"], ft["backend"],
                        ft["lineage_depth"], fb.get("score"),
                        ft["_score_norm"]])

    # ---- 10: per-event Δ-LOC paired with Δ-score (normalised per run) -------
    # For each best-so-far transition, normalise delta_score by the run's
    # total positive score gain (so each run's events sum to <=1).
    events: list[dict[str, Any]] = []
    for ft in features:
        deltas = ft["bsf_deltas"]
        if not deltas:
            continue
        total_gain = sum(d["delta_score"] for d in deltas
                         if d["delta_score"] is not None and d["delta_score"] > 0)
        if total_gain <= 0:
            continue
        for d in deltas:
            if d["delta_score"] is None:
                continue
            events.append({
                "domain": ft["domain"],
                "backend": ft["backend"],
                "delta_lines": d["delta_lines"],
                "delta_score": d["delta_score"],
                "delta_score_norm": d["delta_score"] / total_gain,
            })

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.35)
    ax_h_lines = fig.add_subplot(gs[0, 0])
    ax_h_score = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, :])

    for d in DOMAINS:
        sub = [e for e in events if e["domain"] == d]
        if not sub:
            continue
        ax_h_lines.hist([e["delta_lines"] for e in sub], bins=40,
                        alpha=0.55, color=COLOR_DOMAIN[d],
                        label=f"{d} (n_events={len(sub)})")
        ax_h_score.hist([e["delta_score_norm"] for e in sub], bins=40,
                        alpha=0.55, color=COLOR_DOMAIN[d],
                        label=f"{d} (n_events={len(sub)})")
        ax_scatter.scatter([e["delta_lines"] for e in sub],
                           [e["delta_score_norm"] for e in sub],
                           alpha=0.40, s=20, color=COLOR_DOMAIN[d],
                           edgecolor="white", linewidth=0.3,
                           label=f"{d} (n={len(sub)})")
    ax_h_lines.axvline(0, color="black", linestyle="--", linewidth=0.7)
    ax_h_lines.set_xlabel("Δ lines per best-so-far step")
    ax_h_lines.set_ylabel("# events")
    ax_h_lines.set_title("Δ-LOC distribution")
    ax_h_lines.grid(True, alpha=0.3)
    ax_h_lines.legend(fontsize=8)

    ax_h_score.set_xlabel("Δ score per step (normalised: fraction of run's total gain)")
    ax_h_score.set_ylabel("# events")
    ax_h_score.set_title("Δ-score distribution")
    ax_h_score.grid(True, alpha=0.3)
    ax_h_score.legend(fontsize=8)

    ax_scatter.axvline(0, color="black", linestyle="--", linewidth=0.7,
                       alpha=0.5)
    ax_scatter.set_xlabel("Δ lines per step")
    ax_scatter.set_ylabel("Δ score per step (normalised)")
    ax_scatter.set_title("paired: gain per line added — "
                         "is more code change associated with bigger score steps?")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend()

    fig.suptitle("Fig 10: per-event Δ-LOC and Δ-score (per-run normalised)",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out / "10_event_loc_vs_score.png", dpi=180)
    plt.close(fig)

    with (out / "10_event_loc_vs_score.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "backend", "delta_lines",
                    "delta_score", "delta_score_norm"])
        for e in events:
            w.writerow([e["domain"], e["backend"], e["delta_lines"],
                        f"{e['delta_score']:.6g}",
                        f"{e['delta_score_norm']:.6f}"])

    # quick stats for the 9/10 outputs
    print()
    print("=== Fig 9 correlations (Spearman, lineage_depth vs score_norm) ===")
    try:
        from scipy.stats import spearmanr  # type: ignore
        for d in ("all",) + DOMAINS:
            sub = features if d == "all" else [
                ft for ft in features if ft["domain"] == d]
            xs = [ft["lineage_depth"] for ft in sub
                  if ft["_score_norm"] is not None]
            ys = [ft["_score_norm"] for ft in sub
                  if ft["_score_norm"] is not None]
            if len(xs) >= 3:
                rho, p = spearmanr(xs, ys)
                print(f"  {d:>4}: n={len(xs)}  ρ={rho:+.3f}  p={p:.3g}")
    except ImportError:
        pass

    print(f"wrote {len(features)} run features and 10 outputs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
