"""Public vs private ranking discrepancy on ALE problems.

Reads ``shinka_private_lineage.jsonl`` from refined shinkaevolve runs.
Each row is a best-so-far event w.r.t. the *public* eval, augmented with
the *private* (held-out) test set: ``public_score``, ``private_score``,
``private_rank``, ``private_performance``.

We answer: as the search improves the public score, does the private
ranking go with it — or against it?

Outputs:
  public_vs_private_table.csv     per-problem seed vs. final summary
  public_vs_private_grid.pdf      small multiples (one panel per problem):
                                  public score (left axis, ↑) and private
                                  rank (right axis, inverted so ↑ = better)
                                  along the public best-so-far chain
  public_vs_private_scatter.pdf   per-problem (Δpublic%, Δprivate_perf)
                                  with problem labels — diagonal = aligned
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


_BACKENDS = {"openevolve_native", "evox", "gepa_native", "shinkaevolve"}


def find_lineage_files(roots: list[Path]) -> list[Path]:
    """Find both file conventions used in the dataset:
       - refined:  ``<run>/shinka_private_lineage.jsonl``  (one row per line)
       - raw/legacy: ``<run>/private_eval_lineage.json``  ({lineage: [...]}).
    """
    out: list[Path] = []
    for root in roots:
        for pattern in ("shinka_private_lineage.jsonl",
                        "private_eval_lineage.json"):
            for f in root.rglob(pattern):
                try:
                    if f.stat().st_size > 0:
                        out.append(f)
                except OSError:
                    pass
    return sorted(set(out))


def parse_problem(run_name: str) -> str:
    """``ale_bench_ahc024_local-...`` -> ``ahc024`` (or run_name as fallback)."""
    parts = run_name.split("_")
    for p in parts:
        if p.startswith("ahc"):
            return p
    return run_name


def parse_backend(path: Path) -> str:
    """Walk up the path until we hit a known backend directory name."""
    for parent in path.parents:
        if parent.name in _BACKENDS:
            return parent.name
    return "unknown"


def _read_lineage(f: Path) -> list[dict[str, Any]]:
    """Both file formats decode to a list of dicts."""
    if f.suffix == ".jsonl":
        try:
            return [json.loads(l) for l in f.read_text().splitlines()
                    if l.strip()]
        except (OSError, ValueError):
            return []
    try:
        d = json.loads(f.read_text())
    except (OSError, ValueError):
        return []
    rows = d.get("lineage") if isinstance(d, dict) else None
    return rows if isinstance(rows, list) else []


def load_lineages(roots: list[Path]) -> list[dict[str, Any]]:
    out = []
    for f in find_lineage_files(roots):
        rows = _read_lineage(f)
        if not rows:
            continue
        out.append({
            "run": f.parent.name,
            "problem": parse_problem(f.parent.name),
            "backend": parse_backend(f),
            "source": str(f),
            "rows": rows,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply as _apply_paper_style
    _apply_paper_style()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", type=Path, nargs="+",
                    help="Refined-dataset roots to walk for "
                         "shinka_private_lineage.jsonl files.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    lineages = load_lineages(args.roots)
    print(f"runs with non-empty private lineage: {len(lineages)}")
    if not lineages:
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    # --- per-(backend, problem) summary ------------------------------
    rows_csv = []
    for L in lineages:
        seed, final = L["rows"][0], L["rows"][-1]
        ps = seed.get("public_score")
        pf = final.get("public_score")
        pct = ((pf - ps) / abs(ps) * 100
               if isinstance(ps, (int, float)) and isinstance(pf, (int, float))
               and ps else None)
        perf_s = seed.get("private_performance")
        perf_f = final.get("private_performance")
        rank_s = seed.get("private_rank")
        rank_f = final.get("private_rank")
        rows_csv.append({
            "backend": L["backend"],
            "problem": L["problem"],
            "n_events": len(L["rows"]),
            "pub_seed": ps, "pub_final": pf, "pub_delta_pct": pct,
            "perf_seed": perf_s, "perf_final": perf_f,
            "perf_delta": ((perf_f or 0) - (perf_s or 0)
                           if perf_s is not None and perf_f is not None
                           else None),
            "rank_seed": rank_s, "rank_final": rank_f,
            "rank_delta": ((rank_f or 0) - (rank_s or 0)
                           if rank_s is not None and rank_f is not None
                           else None),
        })

    with (args.out / "public_vs_private_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "problem", "n_events", "pub_seed", "pub_final",
                    "pub_delta_pct", "perf_seed", "perf_final", "perf_delta",
                    "rank_seed", "rank_final", "rank_delta"])
        for r in rows_csv:
            def fmt(v, p=None):
                if v is None: return ""
                return f"{v:+.2f}" if p else (f"{v:.2f}"
                                              if isinstance(v, float) else v)
            w.writerow([r["backend"], r["problem"], r["n_events"],
                        fmt(r["pub_seed"]), fmt(r["pub_final"]),
                        fmt(r["pub_delta_pct"], p=True),
                        r["perf_seed"], r["perf_final"], r["perf_delta"],
                        r["rank_seed"], r["rank_final"], r["rank_delta"]])

    # --- classification helpers -------------------------------------
    def _classify(r):
        if r["n_events"] < 2 or r["pub_delta_pct"] is None \
                or r["perf_delta"] is None or r["perf_seed"] is None:
            return "n/a"
        if r["pub_delta_pct"] > 0 and r["perf_delta"] < -200:
            return "overfit_severe"
        if r["pub_delta_pct"] > 0 and r["perf_delta"] < 0:
            return "overfit_mild"
        if r["pub_delta_pct"] > 0 and r["perf_delta"] > 0:
            return "aligned"
        return "other"

    def _fmt_int_signed(v):
        if v is None: return "---"
        return ("$" + f"{int(v):+,}".replace(",", "{,}") + "$")

    # --- per-(problem × backend) wide table -------------------------
    # rows = problems, columns = 4 backends; each cell shows Δperf,
    # bolded if overfit.
    backends_in_data = sorted({r["backend"] for r in rows_csv
                               if r["backend"] in _BACKENDS})
    problems = sorted({r["problem"] for r in rows_csv})

    # index: (problem, backend) -> row
    idx = {(r["problem"], r["backend"]): r for r in rows_csv}

    tex_lines = [
        r"% public vs. private ranking on ALE, all 4 frameworks.",
        r"% Cell content: \Delta priv. perf. (rating points), seed -> final",
        r"% along the public best-so-far chain. Bold = overfit (public up",
        r"% but private down). 'aligned' rows show both public and private up.",
        r"\begin{tabular}{l" + "r" * len(backends_in_data) + r"}",
        r"\toprule",
        "problem & " + " & ".join(b.replace("_native", "")
                                  .replace("shinkaevolve", "shinka")
                                  for b in backends_in_data)
                    + r" \\",
        r"\midrule",
    ]
    for prob in problems:
        cells = [prob]
        for b in backends_in_data:
            r = idx.get((prob, b))
            if r is None:
                cells.append("---")
                continue
            cls = _classify(r)
            txt = _fmt_int_signed(r["perf_delta"])
            if cls in ("overfit_severe", "overfit_mild"):
                txt = r"\textbf{" + txt + r"}"
            cells.append(txt)
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (args.out / "public_vs_private_table.tex").write_text(
        "\n".join(tex_lines) + "\n")

    # --- per-framework overfit-rate summary table ------------------
    tex2 = [
        r"% Per-framework summary: how often does the public BSF chain",
        r"% overfit (public up but private down) on ALE?",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"framework & runs scored & aligned & overfit (mild) & overfit (severe) \\",
        r"\midrule",
    ]
    for b in backends_in_data:
        sub = [r for r in rows_csv if r["backend"] == b]
        scored = [r for r in sub if _classify(r) != "n/a"]
        n = len(scored)
        n_align = sum(1 for r in scored if _classify(r) == "aligned")
        n_mild = sum(1 for r in scored if _classify(r) == "overfit_mild")
        n_sev = sum(1 for r in scored if _classify(r) == "overfit_severe")
        label = b.replace("_native", "").replace("shinkaevolve", "shinka")
        tex2.append(f"{label} & {n} & {n_align} & {n_mild} & {n_sev} \\\\")
    tex2 += [r"\bottomrule", r"\end{tabular}"]
    (args.out / "public_vs_private_summary.tex").write_text(
        "\n".join(tex2) + "\n")

    # Drop trajectories with < 2 entries (uninformative for trajectory plots)
    plot_lineages = [L for L in lineages if len(L["rows"]) >= 2]

    def _draw_dual_axis(ax, L, *, problem_pos="upper left"):
        rows_ = L["rows"]
        iters = [r.get("iteration") for r in rows_]
        pub = [r.get("public_score") for r in rows_]
        rank = [r.get("private_rank") for r in rows_]

        ax.plot(iters, pub, color="#1f77b4", marker="o", markersize=7,
                linewidth=2.4, label="public score")
        ax.set_xlabel("iteration of public-best update")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_ylabel("public score", color="#1f77b4")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        # private rank: invert so up = better
        finite = [r for r in rank if r is not None]
        if finite:
            ax2.plot(iters,
                     [-(r if r is not None else np.nan) for r in rank],
                     color="#d62728", marker="s", markersize=7, linewidth=2.4,
                     label="private rank (↑ = better)")
            ax2.set_ylabel(r"$-$ private rank  ($\uparrow$ = better)",
                           color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")
        else:
            ax2.set_yticks([])

        # Make spines visible on the right since we use the right axis
        ax.spines["right"].set_visible(True)
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)

        # problem label in the corner least likely to overlap data
        x0, y0 = (0.03, 0.96) if problem_pos == "upper left" else (0.97, 0.04)
        ha = "left" if problem_pos == "upper left" else "right"
        va = "top" if problem_pos == "upper left" else "bottom"
        ax.text(x0, y0, L["problem"], transform=ax.transAxes,
                fontsize=15, fontweight="bold",
                ha=ha, va=va,
                bbox={"facecolor": "white", "alpha": 0.9,
                      "edgecolor": "none", "pad": 2.5})
        return ax2

    # --- Fig 1a: hero plot — the worst-overfit problem ----------------
    overfit_candidates = [
        (r, L) for r, L in zip(rows_csv, lineages)
        if r["pub_delta_pct"] is not None
        and r["pub_delta_pct"] > 0
        and r["perf_delta"] is not None
        and r["perf_delta"] < 0
        and len(L["rows"]) >= 2
    ]
    overfit_candidates.sort(key=lambda rl: rl[0]["perf_delta"])  # most-negative first

    if overfit_candidates:
        worst_row, worst_L = overfit_candidates[0]
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        _draw_dual_axis(ax, worst_L, problem_pos="upper left")
        # annotate the headline numbers
        ax.text(0.97, 0.04,
                (f"public  {worst_row['pub_seed']:.1f} "
                 f"→ {worst_row['pub_final']:.1f}  "
                 f"({worst_row['pub_delta_pct']:+.2f}%)\n"
                 f"private rank  {int(worst_row['rank_seed'])} "
                 f"→ {int(worst_row['rank_final'])}  "
                 f"({worst_row['rank_delta']:+})\n"
                 f"private perf  {int(worst_row['perf_seed'])} "
                 f"→ {int(worst_row['perf_final'])}  "
                 f"({worst_row['perf_delta']:+})"),
                transform=ax.transAxes,
                fontsize=12, ha="right", va="bottom", family="serif",
                bbox={"facecolor": "white", "alpha": 0.92,
                      "edgecolor": "lightgray", "pad": 4})
        fig.tight_layout()
        fig.savefig(args.out / "public_vs_private_hero.pdf")
        plt.close(fig)

    # --- Fig 1b: full per-problem dual-axis trajectories grid ---------
    n = len(plot_lineages)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows),
                             squeeze=False)
    for ax, L in zip([a for row in axes for a in row], plot_lineages):
        _draw_dual_axis(ax, L, problem_pos="upper left")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "public_vs_private_grid.pdf")
    plt.close(fig)

    # --- Fig 2: Δpublic% vs Δprivate_performance scatter --------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs, ys, labels = [], [], []
    for r in rows_csv:
        if r["pub_delta_pct"] is None or r["perf_delta"] is None:
            continue
        if r["perf_seed"] is None or r["perf_final"] is None:
            continue
        xs.append(r["pub_delta_pct"])
        ys.append(r["perf_delta"])
        labels.append(r["problem"])

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.scatter(xs, ys, s=160, alpha=0.85, color="#3b5b9a",
               edgecolor="white", linewidth=1.0, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        # offset labels so they don't overlap markers
        ax.annotate(lab, (x, y), xytext=(7, 5),
                    textcoords="offset points", fontsize=14)
    ax.set_xlabel(r"$\Delta$ public score (%)")
    ax.set_ylabel(r"$\Delta$ private performance  (rating points)")

    # quadrant shading (subtle): top-right = both up, bottom-right = overfit
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.fill_between([0, xmax], 0, ymin, color="#d62728", alpha=0.06,
                    zorder=0, label="public ↑, private ↓ (overfit)")
    ax.fill_between([0, xmax], 0, ymax, color="#2ca02c", alpha=0.06,
                    zorder=0, label="both ↑ (aligned)")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "public_vs_private_scatter.pdf")
    plt.close(fig)

    # --- print headline ----------------------------------------------
    print()
    print("=== per-(framework × problem) summary "
          "(seed → final along public-best chain) ===")
    print(f"{'framework':<12} {'prob':<7} {'pub_Δ%':>8} "
          f"{'perf_Δ':>8} {'rank_Δ':>8}  status")
    for r in sorted(rows_csv, key=lambda x: (x["backend"], x["problem"])):
        cls = _classify(r)
        flag = ""
        if cls in ("overfit_severe", "overfit_mild"):
            flag = "  ← OVERFIT (public ↑, private ↓)"
        elif cls == "aligned":
            flag = "  aligned"
        pct = (f"{r['pub_delta_pct']:+.2f}"
               if r["pub_delta_pct"] is not None else "—")
        pd = (f"{r['perf_delta']:+}"
              if r["perf_delta"] is not None else "—")
        rd = (f"{r['rank_delta']:+}"
              if r["rank_delta"] is not None else "—")
        print(f"{r['backend']:<12} {r['problem']:<7} {pct:>8} "
              f"{pd:>8} {rd:>8}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
