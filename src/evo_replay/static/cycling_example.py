"""Concrete cycling example for the paper: a single line that toggles in/out
of the best-of-iteration trajectory, with code context.

Renders a single PDF with two side-by-side panels:

  Left  — for the first ~12 iterations of one run, the score of the
          best program at that iter and whether the target line is in
          its source. Visualises the in/out cycle as a column of ticks.
  Right — a verbatim code excerpt showing the target line in its
          natural inner-loop context (from the highest-scoring iter
          where the line is present), with the cycled line highlighted.

The example is hard-coded for the canonical paper case
(`heilbronn_triangle_deepseek-deepseek-reasoner_100_868fe3`, line
`if improved:`) but parameters are CLI-overridable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from evo_replay.core.checkpoints import load_programs


DEFAULT_RUN = (
    "evox/heilbronn_triangle_deepseek-deepseek-reasoner_100_868fe3"
)
DEFAULT_LINE = "if improved:"
DEFAULT_N_ITERS = 12


def best_per_iter(progs):
    by_iter = defaultdict(list)
    for p in progs.values():
        s = (p.get("metrics") or {}).get("combined_score")
        if not isinstance(s, (int, float)):
            continue
        by_iter[int(p.get("iteration_found") or 0)].append(p)
    out = []
    for it in sorted(by_iter):
        rows = sorted(by_iter[it],
                      key=lambda p: -((p.get("metrics") or {}).get("combined_score") or -1e18))
        out.append((it, rows[0]))
    return out


def find_code_excerpt(prog, target: str, before=6, after=4):
    sol = prog.get("solution") or ""
    lines = sol.splitlines()
    idx = next((i for i, l in enumerate(lines) if l.strip() == target), None)
    if idx is None:
        idx = next((i for i, l in enumerate(lines) if target in l), None)
    if idx is None:
        return idx, []
    start = max(0, idx - before)
    end = min(len(lines), idx + after + 1)
    return idx, [(j + 1, lines[j]) for j in range(start, end)]


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply
    apply()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--line", default=DEFAULT_LINE)
    ap.add_argument("--n-iters", type=int, default=DEFAULT_N_ITERS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    run_dir = args.data_root / args.run
    progs = load_programs(run_dir)
    bpi = best_per_iter(progs)[:args.n_iters]

    # find the highest-scoring program (within shown window) that contains the line
    code_prog = None
    code_score = -1e18
    for it, p in bpi:
        if args.line in (p.get("solution") or ""):
            s = (p.get("metrics") or {}).get("combined_score") or -1e18
            if s > code_score:
                code_prog = p
                code_score = s
                code_iter = it
    idx, excerpt = (None, [])
    if code_prog:
        idx, excerpt = find_code_excerpt(code_prog, args.line)

    # ---- figure ----
    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.18)
    ax_tab = fig.add_subplot(gs[0, 0])
    ax_code = fig.add_subplot(gs[0, 1])
    for ax in (ax_tab, ax_code):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ---- left: presence table ----
    n = len(bpi)
    score_max = max((p.get("metrics") or {}).get("combined_score") or 0
                    for _, p in bpi)
    head_y, foot_y = 0.95, 0.10
    row_y = lambda i: head_y - (i + 1) * (head_y - foot_y) / (n + 0.7)

    # header
    ax_tab.text(0.04, head_y, "iter",
                fontsize=14, fontweight="bold", va="bottom")
    ax_tab.text(0.22, head_y, "score",
                fontsize=14, fontweight="bold", va="bottom")
    ax_tab.text(0.70, head_y, "presence",
                fontsize=14, fontweight="bold", va="bottom")
    ax_tab.plot([0.02, 0.98], [head_y - 0.005] * 2,
                color="black", linewidth=0.7)

    for i, (it, p) in enumerate(bpi):
        y = row_y(i)
        s = (p.get("metrics") or {}).get("combined_score") or 0
        has = args.line in (p.get("solution") or "")
        # iter
        ax_tab.text(0.04, y, str(it), fontsize=13, va="center")
        # mini-bar + score number
        ax_tab.add_patch(mpatches.Rectangle(
            (0.22, y - 0.012), 0.28 * (s / score_max), 0.018,
            color="#aac0d8", linewidth=0))
        ax_tab.text(0.51, y, f"{s:.3f}", fontsize=12, va="center",
                    family="monospace")
        # presence: filled green box if present, red outline if absent
        x0, w, h = 0.70, 0.04, 0.025
        if has:
            ax_tab.add_patch(mpatches.Rectangle(
                (x0, y - h / 2), w, h,
                facecolor="#2ca02c", edgecolor="#1a6322", linewidth=0.6))
            ax_tab.text(x0 + w + 0.02, y, "present",
                        fontsize=12, va="center", color="#1a6322")
        else:
            ax_tab.add_patch(mpatches.Rectangle(
                (x0, y - h / 2), w, h,
                facecolor="white", edgecolor="#a32218", linewidth=1.2))
            ax_tab.text(x0 + w + 0.02, y, "absent",
                        fontsize=12, va="center", color="#a32218")

    n_toggles = sum(1 for i in range(1, len(bpi))
                    if (args.line in (bpi[i][1].get("solution") or ""))
                    != (args.line in (bpi[i - 1][1].get("solution") or "")))
    ax_tab.text(0.02, foot_y - 0.05,
                f"{n_toggles} toggles in iters 0–{bpi[-1][0]}",
                fontsize=12, style="italic", va="top")

    # ---- right: code excerpt ----
    if excerpt:
        ax_code.text(0.0, 0.95,
                     f"`{args.line}` in iter {code_iter} "
                     f"(score {code_score:.3f})",
                     fontsize=14, fontweight="bold", va="bottom",
                     family="monospace", color="#a32218")
        ax_code.plot([0.0, 1.0], [0.94] * 2,
                     color="black", linewidth=0.7)
        n_lines = len(excerpt)
        top, bot = 0.85, 0.18
        line_h = (top - bot) / max(n_lines, 1)
        for i, (lineno, text) in enumerate(excerpt):
            y = top - (i + 0.5) * line_h
            is_target = text.strip() == args.line
            if is_target:
                ax_code.add_patch(mpatches.Rectangle(
                    (0.005, y - line_h * 0.42), 0.99, line_h * 0.85,
                    facecolor="#fff4f1", edgecolor="#a32218",
                    linewidth=0.8))
            color = "#a32218" if is_target else "black"
            weight = "bold" if is_target else "normal"
            ax_code.text(0.03, y, f"{lineno:>4}", fontsize=11,
                         color="#888", family="monospace", va="center")
            ax_code.text(0.10, y, text, fontsize=11,
                         family="monospace", color=color,
                         fontweight=weight, va="center")
        ax_code.text(
            0.0, foot_y - 0.05,
            (f"In a single run, the search independently re-derives "
             f"this control-flow line {n_toggles} times across "
             f"its first {bpi[-1][0]} iterations."),
            fontsize=12, style="italic", va="top")

    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
