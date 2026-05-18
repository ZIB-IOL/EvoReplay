"""Paper-quality cycling figures + concrete example.

Reads pre-computed cycle_classes.json files and produces:

  C1_cycle_timeline.pdf    horizontal strip chart of presence/absence of
                            specific recycled lines over iterations, for a
                            single run picked as the most thrashy.
  C2_span_cdf.pdf          CDF of cycle span (= added_iter - removed_iter)
                            stratified by class (literal/tuning/trivial).
  C3_cycles_per_edit.pdf   distribution of cycles-per-edit by domain.
  TC1_top_recycled.tex     LaTeX table: top recycled lines + counts in the
                            example run.
  TC2_summary.tex          LaTeX table: cycling summary by (backend × domain).

All numbers are computed from the dataset's `analysis/cycle_classes.json`
plus `analysis/cycle_classes.summary.json`.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


BACKENDS = {"openevolve_native", "evox", "gepa_native", "shinkaevolve"}
DOMAIN_COLOR = {"ale": "#1f77b4", "math": "#d62728"}
KLASS_COLOR = {"literal": "#1f77b4", "tuning": "#d62728", "trivial": "#2ca02c"}


def domain_of(name: str) -> str:
    n = name.lower()
    if n.startswith("ale_bench_") or re.match(r"ahc\d", n):
        return "ale"
    return "math"


def backend_of(p: Path) -> str:
    for parent in p.parents:
        if parent.name in BACKENDS:
            return parent.name
    return "unknown"


def collect_events(root: Path):
    events = []
    summaries = []
    for f in root.rglob("analysis/cycle_classes.json"):
        run_dir = f.parent.parent
        bk = backend_of(f)
        dm = domain_of(run_dir.name)
        try:
            items = json.load(f.open())
        except (OSError, ValueError):
            continue
        for ev in items:
            ev["_run"] = run_dir.name
            ev["_backend"] = bk
            ev["_domain"] = dm
            events.append(ev)
        sf = run_dir / "analysis" / "cycle_classes.summary.json"
        if sf.exists():
            try:
                s = json.load(sf.open())
            except (OSError, ValueError):
                s = {}
            s["_run"] = run_dir.name
            s["_backend"] = bk
            s["_domain"] = dm
            summaries.append(s)
    return events, summaries


def pick_example_run(events: list[dict]) -> tuple[str, list[tuple[str, int]]]:
    """Pick the run with the highest 'most-cycled-line' count, return that
    run name and the top recycled lines in it (line -> count)."""
    by_run_line: dict[tuple[str, str], int] = Counter(
        (e["_run"], e["line"]) for e in events)
    if not by_run_line:
        return "", []
    # group by run, find the run whose top line has the highest count
    runs_max: dict[str, int] = {}
    runs_lines: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (run, line), n in by_run_line.items():
        runs_lines[run].append((line, n))
        if n > runs_max.get(run, 0):
            runs_max[run] = n
    best_run = max(runs_max, key=lambda r: runs_max[r])
    top_lines = sorted(runs_lines[best_run], key=lambda t: -t[1])[:8]
    return best_run, top_lines


def build_presence_timeline(events: list[dict], run: str, line: str,
                             max_iter: int) -> np.ndarray:
    """Approximate "is the line present at iteration t" timeline.

    A cycle event says: line was removed at iter X, re-added at iter Y.
    From a sequence of (X, Y) pairs we infer absence intervals [X, Y).
    Outside those intervals, the line is presumed present.
    """
    events_for_line = sorted(
        [(e["removed_iter"], e["added_iter"]) for e in events
         if e["_run"] == run and e["line"] == line
         and isinstance(e.get("removed_iter"), int)
         and isinstance(e.get("added_iter"), int)],
        key=lambda t: t[0],
    )
    present = np.ones(max_iter + 1, dtype=int)
    for x, y in events_for_line:
        if x <= max_iter and y <= max_iter + 1:
            present[x:y] = 0
    return present


def fig_timeline(events, max_iter, out: Path):
    """Tick-mark rug plot: one tick per cycle event for each line.

    For each of the top recycled lines in the example run, plot a tick at
    every cycle event's `added_iter`. Each tick = one parent→child edit
    that re-introduced the line after some earlier edit had removed it.
    The density of ticks shows recycling intensity over the run.
    """
    run, top = pick_example_run(events)
    if not top:
        return None, []

    LINES_TO_SHOW = 6
    rows = top[:LINES_TO_SHOW]

    fig, ax = plt.subplots(figsize=(11.0, 0.55 * len(rows) + 1.6))
    for idx, (line, n) in enumerate(rows):
        # collect all (added_iter) for cycle events on this line in this run
        added = sorted([e["added_iter"] for e in events
                        if e["_run"] == run and e["line"] == line
                        and isinstance(e.get("added_iter"), int)])
        # background lane (subtle)
        ax.barh(idx, max_iter, left=0, height=0.55,
                color="#eef2f6", edgecolor="none")
        # tick marks for each event
        if added:
            ax.vlines(added, idx - 0.30, idx + 0.30,
                      colors="#3b5b9a", linewidth=0.9, alpha=0.55)
        # row label (right): the line text + count of events
        label = (line if len(line) <= 60 else line[:57] + "...")
        ax.text(max_iter + 2, idx, f"  {label}",
                va="center", fontsize=11, family="monospace")
        ax.text(max_iter + 2, idx + 0.32, f"  {n} re-add events",
                va="center", fontsize=9, alpha=0.7,
                family="serif", style="italic")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"L{i+1}" for i in range(len(rows))], fontsize=12)
    ax.set_xlim(-1, int(max_iter * 1.95))
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.invert_yaxis()
    ax.set_xlabel("iteration")
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return run, rows


def fig_span_cdf(events, out: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for klass, color in KLASS_COLOR.items():
        spans = sorted(e["span"] for e in events
                       if e["klass"] == klass
                       and isinstance(e.get("span"), int))
        if not spans:
            continue
        xs = spans
        ys = np.linspace(1.0 / len(spans), 1.0, len(spans))
        ax.step(xs, ys, where="post", color=color, linewidth=2.4,
                label=f"{klass} (n={len(spans):,}, "
                      f"median={int(np.median(spans))})")
    ax.set_xlabel("cycle span  (iterations between removal and re-addition)")
    ax.set_ylabel("cumulative fraction of events")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_cycles_per_edit(summaries, out: Path):
    by_dom = defaultdict(list)
    for s in summaries:
        edits = s.get("n_edits", 0)
        events = s.get("n_recycle_events", 0)
        if edits and edits > 0:
            by_dom[s["_domain"]].append(events / edits)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for d in ("ale", "math"):
        vals = by_dom.get(d, [])
        if not vals:
            continue
        ax.hist(vals, bins=30, alpha=0.55, color=DOMAIN_COLOR[d],
                label=f"{d} (n={len(vals)} runs, "
                      f"median={np.median(vals):.1f})")
    ax.set_xlabel("cycle events per program edit")
    ax.set_ylabel("# runs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_top_recycled_table(run: str, rows, out: Path):
    """LaTeX table: top recycled lines from the chosen example run."""
    def esc(s): return (s.replace("\\", r"\textbackslash{}")
                          .replace("&", r"\&")
                          .replace("%", r"\%")
                          .replace("_", r"\_")
                          .replace("#", r"\#")
                          .replace("$", r"\$")
                          .replace("{", r"\{")
                          .replace("}", r"\}"))
    lines = [
        r"% Top-cycled lines in a single representative run.",
        r"% Each row is one source-code line; the count is the number of",
        r"% (removal, re-addition) cycle events for that exact line within",
        r"% the run's 100 iterations.",
        f"% run: {run}",
        r"\begin{tabular}{rl}",
        r"\toprule",
        r"\# cycles & line \\",
        r"\midrule",
    ]
    for line, n in rows:
        truncated = line if len(line) <= 80 else line[:77] + "..."
        lines.append(f"{n} & \\verb|{truncated}| \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def write_summary_table(summaries, out: Path):
    """Per-(backend × domain) cycling summary."""
    grouped = defaultdict(list)
    for s in summaries:
        grouped[(s["_backend"], s["_domain"])].append(s)

    backends = sorted({s["_backend"] for s in summaries})
    domains = ["ale", "math"]

    lines = [
        r"% Cycling summary per (backend, domain). Median across runs.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"backend & domain & runs & median cycles & median cycles/edit & median span \\",
        r"\midrule",
    ]
    for b in backends:
        for d in domains:
            grp = grouped.get((b, d), [])
            if not grp:
                continue
            n_evs = [s.get("n_recycle_events", 0) for s in grp]
            ratios = [s.get("n_recycle_events", 0) / max(s.get("n_edits", 1), 1)
                      for s in grp]
            spans = [s.get("span_stats", {}).get("median", 0) or 0
                     for s in grp]
            label = b.replace("_native", "").replace("shinkaevolve", "shinka")
            lines.append(
                f"{label} & {d} & {len(grp)} & "
                f"{int(np.median(n_evs))} & "
                f"{np.median(ratios):.2f} & "
                f"{np.median(spans):.1f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply
    apply()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path,
                    help="Refined dataset root (e.g. evo_trace_anon)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    events, summaries = collect_events(args.root)
    print(f"loaded {len(events):,} cycle events across "
          f"{len({e['_run'] for e in events})} runs")

    # max iter — needed for the timeline. Use the max removed/added we see.
    max_iter = max(
        max(e.get("removed_iter") or 0 for e in events),
        max(e.get("added_iter") or 0 for e in events),
    )
    print(f"max iter observed: {max_iter}")

    run, rows = fig_timeline(events, max_iter,
                             args.out / "C1_cycle_timeline.pdf")
    print(f"  C1 timeline: example run = {run}")

    fig_span_cdf(events, args.out / "C2_span_cdf.pdf")
    fig_cycles_per_edit(summaries, args.out / "C3_cycles_per_edit.pdf")

    write_top_recycled_table(run, rows, args.out / "TC1_top_recycled.tex")
    write_summary_table(summaries, args.out / "TC2_summary.tex")

    print(f"wrote outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
