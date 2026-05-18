"""Plot recycle-ratio over iteration from a cycling CSV.

Reads <out>.programs.csv produced by `detect_cycling --csv <out>`. Produces
two overlays: raw and structural-only (when both CSVs are passed).

Usage:
    uv run python -m evo_replay.cycling.plots \
        --raw cycles_raw.programs.csv \
        --structural cycles_structural.programs.csv \
        --out cycling.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def _load(path: Path) -> tuple[list[int], list[float]]:
    iters: list[int] = []
    ratios: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iters.append(int(row["iteration"]))
                ratios.append(float(row["recycle_ratio"]))
            except (KeyError, ValueError):
                continue
    pairs = sorted(zip(iters, ratios))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


def plot(raw: Path | None, structural: Path | None, out: Path,
         *, window: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for path, label, color in [
        (raw, "raw (incl. tuning)", "#cc6677"),
        (structural, "structural only", "#1f77b4"),
    ]:
        if path is None:
            continue
        x, y = _load(path)
        if not x:
            continue
        ax.scatter(x, y, alpha=0.25, color=color, s=10, label=None)
        smooth = _rolling_mean(y, window)
        ax.plot(x, smooth, color=color, linewidth=2.0,
                label=f"{label} (rolling-{window} mean)")
        plotted = True

    if not plotted:
        raise SystemExit("No usable rows in the supplied CSV(s).")

    ax.set_xlabel("iteration")
    ax.set_ylabel("recycle ratio (recycled added / total added)")
    ax.set_ylim(0, 1)
    ax.set_title("Line-level cycling over time")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None,
                    help="Path to <prefix>.programs.csv from detect_cycling without flags")
    ap.add_argument("--structural", type=Path, default=None,
                    help="Path to <prefix>.programs.csv from detect_cycling "
                         "--collapse-numbers --exclude-hyperparams")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()
    if args.raw is None and args.structural is None:
        ap.error("Provide at least one of --raw or --structural")
    plot(args.raw, args.structural, args.out, window=args.window)


if __name__ == "__main__":
    main()
