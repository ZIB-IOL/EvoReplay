"""Plot edit composition (tuning / structural / comment) over iterations.

Reads `edit_classification.csv` produced by `classify_edits`.

Usage:
    uv run python -m evo_replay.cycling.classify_plots \
        --csv <run_dir>/analysis/edit_classification.csv \
        --out <run_dir>/analysis/edit_composition.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def _load(path: Path) -> List[Tuple[int, int, int, int]]:
    """Return list of (iteration, tuning_lines, structural_lines, comment_lines)."""
    rows: List[Tuple[int, int, int, int]] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                rows.append((
                    int(r["iteration"]),
                    int(r["tuning_lines"]),
                    int(r["structural_lines"]),
                    int(r["comment_lines"]),
                ))
            except (KeyError, ValueError):
                continue
    rows.sort()
    return rows


def plot(csv_path: Path, out_path: Path) -> None:
    rows = _load(csv_path)
    if not rows:
        raise SystemExit(f"No usable rows in {csv_path}")

    iters = [r[0] for r in rows]
    tuning = [r[1] for r in rows]
    structural = [r[2] for r in rows]
    comments = [r[3] for r in rows]

    fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    width = 0.8
    ax_abs.bar(iters, tuning, width=width, color="#cc6677",
               label="tuning (paired skeletons)")
    ax_abs.bar(iters, structural, width=width, bottom=tuning,
               color="#1f77b4", label="structural")
    bottoms = [t + s for t, s in zip(tuning, structural)]
    ax_abs.bar(iters, comments, width=width, bottom=bottoms,
               color="#999999", label="comment")
    ax_abs.set_ylabel("lines changed")
    ax_abs.set_title("Edit composition over iterations")
    ax_abs.legend(loc="upper right")
    ax_abs.grid(alpha=0.3)

    # Normalised stack: tuning fraction of code (excludes comments).
    code = [t + s for t, s in zip(tuning, structural)]
    tuning_frac = [t / c if c else 0.0 for t, c in zip(tuning, code)]
    structural_frac = [1.0 - f for f in tuning_frac]
    ax_rel.bar(iters, tuning_frac, width=width, color="#cc6677")
    ax_rel.bar(iters, structural_frac, width=width, bottom=tuning_frac,
               color="#1f77b4")
    ax_rel.set_ylabel("share of code-changing lines")
    ax_rel.set_xlabel("iteration")
    ax_rel.set_ylim(0, 1)
    ax_rel.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="edit_classification.csv from classify_edits")
    ap.add_argument("--out", type=Path, required=True,
                    help="output PNG path")
    args = ap.parse_args()
    plot(args.csv, args.out)


if __name__ == "__main__":
    main()
