from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from evo_replay.core.checkpoints import CANONICAL_ANALYSIS_DIRNAME, load_programs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot solution length development over the search process for a run directory."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing checkpoints/")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <run_dir>/analysis/{CANONICAL_ANALYSIS_DIRNAME}",
    )
    parser.add_argument(
        "--metric",
        choices=["chars", "lines"],
        default="chars",
        help="Length metric to plot. Default: chars",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a symmetric log scale on the x-axis.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a symmetric log scale on the y-axis.",
    )
    parser.add_argument(
        "--min-x-value",
        type=float,
        default=0.0,
        help="Clamp plotted x-values below this floor to the floor. Default: 0.",
    )
    parser.add_argument(
        "--min-y-value",
        type=float,
        default=0.0,
        help="Clamp plotted y-values below this floor to the floor. Default: 0.",
    )
    return parser.parse_args()


def _plot_suffix(log_x: bool, log_y: bool) -> str:
    parts: list[str] = []
    if log_x:
        parts.append("logx")
    if log_y:
        parts.append("logy")
    return "" if not parts else "_" + "_".join(parts)


def _apply_axis_scales(ax: Any, *, log_x: bool, log_y: bool) -> None:
    if log_x:
        ax.set_xscale("symlog", linthresh=1.0)
    if log_y:
        ax.set_yscale("symlog", linthresh=1.0)


def _clamp(values: list[float], floor: float) -> list[float]:
    return [max(v, floor) for v in values]


def load_program_points(run_dir: Path) -> list[dict[str, Any]]:
    programs = load_programs(run_dir)
    points: list[dict[str, Any]] = []
    for program in programs.values():
        solution = program.get("solution")
        if not solution:
            continue
        metrics = program.get("metrics") or {}
        score = metrics.get("combined_score")
        if score is None:
            continue
        points.append(
            {
                "iteration": int(program.get("iteration_found") or 0),
                "program_id": program["id"],
                "score": float(score),
                "chars": len(solution),
                "lines": len(solution.splitlines()),
            }
        )
    points.sort(key=lambda item: (item["iteration"], item["program_id"]))

    iter0 = [point for point in points if point["iteration"] == 0]
    nonzero = [point for point in points if point["iteration"] != 0]
    if iter0:
        seed = max(iter0, key=lambda item: item["score"])
        points = [seed, *nonzero]
    else:
        points = nonzero
    return points


def build_best_so_far_length_curve(
    points: list[dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    best_score = float("-inf")
    best_length = 0
    curve: list[dict[str, Any]] = []
    for point in points:
        if point["score"] > best_score:
            best_score = point["score"]
            best_length = int(point[metric])
        curve.append(
            {
                "iteration": point["iteration"],
                "best_so_far_score": best_score,
                "best_so_far_length": best_length,
            }
        )
    return curve


def build_iteration_summary(points: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    by_iteration: dict[int, list[dict[str, Any]]] = {}
    for point in points:
        by_iteration.setdefault(point["iteration"], []).append(point)

    summary: list[dict[str, Any]] = []
    for iteration in sorted(by_iteration):
        rows = by_iteration[iteration]
        lengths = [int(row[metric]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        summary.append(
            {
                "iteration": iteration,
                "program_count": len(rows),
                "min_length": min(lengths),
                "median_length": float(statistics.median(lengths)),
                "max_length": max(lengths),
                "best_score_in_iteration": max(scores),
            }
        )
    return summary


def plot_lengths(
    run_dir: Path,
    analysis_dir: Path,
    *,
    metric: str,
    log_x: bool,
    log_y: bool,
    min_x_value: float,
    min_y_value: float,
) -> None:
    points = load_program_points(run_dir)
    best_curve = build_best_so_far_length_curve(points, metric)
    iteration_summary = build_iteration_summary(points, metric)

    point_x = _clamp([float(point["iteration"]) for point in points], min_x_value)
    point_y = _clamp([float(point[metric]) for point in points], min_y_value)
    best_x = _clamp([float(point["iteration"]) for point in best_curve], min_x_value)
    best_y = _clamp([float(point["best_so_far_length"]) for point in best_curve], min_y_value)
    summary_x = _clamp([float(row["iteration"]) for row in iteration_summary], min_x_value)
    summary_y = _clamp([float(row["median_length"]) for row in iteration_summary], min_y_value)

    plt.figure(figsize=(10, 5))
    plt.scatter(
        point_x,
        point_y,
        s=16,
        alpha=0.35,
        color="gray",
        label="All evaluated programs",
    )
    plt.plot(
        summary_x,
        summary_y,
        color="steelblue",
        linewidth=2.0,
        label="Median length per iteration",
    )
    plt.plot(
        best_x,
        best_y,
        color="crimson",
        linewidth=2.2,
        label="Best-so-far solution length",
    )
    _apply_axis_scales(plt.gca(), log_x=log_x, log_y=log_y)
    plt.xlabel("Iteration")
    plt.ylabel("Solution length (chars)" if metric == "chars" else "Solution length (lines)")
    plt.title(f"{run_dir.name}: solution length over iterations")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    suffix = _plot_suffix(log_x, log_y)
    stem = f"solution_length_{metric}_over_iterations"
    plt.savefig(analysis_dir / f"{stem}{suffix}.png", dpi=200)
    plt.close()

    (analysis_dir / f"{stem}{suffix}.json").write_text(
        json.dumps(
            {
                "metric": metric,
                "points": points,
                "best_so_far_length_curve": best_curve,
                "iteration_summary": iteration_summary,
            },
            indent=2,
        )
    )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    analysis_dir = (args.analysis_dir or (run_dir / "analysis" / CANONICAL_ANALYSIS_DIRNAME)).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    plot_lengths(
        run_dir,
        analysis_dir,
        metric=args.metric,
        log_x=args.log_x,
        log_y=args.log_y,
        min_x_value=args.min_x_value,
        min_y_value=args.min_y_value,
    )

    print(f"outputs={analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
