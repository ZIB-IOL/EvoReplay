from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evo_replay.static._plot_solution_length import (
    build_best_so_far_length_curve,
    build_iteration_summary,
    load_program_points,
    plot_lengths,
)

from evo_replay.core.checkpoints import CANONICAL_ANALYSIS_DIRNAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze solution-length trends across one or more run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Run directories to analyze. If omitted, discover runs under outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/solution_length_trends"),
        help="Directory for aggregate JSON/CSV summaries.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also regenerate per-run solution-length plots for chars and lines.",
    )
    return parser.parse_args()


def discover_run_dirs() -> list[Path]:
    outputs = Path("outputs")
    candidates = sorted(p for p in outputs.glob("*/*") if p.is_dir())
    run_dirs: list[Path] = []
    for run_dir in candidates:
        if (run_dir / "programs.jsonl").exists() \
                or (run_dir / "checkpoints").exists():
            run_dirs.append(run_dir.resolve())
    return run_dirs


def is_monotone_non_decreasing(values: list[float]) -> bool:
    return all(values[i] >= values[i - 1] for i in range(1, len(values)))


def first_drop(values: list[float], xs: list[int]) -> dict[str, Any] | None:
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            return {
                "from_iteration": xs[i - 1],
                "to_iteration": xs[i],
                "from_value": values[i - 1],
                "to_value": values[i],
            }
    return None


def summarize_metric(points: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    best_curve = build_best_so_far_length_curve(points, metric)
    iteration_summary = build_iteration_summary(points, metric)

    point_values = [float(point[metric]) for point in points]
    point_iters = [int(point["iteration"]) for point in points]
    median_values = [float(row["median_length"]) for row in iteration_summary]
    median_iters = [int(row["iteration"]) for row in iteration_summary]
    best_values = [float(row["best_so_far_length"]) for row in best_curve]
    best_iters = [int(row["iteration"]) for row in best_curve]

    best_length_change_points: list[dict[str, Any]] = []
    prev_best_len: float | None = None
    for row in best_curve:
        if prev_best_len != row["best_so_far_length"]:
            best_length_change_points.append(
                {
                    "iteration": int(row["iteration"]),
                    "best_so_far_length": float(row["best_so_far_length"]),
                    "best_so_far_score": float(row["best_so_far_score"]),
                }
            )
            prev_best_len = row["best_so_far_length"]

    return {
        "seed_length": int(points[0][metric]) if points else 0,
        "final_best_so_far_length": int(best_curve[-1]["best_so_far_length"]) if best_curve else 0,
        "max_seen_length": int(max(point_values)) if point_values else 0,
        "final_iteration_median_length": float(iteration_summary[-1]["median_length"])
        if iteration_summary
        else 0.0,
        "all_points_monotone": is_monotone_non_decreasing(point_values),
        "median_per_iteration_monotone": is_monotone_non_decreasing(median_values),
        "best_so_far_length_monotone": is_monotone_non_decreasing(best_values),
        "all_points_first_drop": first_drop(point_values, point_iters),
        "median_per_iteration_first_drop": first_drop(median_values, median_iters),
        "best_so_far_length_first_drop": first_drop(best_values, best_iters),
        "best_length_change_points": best_length_change_points,
    }


def analyze_run(run_dir: Path, *, make_plots: bool) -> dict[str, Any]:
    points = load_program_points(run_dir)
    analysis_dir = (run_dir / "analysis" / CANONICAL_ANALYSIS_DIRNAME).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if make_plots:
        plot_lengths(
            run_dir,
            analysis_dir,
            metric="chars",
            log_x=False,
            log_y=False,
            min_x_value=0.0,
            min_y_value=0.0,
        )
        plot_lengths(
            run_dir,
            analysis_dir,
            metric="lines",
            log_x=False,
            log_y=False,
            min_x_value=0.0,
            min_y_value=0.0,
        )

    chars = summarize_metric(points, "chars")
    lines = summarize_metric(points, "lines")

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "program_count": len(points),
        "chars": chars,
        "lines": lines,
    }


def main() -> int:
    args = parse_args()
    run_dirs = [p.resolve() for p in args.run_dirs] if args.run_dirs else discover_run_dirs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if not (run_dir / "programs.jsonl").exists() \
                and not (run_dir / "checkpoints").exists():
            print(f"SKIP {run_dir} (no programs.jsonl or checkpoints/)")
            continue
        print(f"ANALYZE {run_dir}")
        summary = analyze_run(run_dir, make_plots=args.plot)
        summaries.append(summary)
        print(
            f"  lines: seed={summary['lines']['seed_length']} "
            f"best_final={summary['lines']['final_best_so_far_length']} "
            f"best_monotone={summary['lines']['best_so_far_length_monotone']} "
            f"median_monotone={summary['lines']['median_per_iteration_monotone']}"
        )

    (output_dir / "solution_length_trends.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "solution_length_trends.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "program_count",
                "chars_seed_length",
                "chars_final_best_so_far_length",
                "chars_max_seen_length",
                "chars_all_points_monotone",
                "chars_median_per_iteration_monotone",
                "chars_best_so_far_length_monotone",
                "lines_seed_length",
                "lines_final_best_so_far_length",
                "lines_max_seen_length",
                "lines_all_points_monotone",
                "lines_median_per_iteration_monotone",
                "lines_best_so_far_length_monotone",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "run_name": summary["run_name"],
                    "program_count": summary["program_count"],
                    "chars_seed_length": summary["chars"]["seed_length"],
                    "chars_final_best_so_far_length": summary["chars"][
                        "final_best_so_far_length"
                    ],
                    "chars_max_seen_length": summary["chars"]["max_seen_length"],
                    "chars_all_points_monotone": summary["chars"]["all_points_monotone"],
                    "chars_median_per_iteration_monotone": summary["chars"][
                        "median_per_iteration_monotone"
                    ],
                    "chars_best_so_far_length_monotone": summary["chars"][
                        "best_so_far_length_monotone"
                    ],
                    "lines_seed_length": summary["lines"]["seed_length"],
                    "lines_final_best_so_far_length": summary["lines"][
                        "final_best_so_far_length"
                    ],
                    "lines_max_seen_length": summary["lines"]["max_seen_length"],
                    "lines_all_points_monotone": summary["lines"]["all_points_monotone"],
                    "lines_median_per_iteration_monotone": summary["lines"][
                        "median_per_iteration_monotone"
                    ],
                    "lines_best_so_far_length_monotone": summary["lines"][
                        "best_so_far_length_monotone"
                    ],
                }
            )

    print(f"wrote={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
