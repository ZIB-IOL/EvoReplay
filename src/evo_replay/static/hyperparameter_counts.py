from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evo_replay.core.extractors import extract_literal_candidates
from evo_replay._vendored.metrics import get_score
from evo_replay.core.checkpoints import load_programs as load_latest_programs


CANONICAL_ANALYSIS_DIRNAME = "hyperparameter_counts"


@dataclass
class HyperparameterCountPoint:
    program_id: str
    iteration: int
    score: float
    language: str
    hyperparameter_count: int
    counts_by_kind: dict[str, int]
    counts_by_context: dict[str, int]
    valid: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count static hyperparameter candidates across an evolution run and summarize "
            "how candidate counts change over iterations."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing checkpoints/")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <run_dir>/analysis/{CANONICAL_ANALYSIS_DIRNAME}",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include programs whose metrics indicate execution errors or failed validity.",
    )
    return parser.parse_args()


def is_valid_program(program: Mapping[str, Any]) -> bool:
    metrics = program.get("metrics") or {}
    if "error" in metrics:
        return False
    if metrics.get("timeout") is True:
        return False
    if metrics.get("runs_successfully", 1.0) == 0:
        return False
    if metrics.get("validity", 1.0) == 0:
        return False
    return True


def count_hyperparameter_candidates(
    solution: str, language: str
) -> tuple[int, dict[str, int], dict[str, int]]:
    candidates = extract_literal_candidates(solution, language)
    if not candidates:
        return 0, {}, {}
    counts_by_kind = Counter(candidate.kind for candidate in candidates)
    counts_by_context = Counter(candidate.context for candidate in candidates)
    return (
        len(candidates),
        dict(sorted(counts_by_kind.items())),
        dict(sorted(counts_by_context.items())),
    )


def build_points(
    programs_by_id: Mapping[str, Mapping[str, Any]],
    *,
    include_invalid: bool,
) -> list[HyperparameterCountPoint]:
    points: list[HyperparameterCountPoint] = []
    for program in programs_by_id.values():
        solution = program.get("solution")
        if not isinstance(solution, str) or not solution:
            continue

        valid = is_valid_program(program)
        if not valid and not include_invalid:
            continue

        language = str(program.get("language") or "python")
        count, counts_by_kind, counts_by_context = count_hyperparameter_candidates(
            solution,
            language,
        )
        points.append(
            HyperparameterCountPoint(
                program_id=str(program.get("id") or ""),
                iteration=int(program.get("iteration_found") or 0),
                score=get_score(program.get("metrics") or {}),
                language=language,
                hyperparameter_count=count,
                counts_by_kind=counts_by_kind,
                counts_by_context=counts_by_context,
                valid=valid,
            )
        )

    return sorted(points, key=lambda item: (item.iteration, item.program_id))


def _mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_iterations(points: Sequence[HyperparameterCountPoint]) -> list[dict[str, Any]]:
    by_iteration: dict[int, list[HyperparameterCountPoint]] = {}
    for point in points:
        by_iteration.setdefault(point.iteration, []).append(point)

    rows: list[dict[str, Any]] = []
    for iteration in sorted(by_iteration):
        group = by_iteration[iteration]
        counts = [point.hyperparameter_count for point in group]
        best = max(group, key=lambda point: (point.score, -point.hyperparameter_count))
        max_count = max(group, key=lambda point: (point.hyperparameter_count, point.score))
        rows.append(
            {
                "iteration": iteration,
                "program_count": len(group),
                "avg_hyperparameters": _mean(counts),
                "median_hyperparameters": float(statistics.median(counts)),
                "min_hyperparameters": min(counts),
                "max_hyperparameters": max(counts),
                "best_program_id": best.program_id,
                "best_score": best.score,
                "best_program_hyperparameters": best.hyperparameter_count,
                "max_hyperparameter_program_id": max_count.program_id,
                "max_hyperparameter_program_score": max_count.score,
            }
        )
    return rows


def build_best_so_far_series(points: Sequence[HyperparameterCountPoint]) -> list[dict[str, Any]]:
    best: HyperparameterCountPoint | None = None
    series: list[dict[str, Any]] = []
    for point in sorted(points, key=lambda item: (item.iteration, item.program_id)):
        if best is None or point.score > best.score:
            best = point
        series.append(
            {
                "iteration": point.iteration,
                "program_id": point.program_id,
                "score": point.score,
                "hyperparameter_count": point.hyperparameter_count,
                "best_so_far_program_id": best.program_id,
                "best_so_far_score": best.score,
                "best_so_far_hyperparameters": best.hyperparameter_count,
            }
        )
    return series


def summarize_run(points: Sequence[HyperparameterCountPoint]) -> dict[str, Any]:
    if not points:
        return {
            "program_count": 0,
            "iteration_count": 0,
            "avg_hyperparameters": 0.0,
            "median_hyperparameters": 0.0,
            "min_hyperparameters": 0,
            "max_hyperparameters": 0,
            "best_program": None,
            "max_hyperparameter_program": None,
        }

    counts = [point.hyperparameter_count for point in points]
    best = max(points, key=lambda point: (point.score, -point.hyperparameter_count))
    max_count = max(points, key=lambda point: (point.hyperparameter_count, point.score))
    return {
        "program_count": len(points),
        "iteration_count": len({point.iteration for point in points}),
        "avg_hyperparameters": _mean(counts),
        "median_hyperparameters": float(statistics.median(counts)),
        "min_hyperparameters": min(counts),
        "max_hyperparameters": max(counts),
        "best_program": {
            "program_id": best.program_id,
            "iteration": best.iteration,
            "score": best.score,
            "hyperparameter_count": best.hyperparameter_count,
        },
        "max_hyperparameter_program": {
            "program_id": max_count.program_id,
            "iteration": max_count.iteration,
            "score": max_count.score,
            "hyperparameter_count": max_count.hyperparameter_count,
        },
    }


def analyze_run(
    run_dir: Path,
    *,
    include_invalid: bool = False,
) -> dict[str, Any]:
    programs_by_id = load_latest_programs(run_dir)
    points = build_points(programs_by_id, include_invalid=include_invalid)
    return {
        "run_dir": str(run_dir),
        "include_invalid": include_invalid,
        "summary": summarize_run(points),
        "points": [asdict(point) for point in points],
        "iteration_summary": summarize_iterations(points),
        "best_so_far_series": build_best_so_far_series(points),
    }


def write_outputs(payload: Mapping[str, Any], analysis_dir: Path) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "hyperparameter_counts.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    with (analysis_dir / "hyperparameter_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "program_id",
                "iteration",
                "score",
                "language",
                "hyperparameter_count",
                "valid",
            ],
        )
        writer.writeheader()
        for point in payload["points"]:
            writer.writerow(
                {
                    "program_id": point["program_id"],
                    "iteration": point["iteration"],
                    "score": point["score"],
                    "language": point["language"],
                    "hyperparameter_count": point["hyperparameter_count"],
                    "valid": point["valid"],
                }
            )

    with (analysis_dir / "hyperparameter_iteration_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "program_count",
                "avg_hyperparameters",
                "median_hyperparameters",
                "min_hyperparameters",
                "max_hyperparameters",
                "best_program_id",
                "best_score",
                "best_program_hyperparameters",
                "max_hyperparameter_program_id",
                "max_hyperparameter_program_score",
            ],
        )
        writer.writeheader()
        for row in payload["iteration_summary"]:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "programs.jsonl").exists() \
            and not (run_dir / "checkpoints").exists():
        raise SystemExit(
            f"No programs.jsonl or checkpoints/ directory found under {run_dir}")

    analysis_dir = (
        args.analysis_dir or (run_dir / "analysis" / CANONICAL_ANALYSIS_DIRNAME)
    ).resolve()
    payload = analyze_run(run_dir, include_invalid=args.include_invalid)
    write_outputs(payload, analysis_dir)

    summary = payload["summary"]
    best = summary["best_program"]
    max_program = summary["max_hyperparameter_program"]
    print(f"programs={summary['program_count']} iterations={summary['iteration_count']}")
    print(
        "hyperparameters: "
        f"avg={summary['avg_hyperparameters']:.2f} "
        f"median={summary['median_hyperparameters']:.2f} "
        f"min={summary['min_hyperparameters']} "
        f"max={summary['max_hyperparameters']}"
    )
    if best:
        print(
            "best_program: "
            f"id={best['program_id']} score={best['score']:.6g} "
            f"hparams={best['hyperparameter_count']} iteration={best['iteration']}"
        )
    if max_program:
        print(
            "max_hparam_program: "
            f"id={max_program['program_id']} score={max_program['score']:.6g} "
            f"hparams={max_program['hyperparameter_count']} "
            f"iteration={max_program['iteration']}"
        )
    print(f"outputs={analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
