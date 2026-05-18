from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evo_replay.static.hyperparameter_counts import (
    build_points,
    count_hyperparameter_candidates,
    is_valid_program,
)
from evo_replay.core.checkpoints import load_programs as load_latest_programs
from evo_replay._vendored.metrics import get_score

CANONICAL_ANALYSIS_DIRNAME = "best_so_far_lineages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the parent-chain lineages of programs that became best-so-far during a run."
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
        help="Allow invalid programs to become best-so-far candidates if their score says so.",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Do not plot all programs as gray background points.",
    )
    return parser.parse_args()


def program_sort_key(program: Mapping[str, Any]) -> tuple[int, str]:
    return (int(program.get("iteration_found") or 0), str(program.get("id") or ""))


def make_program_node(program: Mapping[str, Any]) -> dict[str, Any]:
    solution = str(program.get("solution") or "")
    language = str(program.get("language") or "python")
    hparam_count, counts_by_kind, counts_by_context = count_hyperparameter_candidates(
        solution,
        language,
    )
    metrics = program.get("metrics") or {}
    return {
        "program_id": str(program.get("id") or ""),
        "parent_id": program.get("parent_id"),
        "iteration": int(program.get("iteration_found") or 0),
        "score": get_score(metrics),
        "language": language,
        "hyperparameter_count": hparam_count,
        "counts_by_kind": counts_by_kind,
        "counts_by_context": counts_by_context,
        "valid": is_valid_program(program),
    }


def find_best_so_far_targets(
    programs_by_id: Mapping[str, Mapping[str, Any]],
    *,
    include_invalid: bool,
) -> list[dict[str, Any]]:
    best_score = float("-inf")
    targets: list[dict[str, Any]] = []
    for program in sorted(programs_by_id.values(), key=program_sort_key):
        if not include_invalid and not is_valid_program(program):
            continue
        score = get_score(program.get("metrics") or {})
        if score > best_score:
            node = make_program_node(program)
            node["best_so_far_rank"] = len(targets) + 1
            targets.append(node)
            best_score = score
    return targets


def build_lineage(
    programs_by_id: Mapping[str, Mapping[str, Any]],
    target_id: str,
) -> tuple[list[dict[str, Any]], str | None]:
    current_id: str | None = target_id
    seen: set[str] = set()
    lineage: list[dict[str, Any]] = []

    while current_id:
        if current_id in seen:
            raise ValueError(f"Cycle detected in lineage at program {current_id}")
        seen.add(current_id)

        program = programs_by_id.get(current_id)
        if program is None:
            return list(reversed(lineage)), current_id

        lineage.append(make_program_node(program))
        parent_id = program.get("parent_id")
        current_id = str(parent_id) if parent_id else None

    return list(reversed(lineage)), None


def build_lineage_payload(
    programs_by_id: Mapping[str, Mapping[str, Any]],
    *,
    include_invalid: bool,
) -> dict[str, Any]:
    points = build_points(programs_by_id, include_invalid=include_invalid)
    targets = find_best_so_far_targets(programs_by_id, include_invalid=include_invalid)
    lineages: list[dict[str, Any]] = []

    for target in targets:
        nodes, missing_parent_id = build_lineage(programs_by_id, target["program_id"])
        lineages.append(
            {
                "target": target,
                "missing_parent_id": missing_parent_id,
                "nodes": nodes,
            }
        )

    return {
        "include_invalid": include_invalid,
        "points": [point.__dict__ for point in points],
        "best_so_far_targets": targets,
        "lineages": lineages,
    }


def _plot_line(
    ax: Any,
    nodes: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    color: Any,
    alpha: float,
    linewidth: float,
    label: str | None = None,
) -> None:
    ax.plot(
        [node["iteration"] for node in nodes],
        [node[metric] for node in nodes],
        marker="o",
        markersize=3.5,
        linewidth=linewidth,
        alpha=alpha,
        color=color,
        label=label,
    )


def plot_lineages(payload: Mapping[str, Any], output_path: Path, *, background: bool) -> None:
    import matplotlib.pyplot as plt

    points = list(payload["points"])
    targets = list(payload["best_so_far_targets"])
    lineages = list(payload["lineages"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    score_ax, hparam_ax = axes

    if background and points:
        xs = [point["iteration"] for point in points]
        score_ax.scatter(
            xs,
            [point["score"] for point in points],
            s=18,
            alpha=0.25,
            color="gray",
            label="all programs",
        )
        hparam_ax.scatter(
            xs,
            [point["hyperparameter_count"] for point in points],
            s=18,
            alpha=0.25,
            color="gray",
            label="all programs",
        )

    cmap = plt.get_cmap("viridis")
    denom = max(len(lineages) - 1, 1)
    for index, lineage in enumerate(lineages):
        nodes = lineage["nodes"]
        if not nodes:
            continue
        color = cmap(index / denom)
        is_final = index == len(lineages) - 1
        label = "best-so-far lineages" if index == 0 else None
        _plot_line(
            score_ax,
            nodes,
            "score",
            color=color if not is_final else "black",
            alpha=0.28 if not is_final else 0.95,
            linewidth=1.1 if not is_final else 2.4,
            label=label,
        )
        _plot_line(
            hparam_ax,
            nodes,
            "hyperparameter_count",
            color=color if not is_final else "black",
            alpha=0.28 if not is_final else 0.95,
            linewidth=1.1 if not is_final else 2.4,
        )

    if targets:
        target_x = [target["iteration"] for target in targets]
        score_ax.step(
            target_x,
            [target["score"] for target in targets],
            where="post",
            color="crimson",
            linewidth=2.0,
            label="best-so-far frontier",
        )
        score_ax.scatter(
            target_x,
            [target["score"] for target in targets],
            marker="*",
            s=110,
            color="crimson",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
            label="frontier updates",
        )
        hparam_ax.step(
            target_x,
            [target["hyperparameter_count"] for target in targets],
            where="post",
            color="crimson",
            linewidth=2.0,
        )
        hparam_ax.scatter(
            target_x,
            [target["hyperparameter_count"] for target in targets],
            marker="*",
            s=110,
            color="crimson",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    score_ax.set_ylabel("score")
    hparam_ax.set_ylabel("static hparam candidates")
    hparam_ax.set_xlabel("iteration")
    score_ax.set_title("Best-so-far lineages: score and hyperparameter-candidate count")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_outputs(payload: Mapping[str, Any], analysis_dir: Path, *, background: bool) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "best_so_far_lineages.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    plot_lineages(payload, analysis_dir / "best_so_far_lineages.png", background=background)

    with (analysis_dir / "best_so_far_targets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "program_id",
                "parent_id",
                "iteration",
                "score",
                "hyperparameter_count",
                "valid",
            ],
        )
        writer.writeheader()
        for target in payload["best_so_far_targets"]:
            writer.writerow(
                {
                    "rank": target["best_so_far_rank"],
                    "program_id": target["program_id"],
                    "parent_id": target["parent_id"],
                    "iteration": target["iteration"],
                    "score": target["score"],
                    "hyperparameter_count": target["hyperparameter_count"],
                    "valid": target["valid"],
                }
            )

    with (analysis_dir / "best_so_far_lineage_nodes.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_rank",
                "target_program_id",
                "depth",
                "program_id",
                "parent_id",
                "iteration",
                "score",
                "hyperparameter_count",
                "valid",
            ],
        )
        writer.writeheader()
        for lineage in payload["lineages"]:
            target = lineage["target"]
            for depth, node in enumerate(lineage["nodes"]):
                writer.writerow(
                    {
                        "target_rank": target["best_so_far_rank"],
                        "target_program_id": target["program_id"],
                        "depth": depth,
                        "program_id": node["program_id"],
                        "parent_id": node["parent_id"],
                        "iteration": node["iteration"],
                        "score": node["score"],
                        "hyperparameter_count": node["hyperparameter_count"],
                        "valid": node["valid"],
                    }
                )


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
    programs_by_id = load_latest_programs(run_dir)
    payload = build_lineage_payload(programs_by_id, include_invalid=args.include_invalid)
    payload["run_dir"] = str(run_dir)
    write_outputs(payload, analysis_dir, background=not args.no_background)

    targets = payload["best_so_far_targets"]
    lineages = payload["lineages"]
    final = targets[-1] if targets else None
    print(f"programs={len(payload['points'])} best_so_far_updates={len(targets)}")
    if final:
        final_lineage = lineages[-1]["nodes"] if lineages else []
        print(
            "final_best: "
            f"id={final['program_id']} score={final['score']:.6g} "
            f"hparams={final['hyperparameter_count']} "
            f"iteration={final['iteration']} lineage_depth={len(final_lineage)}"
        )
    print(f"outputs={analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
