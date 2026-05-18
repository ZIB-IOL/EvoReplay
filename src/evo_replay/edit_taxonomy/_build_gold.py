"""One-off helper to materialise gold/examples.jsonl from real traces.

Run once to (re)generate the gold-set file. The labels and notes below were
applied by hand after inspecting each diff in evo_trace_anon/.

    uv run python -m evo_replay.edit_taxonomy._build_gold

Outputs to: src/evo_replay/edit_taxonomy/gold/examples.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

from evo_replay.core.checkpoints import load_programs
from evo_replay.edit_taxonomy.judge import make_unified_diff


TRACE_ROOT = Path(os.environ.get("EVO_TRACE_ROOT", "../evo_trace_anon")).resolve()

# (run subpath, child id prefix, parent id prefix, labels, note)
ENTRIES: List[Tuple[str, str, str, List[str], str]] = [
    # Pure hyperparameter tuning — single literal change.
    (
        "openevolve_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_1dc568",
        "163e54c3", "4d6f8382",
        ["hyperparameter_tuning"],
        "single literal change: cooling_rate 0.99992 -> 0.99993",
    ),
    # Pure hyperparameter tuning — several literals, no structural change.
    (
        "openevolve_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_1dc568",
        "f7ed2d60", "31b6af2d",
        ["hyperparameter_tuning"],
        "num_restarts/steps_per_run/local_steps swept; surrounding code identical",
    ),
    (
        "openevolve_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_1dc568",
        "97f73af1", "063e4d94",
        ["hyperparameter_tuning"],
        "SA/adaptive-step config block retuned; structure identical",
    ),
    (
        "openevolve_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_1dc568",
        "d842d930", "d268191f",
        ["hyperparameter_tuning"],
        "two config blocks retuned; comments updated; no formula change",
    ),
    # Hyperparameter tuning + pruning (deletes a trailing 'final shake' phase).
    (
        "openevolve_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_1dc568",
        "8abb21a6", "a03448e8",
        ["hyperparameter_tuning", "pruning"],
        "retunes hyperparameters AND deletes the final global-shake phase",
    ),
    # Architectural change + pruning (replaces evolutionary algorithm with SA).
    (
        "gepa_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_683ca7",
        "d85e8b0f", "b40c8dc4",
        ["architectural_change", "pruning"],
        "replaces population-based evolutionary algorithm with multi-restart SA + hill-climbing",
    ),
    # Architectural change + external dep (full SA solver introduced; itertools.combinations import added).
    (
        "openevolve_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_1dc568",
        "4eb89dc6", "2514cded",
        ["architectural_change", "external_dependency"],
        "trivial seed -> full barycentric SA solver; adds `from itertools import combinations`",
    ),
    # Architectural change + external dep (full hull-aware SA on convex 13).
    (
        "openevolve_native/heilbronn_convex-13_deepseek-deepseek-reasoner_100_a85c1a",
        "5038b663", "f4510ff4",
        ["architectural_change", "external_dependency"],
        "trivial random init -> hull-aware double-hexagon SA; `import math` added",
    ),
    # Architectural change + hyperparameter tuning (initial-config redesign + SA budget swing).
    (
        "openevolve_native/heilbronn_convex-13_deepseek-deepseek-reasoner_100_a85c1a",
        "49051d4b", "7a0e41d5",
        ["architectural_change", "hyperparameter_tuning"],
        "perturbed-13gon + greedy-fine-tune phases replaced by deterministic-inward init; SA budget retuned",
    ),
    # Architectural change + pruning (hex-lattice solver replaced by 5x5 grid + delete helper).
    (
        "openevolve_native/circle_packing_deepseek-deepseek-reasoner_100_056bfb",
        "e6dba4bd", "0e8dc7ef",
        ["architectural_change", "pruning"],
        "hex-lattice + hill-climb replaced by 5x5 grid + central gap; large net deletion",
    ),
    # Pure architectural change (hex-lattice variant -> 5x5 + outward push).
    (
        "openevolve_native/circle_packing_deepseek-deepseek-reasoner_100_056bfb",
        "65ab1248", "48cd4423",
        ["architectural_change", "pruning"],
        "swept-config search replaced by 5x5 grid + deterministic outward push",
    ),
    # Local refinement: enhances compute_max_radii to iterate; replaces equal-radius
    # assignment with the iterated solver. Same algorithm shape, different formula.
    (
        "openevolve_native/circle_packing_deepseek-deepseek-reasoner_100_056bfb",
        "73ec3222", "c4863325",
        ["local_refinement"],
        "compute_max_radii now iterates with max_iter/tol; r=full(n,r) -> compute_max_radii(centers)",
    ),
    # Composition + hyperparameter_tuning: adds linear-cooling AND a no-improvement-still-counts
    # acceptance step on top of the existing SA, while also retuning constants.
    (
        "openevolve_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_1dc568",
        "c2a79ba4", "0b8aed41",
        ["composition", "hyperparameter_tuning"],
        "adds linear-cooling phase and step-adaptation acceptance on top of SA + retunes",
    ),
    # External dependency only: typing.List import + type hints + return-type fix; algorithm unchanged.
    # NOTE: this is a shinkaevolve degenerate case where the EVOLVE-BLOCK got reset to a Kadane
    # reference; we still use it as a clean external-dep example.
    (
        "shinkaevolve/heilbronn_triangle_dpsk-reasoner_100_373173",
        "9ab74840", "9379fe86",
        ["external_dependency"],
        "adds `from typing import List`, type annotations, docstring; algorithm unchanged",
    ),
    # Pruning: implementation deleted, replaced by a placeholder comment.
    (
        "shinkaevolve/heilbronn_triangle_dpsk-reasoner_100_373173",
        "98ae211f", "e3e1d25a",
        ["pruning"],
        "entire heilbronn implementation deleted, replaced by a placeholder comment",
    ),
    # Trivial: docstring-only edit, no code change.
    (
        "openevolve_native/circle_packing_deepseek-deepseek-reasoner_100_056bfb",
        "cce061a1", "f06e3174",
        [],
        "docstring expansion only; no code change",
    ),
    # Bug fix + hyperparameter tuning: parent silently ignored
    # apply_combined_strategy()'s success/failure return AND its INF_COST
    # sentinel, so failed segments were treated as successful. Child captures
    # the return into success_seg and breaks out of the greedy loop on
    # failure. Same diff retunes GREEDY_REOPTIMIZE_SUBSET_SIZE 40 -> 170.
    (
        "evox/ale_bench_ahc046_deepseek-deepseek-reasoner_100_644ac1",
        "d0ffc770", "923e55b4",
        ["bug_fix", "hyperparameter_tuning"],
        "parent ignored apply_combined_strategy()'s return + INF_COST; child checks both; also tunes SUBSET_SIZE",
    ),
    # Efficiency + hyperparameter tuning: replaces full std::sort with
    # std::nth_element to extract top-k without sorting the rest. Same outputs,
    # asymptotically faster. POST_MAX_ATTEMPTS 500 -> 200 in the same diff.
    (
        "evox/ale_bench_ahc027_deepseek-deepseek-reasoner_100_1dc55b",
        "951a3f7a", "7da927a1",
        ["efficiency", "hyperparameter_tuning"],
        "std::sort -> std::nth_element for top-k selection; POST_MAX_ATTEMPTS retuned",
    ),
    # Efficiency + local refinement + hyperparameter tuning: introduces a
    # delta-update perimeter cache (avoids recomputation = efficiency) and
    # adds a perimeter-limit pre-acceptance check (formula change to the SA
    # acceptance step = local_refinement). Two literals tweaked too.
    (
        "openevolve_native/ale_bench_ahc039_deepseek-deepseek-reasoner_100_13b29f",
        "300671b3", "3171596b",
        ["efficiency", "local_refinement", "hyperparameter_tuning"],
        "delta-update perimeter cache; perimeter-limit guard added to SA acceptance; two literals tuned",
    ),
]


def main() -> None:
    out_path = Path(__file__).parent / "gold" / "examples.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for run_sub, pid_prefix, par_prefix, labels, note in ENTRIES:
            run_dir = TRACE_ROOT / run_sub
            ps = load_programs(run_dir)
            pid = next((k for k in ps if k.startswith(pid_prefix)), None)
            if pid is None:
                print(f"SKIP missing child {pid_prefix} in {run_sub}", file=sys.stderr)
                continue
            par_id = ps[pid].get("parent_id")
            if not par_id or not par_id.startswith(par_prefix):
                print(f"SKIP parent mismatch for {pid_prefix}: expected {par_prefix}, got {par_id}", file=sys.stderr)
                continue
            par = ps[par_id]
            child = ps[pid]
            diff = make_unified_diff(par["solution"], child["solution"], n=3)
            rec = {
                "run": run_sub,
                "child_id": pid,
                "parent_id": par_id,
                "labels": labels,
                "note": note,
                "diff": diff,
            }
            f.write(json.dumps(rec) + "\n")
            print(f"  + {pid_prefix}  {labels}  ({len(diff)} chars)")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
