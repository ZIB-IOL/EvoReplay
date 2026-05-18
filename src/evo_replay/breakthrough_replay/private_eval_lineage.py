#!/usr/bin/env python3
"""
private_eval_lineage.py — Evaluate the best-so-far lineage on the private test set.

Extracts each program that set a new best during evolution, runs private
evaluation on each (in parallel), and reports public vs private scores
side by side to detect overfitting.

Usage:
    uv run scripts/private_eval_lineage.py <run_dir> --problem-id ahc027
    uv run scripts/private_eval_lineage.py <run_dir> --problem-id ahc027 --workers 4
    uv run scripts/private_eval_lineage.py <run_dir> --problem-id ahc027 --runs 1  # faster, less precise
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LineageEntry:
    iteration: int
    program_id: str
    public_score: float
    model_name: str
    checkpoint: str
    program_path: str  # path to extracted .cpp file
    # filled after private eval
    private_score: float | None = None
    private_rank: float | None = None
    private_performance: float | None = None
    private_passed: int | None = None
    private_failed: int | None = None
    private_error: str | None = None


def extract_lineage(run_dir: Path) -> list[LineageEntry]:
    """Find every checkpoint where the best program changed."""
    cp_root = run_dir / "checkpoints"
    if not cp_root.exists():
        sys.exit(f"No checkpoints directory found under {run_dir}")

    cp_dirs = sorted(
        [p for p in cp_root.iterdir() if p.is_dir()],
        key=lambda p: int(re.search(r"\d+", p.name).group()),
    )

    entries: list[LineageEntry] = []
    prev_id: str | None = None

    for cp in cp_dirs:
        info_file = cp / "best_program_info.json"
        code_file = None
        for ext in ("cpp", "py", "c", "rs"):
            candidate = cp / f"best_program.{ext}"
            if candidate.exists():
                code_file = candidate
                break

        if not info_file.exists() or code_file is None:
            continue

        info = json.loads(info_file.read_text())
        pid = info["id"]
        if pid == prev_id:
            continue
        prev_id = pid

        # Look up model_name from programs dir
        model_name = "unknown"
        prog_json = cp / "programs" / f"{pid}.json"
        if prog_json.exists():
            prog_data = json.loads(prog_json.read_text())
            model_name = prog_data.get("metadata", {}).get("model_name", "unknown")

        public_score = info["metrics"].get("combined_score", float("nan"))

        entries.append(
            LineageEntry(
                iteration=info.get("iteration", 0),
                program_id=pid,
                public_score=public_score,
                model_name=model_name,
                checkpoint=cp.name,
                program_path=str(code_file),
            )
        )

    return entries


def _run_single_private_eval(
    program_path: str, problem_id: str, num_runs: int
) -> dict[str, Any]:
    """Run private evaluation in a subprocess-safe function."""
    # Ensure project root is on sys.path for worker processes
    import os
    import subprocess
    import sys
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # cairocffi needs libcairo on macOS — DYLD_LIBRARY_PATH gets stripped by SIP
    # in child processes, so preload it via ctypes before cairocffi tries
    if sys.platform == "darwin":
        import ctypes.util
        if not ctypes.util.find_library("cairo"):
            try:
                cairo_prefix = subprocess.run(
                    ["brew", "--prefix", "cairo"], capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                if cairo_prefix:
                    dylib = f"{cairo_prefix}/lib/libcairo.2.dylib"
                    if os.path.exists(dylib):
                        ctypes.cdll.LoadLibrary(dylib)
                        os.environ["DYLD_LIBRARY_PATH"] = f"{cairo_prefix}/lib"
            except Exception:
                pass

    from benchmarks.ale_bench.ale_session_helper import (
        start_ale_bench_robust,
        strip_evolve_markers,
    )

    results: list[dict[str, Any]] = []

    for run_idx in range(num_runs):
        try:
            session = start_ale_bench_robust(
                problem_id=problem_id,
                lite_version=False,
                num_workers=13,
            )
            code = strip_evolve_markers(Path(program_path).read_text())
            private_result, final_rank, final_performance = session.private_eval(
                code, code_language="cpp20",
            )

            private_json = json.loads(private_result.model_dump_json())
            passed = sum(
                1 for c in private_json["case_results"]
                if c["judge_result"] == "ACCEPTED"
            )
            failed = len(private_json["case_results"]) - passed

            results.append({
                "private_score": private_result.overall_absolute_score,
                "private_rank": final_rank,
                "private_performance": final_performance,
                "private_passed": passed,
                "private_failed": failed,
            })
        except Exception as e:
            results.append({
                "private_score": None,
                "private_rank": None,
                "private_performance": None,
                "private_passed": None,
                "private_failed": None,
                "error": f"{e}",
            })

    # Average across runs
    valid = [r for r in results if r["private_score"] is not None]
    if not valid:
        return {
            "error": "; ".join(r.get("error", "unknown") for r in results),
        }

    return {
        "private_score": sum(r["private_score"] for r in valid) / len(valid),
        "private_rank": sum(r["private_rank"] for r in valid) / len(valid),
        "private_performance": sum(r["private_performance"] for r in valid) / len(valid),
        "private_passed": round(sum(r["private_passed"] for r in valid) / len(valid)),
        "private_failed": round(sum(r["private_failed"] for r in valid) / len(valid)),
        "num_successful_runs": len(valid),
    }


def run_private_evals(
    entries: list[LineageEntry],
    problem_id: str,
    *,
    max_workers: int = 2,
    num_runs: int = 3,
) -> list[LineageEntry]:
    """Run private evaluations in parallel."""
    # Skip the seed if it has -inf score
    eval_entries = [e for e in entries if e.public_score > -1e15]

    if not eval_entries:
        print("No valid entries to evaluate.")
        return entries

    print(f"Running private evaluation on {len(eval_entries)} programs "
          f"({num_runs} run(s) each, {max_workers} workers)")
    print()

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for entry in eval_entries:
            future = executor.submit(
                _run_single_private_eval,
                entry.program_path,
                problem_id,
                num_runs,
            )
            futures[future] = entry

        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
                if "error" in result:
                    entry.private_error = result["error"]
                    print(f"  iter {entry.iteration:>3}: FAILED - {result['error'][:80]}")
                else:
                    entry.private_score = result["private_score"]
                    entry.private_rank = result["private_rank"]
                    entry.private_performance = result["private_performance"]
                    entry.private_passed = result["private_passed"]
                    entry.private_failed = result["private_failed"]
                    print(f"  iter {entry.iteration:>3}: private_score={entry.private_score:.2f}  "
                          f"rank={entry.private_rank:.1f}  perf={entry.private_performance:.4f}")
            except Exception as e:
                entry.private_error = str(e)
                print(f"  iter {entry.iteration:>3}: ERROR - {e}")

    return entries


def print_results(entries: list[LineageEntry], run_dir: Path) -> None:
    """Print comparison table."""
    print()
    print("=" * 110)
    print(f"  Private vs Public Score Comparison — {run_dir.name}")
    print("=" * 110)
    print()
    print(f"{'iter':>4}  {'model':<28s}  {'public_score':>14s}  {'private_score':>14s}  "
          f"{'rank':>6s}  {'perf':>8s}  {'pass/fail':>10s}")
    print("-" * 110)

    valid_public = []
    valid_private = []

    for e in entries:
        if e.public_score < -1e15:
            pub_str = f"{'seed':>14s}"
        else:
            pub_str = f"{e.public_score:>14.2f}"

        if e.private_score is not None:
            priv_str = f"{e.private_score:>14.2f}"
            rank_str = f"{e.private_rank:>6.1f}"
            perf_str = f"{e.private_performance:>8.4f}"
            pf_str = f"{e.private_passed}/{e.private_failed}"
            valid_public.append(e.public_score)
            valid_private.append(e.private_score)
        elif e.private_error:
            priv_str = "FAILED"
            rank_str = ""
            perf_str = ""
            pf_str = ""
        else:
            priv_str = "—"
            rank_str = ""
            perf_str = ""
            pf_str = ""

        print(f"{e.iteration:>4}  {e.model_name:<28s}  {pub_str:>14s}  {priv_str:>14s}  "
              f"{rank_str:>6s}  {perf_str:>8s}  {pf_str:>10s}")

    print("-" * 110)

    # Overfitting analysis
    if len(valid_public) >= 2:
        pub_improvement = valid_public[-1] - valid_public[0]
        priv_improvement = valid_private[-1] - valid_private[0]
        pub_pct = pub_improvement / abs(valid_public[0]) * 100 if valid_public[0] else 0
        priv_pct = priv_improvement / abs(valid_private[0]) * 100 if valid_private[0] else 0

        print()
        print(f"  Public improvement:  {valid_public[0]:.2f} -> {valid_public[-1]:.2f}  ({pub_pct:+.2f}%)")
        print(f"  Private improvement: {valid_private[0]:.2f} -> {valid_private[-1]:.2f}  ({priv_pct:+.2f}%)")
        if abs(pub_pct) > 0:
            ratio = priv_pct / pub_pct
            print(f"  Private/Public improvement ratio: {ratio:.2f}")
            if ratio < 0.5:
                print("  WARNING: Strong overfitting signal — private improvement much less than public")
            elif ratio < 0.8:
                print("  CAUTION: Moderate overfitting — private gains lag behind public")
            else:
                print("  OK: Private improvements track public improvements")

    print("=" * 110)


def _setup_cairo_macos() -> None:
    """Pre-load libcairo on macOS so forked workers inherit it."""
    import ctypes.util
    import os
    import subprocess

    if sys.platform != "darwin":
        return
    if ctypes.util.find_library("cairo"):
        return
    try:
        import ctypes

        cairo_prefix = subprocess.run(
            ["brew", "--prefix", "cairo"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if cairo_prefix:
            dylib = f"{cairo_prefix}/lib/libcairo.2.dylib"
            if os.path.exists(dylib):
                ctypes.cdll.LoadLibrary(dylib)
                os.environ["DYLD_LIBRARY_PATH"] = f"{cairo_prefix}/lib"
    except Exception:
        pass


def main() -> None:
    _setup_cairo_macos()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path, help="Run output directory")
    parser.add_argument("--problem-id", required=True, help="ALE-Bench problem ID (e.g. ahc027)")
    parser.add_argument("--workers", type=int, default=2, help="Max parallel evaluations (default: 2)")
    parser.add_argument("--runs", type=int, default=3, help="Private eval runs per program (default: 3)")
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    entries = extract_lineage(run_dir)

    if not entries:
        sys.exit("No best-program lineage found in checkpoints")

    print(f"Found {len(entries)} programs in best-so-far lineage:")
    for e in entries:
        pub = f"{e.public_score:.2f}" if e.public_score > -1e15 else "seed"
        print(f"  iter {e.iteration:>3}: {e.model_name:<28s}  public={pub}")
    print()

    entries = run_private_evals(
        entries, args.problem_id, max_workers=args.workers, num_runs=args.runs,
    )

    print_results(entries, run_dir)

    # Save results
    output_path = args.output or (run_dir / "private_eval_lineage.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        {
            "run_dir": str(run_dir),
            "problem_id": args.problem_id,
            "num_runs": args.runs,
            "lineage": [asdict(e) for e in entries],
        },
        indent=2,
    ))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
