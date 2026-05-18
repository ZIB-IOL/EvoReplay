"""Run all three static analyses on a single run directory.

Produces under <run_dir>/analysis/:
  - solution_length/
  - hyperparameter_counts/
  - best_so_far_lineages.{json,png,...}

Usage:
    uv run python -m evo_replay.static.run_static <run_dir> [--language cpp|python]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evo_replay.core.checkpoints import detect_language


def _run(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--language", default=None,
                    help="Source language. Auto-detected from run_config.yaml if omitted.")
    ap.add_argument("--include-invalid", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    lang = args.language or detect_language(run_dir) or "python"
    print(f"Language: {lang}")

    common = [str(run_dir)]
    if args.include_invalid:
        common.append("--include-invalid")

    rc = 0
    rc |= _run("evo_replay.static.solution_length", [str(run_dir)])
    rc |= _run("evo_replay.static.hyperparameter_counts", common)
    rc |= _run("evo_replay.static.lineage", common)
    sys.exit(rc)


if __name__ == "__main__":
    main()
