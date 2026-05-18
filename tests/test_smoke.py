"""Smoketest: import all modules, then run static + cycling end-to-end on a real run dir."""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# Set EVO_REPLAY_TEST_RUN_DIR to an existing run directory (refined or raw
# layout) to enable the end-to-end smoketests. Skipped if unset.
_TEST_RUN = os.environ.get("EVO_REPLAY_TEST_RUN_DIR", "")
SOURCE_RUN = Path(_TEST_RUN) if _TEST_RUN else None


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    if SOURCE_RUN is None or not SOURCE_RUN.exists():
        pytest.skip("set EVO_REPLAY_TEST_RUN_DIR to a run directory to enable")
    dst = tmp_path_factory.mktemp("evo_replay_smoke") / "run"
    shutil.copytree(SOURCE_RUN, dst)
    return dst


@pytest.mark.parametrize(
    "name",
    [
        "evo_replay",
        "evo_replay.core.checkpoints",
        "evo_replay.core.extractors",
        "evo_replay._vendored.metrics",
        "evo_replay.static.run_static",
        "evo_replay.static.solution_length",
        "evo_replay.static.hyperparameter_counts",
        "evo_replay.static.lineage",
        "evo_replay.cycling.detect_cycling",
        "evo_replay.cycling.classify_edits",
        "evo_replay.cycling.classify_cycles",
        "evo_replay.cycling.classify_plots",
        "evo_replay.cycling.plots",
        "evo_replay.cycling.timeseries",
        "evo_replay.agentic_tuning.llm_propose",
        "evo_replay.agentic_tuning.run_bo",
        "evo_replay.agentic_tuning.aggregate_bo",
        "evo_replay.edit_taxonomy",
        "evo_replay.edit_taxonomy.rubric",
        "evo_replay.edit_taxonomy.judge",
        "evo_replay.edit_taxonomy.run_classify",
        "evo_replay.edit_taxonomy.gold",
    ],
)
def test_imports(name: str) -> None:
    importlib.import_module(name)


def test_breakthrough_replay_imports() -> None:
    pytest.importorskip("skydiscover")
    importlib.import_module("evo_replay.breakthrough_replay.replay")
    importlib.import_module("evo_replay.breakthrough_replay.counterfactuals")
    importlib.import_module("evo_replay.breakthrough_replay.run_replay")
    importlib.import_module("evo_replay.breakthrough_replay.private_eval_lineage")
    importlib.import_module("evo_replay.breakthrough_replay.sample_targets")


def _run_module(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
        check=True,
    )


def test_static_run_end_to_end(run_dir: Path) -> None:
    _run_module(["evo_replay.static.run_static", str(run_dir)])
    analysis = run_dir / "analysis"
    assert (analysis / "hyperparameter_counts" / "hyperparameter_counts.json").stat().st_size > 0
    assert (analysis / "hyperparameter_counts" / "hyperparameter_points.csv").stat().st_size > 0
    assert (analysis / "best_so_far_lineages" / "best_so_far_lineages.json").stat().st_size > 0
    assert (analysis / "best_so_far_lineages" / "best_so_far_targets.csv").stat().st_size > 0


def test_cycling_detect(run_dir: Path) -> None:
    csv_out = run_dir / "analysis" / "cycles_raw.csv"
    _run_module(
        ["evo_replay.cycling.detect_cycling", str(run_dir), "--csv", str(csv_out)]
    )
    assert csv_out.stat().st_size > 0
    assert csv_out.with_suffix(".programs.csv").stat().st_size > 0


def test_cycling_classify_edits(run_dir: Path) -> None:
    _run_module(["evo_replay.cycling.classify_edits", str(run_dir)])
    analysis = run_dir / "analysis"
    assert (analysis / "edit_classification.csv").stat().st_size > 0
    assert (analysis / "edit_classification.json").stat().st_size > 0


def test_bo_rewrite_program() -> None:
    """Round-trip rewrite_program on a tiny synthetic source. No LLM, no gp_minimize."""
    from evo_replay.agentic_tuning.llm_propose import HparamSpec, rewrite_program

    source = (
        "def step(x):\n"
        "    lr = 0.05\n"
        "    return x - lr * x\n"
    )
    specs = [
        HparamSpec(
            name="lr",
            source_literal="0.05",
            context_line="    lr = 0.05",
            default=0.05,
            low=1e-3,
            high=1.0,
            scale="log",
            kind="float",
            rationale="learning rate",
        )
    ]
    rewritten, used = rewrite_program(source, specs)
    assert "PARAMS" in rewritten
    assert "lr" in rewritten
    assert len(used) == 1
