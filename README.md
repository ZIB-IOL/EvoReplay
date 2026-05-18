# `evo_replay`

Post-run analysis suite for evolutionary code-search traces. Companion code
for the paper.

## Supported run layouts

`evo_replay` operates on a `<run_dir>/` and auto-detects which of two layouts
it is reading. Both produce the same analyses.

**Refined** (preferred — produced by `skydiscover/scripts/refine_outputs.py`):

```
<run_dir>/
    meta.json
    run_config.yaml             (canonical 3 backends only)
    programs.jsonl              one row per unique program; canonical fields
                                (incl. solution_sha256, prompts_sha256)
    iterations.jsonl
    iter_scalars.jsonl
    blobs/<sha[:2]>/<sha>.{txt,json}    content-addressed code & prompts
    best/, logs/, analysis/     (canonical 3 backends; symlinks or copies)
```

**Raw** (legacy skydiscover output):

```
<run_dir>/
    run_config.yaml
    run_info.json
    checkpoints/checkpoint_<N>/programs/<uuid>.json
    best/
    logs/
```

`core.checkpoints.load_programs(run_dir)` returns the same
`{pid: program_record}` dict from either layout. For the refined layout it
dereferences the content-addressed blobs and re-injects them as
`program["solution"]` / `program["prompts"]`, so downstream code does not
need to know which layout it is reading.

## Layout

| Folder                  | Purpose                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `core/`                 | Shared utilities: program loading, lineage walks, extractors |
| `static/`               | LOC, hyperparameter counts, best-program lineage depth       |
| `cycling/`              | Line-level cycling detection (raw + structural-only) + plots |
| `agentic_tuning/`       | LLM-proposed Bayesian-optimisation tuning of hyperparameters |
| `breakthrough_replay/`  | Replay best-so-far events under different models / prompts (requires `skydiscover`) |

## Install

```
uv sync
```

`breakthrough_replay/` additionally needs `skydiscover` installed in the same
environment. Install it from your local checkout, e.g.:

```
uv pip install -e /path/to/skydiscover
```

## LLM endpoint configuration

`agentic_tuning/` and `breakthrough_replay/` need access to an OpenAI-compatible
endpoint:

```
export OPENAI_API_KEY=...
export EVO_REPLAY_API_BASE=https://your-endpoint/v1
```

Both also accept `--api-base` on the command line.

## Usage

### Static analysis (LOC, hyperparameter counts, lineage)

```
uv run python -m evo_replay.static.run_static <run_dir>
```

Auto-detects language (Python / C++) from `run_config.yaml`, falling back to
the first `programs.jsonl` row, then `best/best_program.{cpp,py}`. Outputs
land under `<run_dir>/analysis/`.

### Cycling detection

```
# Raw cycling
uv run python -m evo_replay.cycling.detect_cycling <run_dir> \
    --csv <run_dir>/analysis/cycles_raw.csv

# Structural-only (strips numeric-tuning churn)
uv run python -m evo_replay.cycling.detect_cycling <run_dir> \
    --collapse-numbers --exclude-hyperparams \
    --csv <run_dir>/analysis/cycles_structural.csv

# Per-edit composition
uv run python -m evo_replay.cycling.classify_edits <run_dir>
```

### Bayesian-optimisation tuning

```
uv run python -m evo_replay.agentic_tuning.run_bo \
    --run-dir <run_dir> --program-id best \
    --evaluator <path_to_evaluator.py> \
    --calls 24 --initial-points 8 \
    --propose-model deepseek/deepseek-reasoner \
    --api-base "$EVO_REPLAY_API_BASE"
```

Pipeline: load a program from a run, ask an LLM (OpenAI-compatible endpoint)
to propose tunable hparams + intervals, rewrite the source as `PARAMS = {...}`
+ literal substitutions, then run `skopt.gp_minimize` against the evaluator.

Aggregate ceilings across an experiment dir:

```
uv run python -m evo_replay.agentic_tuning.aggregate_bo <experiment_dir>
```

### Breakthrough replay (needs `skydiscover`)

```
uv run python -m evo_replay.breakthrough_replay.run_replay <run_dir> \
    --top-events 3 \
    --models "model-a,model-b" \
    --prompts "exact,strict_diff,no_history,no_other_context" \
    --repeats 1 --attempts 3
```

## Tests

```
uv sync --extra dev
uv run pytest -v
```

End-to-end smoketests are gated on `EVO_REPLAY_TEST_RUN_DIR`:

```
EVO_REPLAY_TEST_RUN_DIR=/path/to/some/run_dir uv run pytest -v
```
