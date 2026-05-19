<h1 align="center">
  <b>EvoReplay</b><br>
  <sub>What Do Evolutionary Coding Agents Evolve?</sub>
</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge"></a>
  <!-- TODO: replace XXXX.XXXXX with the real arXiv ID once posted -->
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arxiv&style=for-the-badge"></a>
  <a href="https://huggingface.co/datasets/ZIB-IOL/EvoTrace"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EvoTrace-ffd21e?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="https://www.pelleriti.org/">Nico Pelleriti</a> ·
  <a href="https://nelaturuharsha.github.io/">Sree Harsha Nelaturu</a> ·
  <a href="https://andrewzhou924.github.io/">Zhanke Zhou</a> ·
  Zongze Li ·
  <a href="https://maxzimmer.org/">Max Zimmer</a> ·
  <a href="https://bhanml.github.io/">Bo Han</a> ·
  <a href="https://pokutta.com/">Sebastian Pokutta</a>
</p>

---

## Overview

**EvoReplay** is the post-run analysis suite that accompanies our paper
*What Do Evolutionary Coding Agents Evolve?* It takes the raw traces produced
by an evolutionary code-search run — the population of candidate programs,
their parent links, prompts, scores, and per-iteration metrics — and turns
them into the static measurements, cycling detections, counterfactual
replays, and LLM-judged edit-taxonomy labels we report in the paper.

The companion dataset of traces lives on the Hugging Face Hub:

<p align="center">
  <a href="https://huggingface.co/datasets/ZIB-IOL/EvoTrace">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20EvoTrace-ZIB--IOL%2FEvoTrace-ffd21e?style=for-the-badge">
  </a>
</p>

## Key Features

* 📐 **Static analysis** — lines of code, hyperparameter counts, lineage depth, and best-so-far trajectories per run
* 🔁 **Cycling detection** — line-level recycling of removed code, with structural-only and tuning-only modes
* 🏷️ **Edit taxonomy** — LLM-as-judge labelling of every parent → child diff into nine categories, with a hand-labelled gold set and inter-rater agreement tooling
* 🎯 **Agentic Bayesian-optimisation tuning** — an LLM proposes tunable knobs + intervals on a frozen program, `scikit-optimize` searches over them
* 🎬 **Breakthrough replay** — re-run the prompts that caused best-so-far updates under different models / context strategies
* 🐍 **Python + C++ literal extractors** so the same pipeline works on both supported trace languages

---

## Supported Run Layouts

`evo_replay` operates on a `<run_dir>/` and auto-detects which of two layouts
it is reading. Both produce the same analyses.

**Refined** (preferred — produced by `scripts/refine_outputs.py`):

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

**Raw** (legacy search-framework output):

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

---

## Setup

```bash
# Clone
git clone https://github.com/ZIB-IOL/EvoReplay.git
cd EvoReplay

# Install (uses uv: https://docs.astral.sh/uv/)
uv sync
```

The `breakthrough_replay/` module additionally depends on the underlying
evolutionary-search framework that produced the traces. Install it from your
local checkout:

```bash
uv pip install -e /path/to/search-framework
```

### LLM endpoint

`agentic_tuning/` and `breakthrough_replay/` need an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
export EVO_REPLAY_API_BASE=https://your-endpoint/v1
```

Both also accept `--api-base` on the command line.

### Pointing at the dataset

The example scripts and gold-set builder expect a local checkout of the
companion dataset:

```bash
# Either fetch with the HF CLI:
huggingface-cli download ZIB-IOL/EvoTrace --repo-type dataset --local-dir ./evo_trace_anon

# ...or with git-lfs:
git clone https://huggingface.co/datasets/ZIB-IOL/EvoTrace evo_trace_anon

# Then tell EvoReplay where it lives (or pass --trace-root):
export EVO_TRACE_ROOT="$(pwd)/evo_trace_anon"
```

---

## Usage

### 1. Static analysis (LOC, hyperparameter counts, lineage)

```bash
uv run python -m evo_replay.static.run_static <run_dir>
```

Auto-detects language (Python / C++) from `run_config.yaml`, falling back to
the first `programs.jsonl` row, then `best/best_program.{cpp,py}`. Outputs land
under `<run_dir>/analysis/`.

### 2. Cycling detection

```bash
# Raw cycling
uv run python -m evo_replay.cycling.detect_cycling <run_dir> \
    --csv <run_dir>/analysis/cycles_raw.csv

# Structural-only (strips numeric-tuning churn)
uv run python -m evo_replay.cycling.detect_cycling <run_dir> \
    --collapse-numbers --exclude-hyperparams \
    --csv <run_dir>/analysis/cycles_structural.csv

# Per-edit composition (pure-tuning vs structural)
uv run python -m evo_replay.cycling.classify_edits <run_dir>
```

### 3. Edit-taxonomy classification

LLM-judge every parent → child diff into the nine taxonomy categories. The
on-disk cache is content-addressed, so re-runs and shared parents across runs
do not re-pay:

```bash
# Score the judge against the hand-labelled gold set
uv run python -m evo_replay.edit_taxonomy.gold score

# Classify every edit in a run
uv run python -m evo_replay.edit_taxonomy.run_classify <run_dir>
```

The shipped wrapper `scripts/classify_edits.sh` runs the gold check followed
by three representative runs from the dataset.

### 4. Agentic Bayesian-optimisation tuning

```bash
uv run python -m evo_replay.agentic_tuning.run_bo \
    --run-dir <run_dir> --program-id best \
    --evaluator <path_to_evaluator.py> \
    --calls 24 --initial-points 8 \
    --propose-model deepseek/deepseek-reasoner \
    --api-base "$EVO_REPLAY_API_BASE"
```

Pipeline: load a program from a run, ask an LLM to propose tunable hparams +
intervals, rewrite the source as `PARAMS = {...}` + literal substitutions,
then run `skopt.gp_minimize` against the evaluator.

Aggregate ceilings across an experiment dir:

```bash
uv run python -m evo_replay.agentic_tuning.aggregate_bo <experiment_dir>
```

### 5. Breakthrough replay (needs the search framework)

```bash
uv run python -m evo_replay.breakthrough_replay.run_replay <run_dir> \
    --top-events 3 \
    --models "model-a,model-b" \
    --prompts "exact,strict_diff,no_history,no_other_context" \
    --repeats 1 --attempts 3
```

---

## Module Layout

| Folder | Purpose |
| --- | --- |
| `core/` | Shared utilities: program loading, lineage walks, literal extractors |
| `static/` | LOC, hyperparameter counts, best-program lineage depth, paper figures |
| `cycling/` | Line-level cycling detection (raw + structural-only) + plots |
| `edit_taxonomy/` | Nine-category LLM-judge classifier + gold set + agreement tooling |
| `agentic_tuning/` | LLM-proposed Bayesian-optimisation tuning of hyperparameters |
| `breakthrough_replay/` | Replay best-so-far events under different models / prompts |

---

## Tests

```bash
uv sync --extra dev
uv run pytest -v
```

End-to-end smoketests are gated on `EVO_REPLAY_TEST_RUN_DIR` (they run the
static + cycling pipelines against a real run directory) — the rest of the
suite covers the rubric, judge, gold set, agreement, and BO rewriter without
network or filesystem fixtures:

```bash
EVO_REPLAY_TEST_RUN_DIR=/path/to/some/run_dir uv run pytest -v
```

---

## Dataset

The companion dataset of evolutionary code-search traces is published on the
Hugging Face Hub as [**ZIB-IOL/EvoTrace**](https://huggingface.co/datasets/ZIB-IOL/EvoTrace).
It contains the runs analysed in the paper across multiple search backends,
benchmark domains, and model configurations. See the dataset card for layout
and licence details.

---

## Citation

If you use EvoReplay or the EvoTrace dataset in your research, please cite:

```bibtex
@article{pelleriti2026evoreplay,
  title  = {What Do Evolutionary Coding Agents Evolve?},
  author = {Pelleriti, Nico and Nelaturu, Sree Harsha and Zhou, Zhanke
            and Li, Zongze and Zimmer, Max and Han, Bo and Pokutta, Sebastian},
  year   = {2026},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
}
```

<!-- TODO: replace eprint with the real arXiv ID and update venue/journal once known. -->

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
