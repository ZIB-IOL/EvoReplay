#!/usr/bin/env bash
# Classify parent->child edits on a few representative runs from evo_trace_anon
# using an OpenAI-compatible endpoint.
#
# Usage:
#   LLM_API_KEY=... ./scripts/classify_edits.sh                # gold + 3 runs
#   LLM_API_KEY=... ./scripts/classify_edits.sh --gold-only    # just gold
#   LLM_API_KEY=... ./scripts/classify_edits.sh --smoke        # 5 edits, one run
#   LLM_API_KEY=... ./scripts/classify_edits.sh /path/to/run   # one specific run
#
# Optional overrides:
#   EVO_REPLAY_API_BASE   OpenAI-compatible endpoint URL (defaults below)
#   EVO_TRACE_ROOT        path to the evo_trace_anon dataset (defaults to ../evo_trace_anon)

set -euo pipefail

: "${LLM_API_KEY:?Set LLM_API_KEY before running this script.}"

export EVO_REPLAY_API_BASE="${EVO_REPLAY_API_BASE:-https://your-endpoint/v1}"
export OPENAI_API_KEY="$LLM_API_KEY"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_ROOT="${EVO_TRACE_ROOT:-$REPO_ROOT/../evo_trace_anon}"
cd "$REPO_ROOT"

RUNS=(
  "$TRACE_ROOT/openevolve_native/heilbronn_triangle_deepseek-deepseek-reasoner_100_1dc568"
  "$TRACE_ROOT/openevolve_native/ale_bench_ahc027_deepseek-deepseek-reasoner_100_ca4fd5"
  "$TRACE_ROOT/evox/ale_bench_ahc046_deepseek-deepseek-reasoner_100_644ac1"
)

mode="${1:-full}"
case "$mode" in
  --gold-only)
    echo "==> Scoring judge against gold set"
    uv run python -m evo_replay.edit_taxonomy.gold score
    ;;
  --smoke)
    echo "==> Smoke: 5 edits from ${RUNS[0]##*/}"
    uv run python -m evo_replay.edit_taxonomy.run_classify "${RUNS[0]}" --max-edits 5
    ;;
  full)
    echo "==> Step 1/2: gold-set sanity check"
    uv run python -m evo_replay.edit_taxonomy.gold score
    echo
    echo "==> Step 2/2: classifying ${#RUNS[@]} runs"
    for run in "${RUNS[@]}"; do
      [ -d "$run" ] || { echo "MISSING: $run" >&2; continue; }
      echo "---- $(basename "$run") ----"
      uv run python -m evo_replay.edit_taxonomy.run_classify "$run"
    done
    ;;
  *)
    [ -d "$mode" ] || { echo "Not a directory: $mode" >&2; exit 1; }
    uv run python -m evo_replay.edit_taxonomy.run_classify "$mode"
    ;;
esac
