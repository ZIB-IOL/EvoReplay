"""End-to-end BO over a single program's hparams.

Pipeline:
  1. Load a target program from a run.
  2. Single LLM call to propose tunable hparams + intervals (`llm_propose`).
  3. Rewrite the source: PARAMS = {...} + literal substitutions.
  4. Build a skopt Space; run gp_minimize against the benchmark evaluator.
  5. Persist trace + summary; report bo_best vs original_score.

Usage:
    uv run python -m evo_replay.agentic_tuning.run_bo \
        --run-dir <path> --program-id <prefix-or-best> \
        --evaluator <path> --calls 24 --initial-points 8 \
        --propose-model deepseek/deepseek-reasoner

By default the proposer model is whatever the run used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from evo_replay.agentic_tuning.llm_propose import (
    HparamSpec,
    propose_hparams,
    rewrite_program,
    render_with_params,
)
from evo_replay.core.checkpoints import load_programs


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------

def _resolve_program(run_dir: Path,
                     program_id: str | None,
                     at_iteration: int | None = None) -> Dict[str, Any]:
    """Pick a target program by id-prefix, by 'best at iter N', or run's best."""
    progs = load_programs(run_dir)

    if at_iteration is not None:
        # Best valid program with iteration_found <= at_iteration
        candidates = []
        for p in progs.values():
            it = int(p.get("iteration_found") or 0)
            if it > at_iteration:
                continue
            metrics = p.get("metrics") or {}
            if metrics.get("validity") == 0 or metrics.get("error"):
                continue
            sc = metrics.get("combined_score")
            if not isinstance(sc, (int, float)):
                continue
            candidates.append((sc, it, p))
        if not candidates:
            sys.exit(f"No valid program at iter <= {at_iteration} in {run_dir}")
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)  # highest score, latest iter
        return candidates[0][2]

    if program_id is None:
        # Prefer top-level best/. Fall back to latest checkpoint's
        # best_program_info.json (raw layout) or, for the refined layout
        # which has no per-checkpoint dirs, the highest-scoring valid program
        # in the loaded set.
        top = run_dir / "best/best_program_info.json"
        if top.exists():
            info = json.loads(top.read_text())
            target_id = info["id"]
        elif (run_dir / "checkpoints").is_dir():
            ckpts = sorted((run_dir / "checkpoints").glob("checkpoint_*"),
                           key=lambda p: int(p.name.split("_")[-1]))
            for ck in reversed(ckpts):
                p = ck / "best_program_info.json"
                if p.exists():
                    info = json.loads(p.read_text())
                    target_id = info["id"]
                    break
            else:
                sys.exit(f"No best_program_info.json found in {run_dir}")
        else:
            scored = [(p.get("metrics", {}).get("combined_score"), p)
                      for p in progs.values()
                      if isinstance(p.get("metrics", {}).get("combined_score"),
                                    (int, float))]
            if not scored:
                sys.exit(f"No scored programs found in {run_dir}")
            scored.sort(key=lambda c: c[0], reverse=True)
            target_id = scored[0][1]["id"]
    else:
        matches = [p for pid, p in progs.items() if pid.startswith(program_id)]
        if not matches:
            sys.exit(f"No program matches prefix {program_id!r}")
        if len(matches) > 1:
            sys.exit(f"Ambiguous prefix {program_id!r}, matches "
                     f"{[m['id'][:12] for m in matches]}")
        target = matches[0]
        target_id = target["id"]
    if target_id not in progs:
        # Fall back to reading the best dir directly if pruning evicted it.
        info = json.loads((run_dir / "best/best_program_info.json").read_text())
        if info["id"] != target_id:
            sys.exit(f"Target {target_id} not in active database.")
        prog = dict(info)
        for ext in ("py", "cpp"):
            best_src = run_dir / f"best/best_program.{ext}"
            if best_src.exists():
                prog["solution"] = best_src.read_text()
                break
        return prog
    return progs[target_id]


# ---------------------------------------------------------------------------
# Evaluator wrapping
# ---------------------------------------------------------------------------

def _import_evaluator(evaluator_path: Path):
    spec = importlib.util.spec_from_file_location("eval_module", evaluator_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_module"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "evaluate"):
        sys.exit(f"Evaluator {evaluator_path} has no evaluate(program_path) function.")
    return mod


def _evaluate(eval_mod, source: str, language: str = "python") -> Dict[str, Any]:
    """Write source to a temp file, call evaluator.evaluate(), return metrics dict."""
    import tempfile
    suffix = ".py" if language == "python" else (".cpp" if language == "cpp" else ".txt")
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(source)
        path = f.name
    try:
        return eval_mod.evaluate(path)
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass


def _score_of(metrics: Dict[str, Any]) -> float:
    for key in ("combined_score", "overall_score", "target_ratio", "score"):
        v = metrics.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


# ---------------------------------------------------------------------------
# BO
# ---------------------------------------------------------------------------

def _build_space(specs: List[HparamSpec]):
    from skopt.space import Integer, Real
    dims = []
    for s in specs:
        prior = "log-uniform" if s.scale == "log" else "uniform"
        if s.kind == "int":
            dims.append(Integer(int(s.low), int(s.high), prior=prior, name=s.name))
        else:
            dims.append(Real(float(s.low), float(s.high), prior=prior, name=s.name))
    return dims


def _coerce(specs: List[HparamSpec], xs: Sequence[Any]) -> Dict[str, Any]:
    out = {}
    for s, v in zip(specs, xs):
        out[s.name] = int(v) if s.kind == "int" else float(v)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--program-id", type=str, default=None,
                    help="program id prefix; default = run's best")
    ap.add_argument("--at-iteration", type=int, default=None,
                    help="pick the best valid program at iter <= N "
                         "(e.g. 30 = 'what BO finds in the first 30 iterations')")
    ap.add_argument("--evaluator", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=24)
    ap.add_argument("--initial-points", type=int, default=8)
    ap.add_argument("--propose-model", default="deepseek/deepseek-reasoner")
    ap.add_argument("--api-base",
                    default=os.environ.get("EVO_REPLAY_API_BASE"),
                    help="OpenAI-compatible base URL "
                         "(or set EVO_REPLAY_API_BASE)")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY for the LLM that proposes hparams.")
    if not args.api_base:
        sys.exit("Set --api-base or EVO_REPLAY_API_BASE.")

    run_dir = args.run_dir.resolve()
    out_dir = args.output or (run_dir / "analysis" / "bo_runs"
                              / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pick target
    target = _resolve_program(run_dir, args.program_id, args.at_iteration)
    source = target["solution"]
    original_score = _score_of(target.get("metrics") or {})
    target_id = target.get("id", "?")
    print(f"target: {target_id[:12]}  iter={target.get('iteration_found','?')}  "
          f"score={original_score:.6f}")

    language = target.get("language", "python")
    src_ext = ".cpp" if language == "cpp" else ".py"

    # 2. LLM-propose
    print(f"asking {args.propose_model} for hparam proposals...")
    t0 = time.time()
    specs = propose_hparams(source, language=language,
                            model=args.propose_model, api_base=args.api_base)
    print(f"  proposed in {time.time()-t0:.1f}s; {len(specs)} candidates:")
    for s in specs:
        print(f"    {s.name:<24} {s.source_literal:<10} -> "
              f"[{s.low:g}, {s.high:g}] ({s.scale}, {s.kind})  -- {s.rationale}")
    if not specs:
        sys.exit("No knobs proposed; nothing to BO.")

    # 3. Rewrite
    rewritten, applied = rewrite_program(source, specs, language=language)
    print(f"  applied {len(applied)} of {len(specs)} proposals after dedup/match check")
    (out_dir / f"rewritten{src_ext}").write_text(rewritten)
    (out_dir / "proposals.json").write_text(json.dumps(
        [asdict(s) for s in applied], indent=2))

    # 4. BO loop
    eval_mod = _import_evaluator(args.evaluator.resolve())
    space = _build_space(applied)
    trace: List[Dict[str, Any]] = []

    # Verify defaults reproduce original score
    defaults = {s.name: s.default for s in applied}
    baseline_src = render_with_params(rewritten, defaults, language=language)
    (out_dir / f"baseline{src_ext}").write_text(baseline_src)
    base_metrics = _evaluate(eval_mod, baseline_src, language=language)
    base_score = _score_of(base_metrics)
    print(f"  baseline (defaults from rewritten) score: {base_score:.6f}  "
          f"(original was {original_score:.6f})")

    from skopt import gp_minimize

    x0 = [s.default for s in applied]

    def objective(x: Sequence[Any]) -> float:
        idx = len(trace)
        params = _coerce(applied, x)
        try:
            src = render_with_params(rewritten, params, language=language)
        except Exception as exc:
            trace.append({"call": idx, "params": params, "score": 0.0,
                          "error": f"render: {exc}"})
            return 0.0
        try:
            m = _evaluate(eval_mod, src, language=language)
            sc = _score_of(m)
            trace.append({"call": idx, "params": params, "metrics": m, "score": sc})
            return -float(sc)
        except Exception as exc:
            trace.append({"call": idx, "params": params, "score": 0.0,
                          "error": f"eval: {exc}"})
            return 0.0

    print(f"\nrunning gp_minimize with {args.calls} calls "
          f"({args.initial_points} initial random)...")
    gp_minimize(objective, space,
                n_calls=max(args.calls, args.initial_points + 1),
                n_initial_points=args.initial_points,
                x0=x0, random_state=args.seed)

    # 5. Summary
    best = max(trace, key=lambda r: r["score"])
    regret = []
    running = float("-inf")
    for r in trace:
        running = max(running, r["score"])
        regret.append(running)
    summary = {
        "target": {"id": target_id, "iteration": target.get("iteration_found"),
                   "original_score": original_score, "baseline_score": base_score},
        "n_calls": len(trace),
        "best_score": best["score"],
        "best_params": best["params"],
        "regret_curve": regret,
        "bo_minus_original": best["score"] - original_score,
        "applied_specs": [asdict(s) for s in applied],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2, default=str))

    print(f"\n=== RESULT ===")
    print(f"original_score : {original_score:.6f}")
    print(f"baseline (rewritten defaults): {base_score:.6f}")
    print(f"BO best        : {best['score']:.6f}  (Δ = {best['score']-original_score:+.6f})")
    print(f"best params    : {best['params']}")
    print(f"output dir     : {out_dir}")


if __name__ == "__main__":
    main()
