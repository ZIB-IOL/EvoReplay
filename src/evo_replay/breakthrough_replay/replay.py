"""Trace replay CLI for prompt-stability analysis.

Loads a completed run, resolves the ancestry trace for a selected node,
and replays saved prompts multiple times to measure response/code/score
stability under identical prompt content.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig, apply_overrides, build_output_dir, load_config
from skydiscover.evaluation import create_evaluator
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.refine import find_checkpoint_dir
from skydiscover.search.base_database import Program
from skydiscover.search.utils.checkpoint_manager import CheckpointManager
from skydiscover.utils.code_utils import apply_diff, extract_diffs, parse_full_rewrite
from skydiscover.utils.metrics import get_score

logger = logging.getLogger("skydiscover.replay")


def _load_checkpoint(path: str) -> Tuple[str, Dict[str, Program], Optional[str], int]:
    ckpt = find_checkpoint_dir(path)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found at {path}")
    mgr = CheckpointManager(DatabaseConfig())
    programs, best_id, last_iteration = mgr.load(ckpt)
    if not programs:
        raise ValueError(f"No programs found in checkpoint {ckpt}")
    return ckpt, programs, best_id, last_iteration


def _resolve_program(
    programs: Dict[str, Program], best_id: Optional[str], program_id: Optional[str]
) -> Program:
    if program_id is None:
        if best_id and best_id in programs:
            return programs[best_id]
        return max(programs.values(), key=lambda p: get_score(p.metrics or {}))

    if program_id in programs:
        return programs[program_id]

    matches = [program for pid, program in programs.items() if pid.startswith(program_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous program ID prefix '{program_id}', matches: {[m.id[:12] for m in matches]}"
        )
    raise KeyError(f"Program '{program_id}' not found")


def _build_lineage(programs: Dict[str, Program], target: Program) -> List[Program]:
    lineage: List[Program] = []
    current = target
    seen: set[str] = set()
    while current is not None:
        if current.id in seen:
            raise ValueError(f"Cycle detected while building lineage at program {current.id}")
        seen.add(current.id)
        lineage.append(current)
        if not current.parent_id:
            break
        if current.parent_id not in programs:
            # Population pruning may have evicted older ancestors from the active
            # database. Treat the current program as the lineage root rather than
            # failing — replay only needs the target's prompt + its parent's code,
            # which is fetched separately via program.parent_info if needed.
            break
        current = programs[current.parent_id]
    lineage.reverse()
    return lineage


def _candidate_prompt_entries(program: Program) -> List[Tuple[str, Dict[str, Any]]]:
    prompts = program.prompts or {}
    entries = []
    for key, value in prompts.items():
        if isinstance(value, dict) and "system" in value and "user" in value:
            entries.append((key, value))
    return entries


def _select_prompt_entry(
    program: Program, prompt_key: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    entries = _candidate_prompt_entries(program)
    if not entries:
        raise ValueError(f"Program {program.id[:12]} has no logged prompts to replay")

    if prompt_key is not None:
        for key, value in entries:
            if key == prompt_key:
                return key, value
        raise KeyError(
            f"Prompt key '{prompt_key}' not found for program {program.id[:12]}; "
            f"available keys: {[k for k, _ in entries]}"
        )

    if len(entries) == 1:
        return entries[0]

    non_planner = [
        (key, value)
        for key, value in entries
        if "strategy_sampling" not in key and "planner" not in key
    ]
    if len(non_planner) == 1:
        return non_planner[0]

    raise ValueError(
        f"Ambiguous prompt selection for program {program.id[:12]}; "
        f"use --prompt-key from {[k for k, _ in entries]}"
    )


def _get_replay_eligibility(
    program: Program, prompt_key: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        selected_key, _ = _select_prompt_entry(program, prompt_key)
        return True, None, selected_key
    except Exception as exc:
        return False, str(exc), None


def _parse_replay_response(
    *,
    program: Program,
    programs: Dict[str, Program],
    prompt_key: str,
    llm_response: str,
    language: str,
) -> Tuple[Optional[str], Optional[str]]:
    if "diff" in prompt_key:
        if not program.parent_id or program.parent_id not in programs:
            return None, "Diff replay requires a parent solution"
        parent_solution = programs[program.parent_id].solution
        diff_blocks = extract_diffs(llm_response)
        if not diff_blocks:
            return None, "No valid diffs found in response"
        child_solution = apply_diff(parent_solution, llm_response)
        if child_solution == parent_solution:
            return None, "Diff SEARCH blocks did not match parent solution"
        return child_solution, None

    child_solution = parse_full_rewrite(llm_response, language)
    if not child_solution:
        return None, "No valid solution found in response"
    return child_solution, None


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _trace_entry(program: Program) -> Dict[str, Any]:
    return {
        "id": program.id,
        "iteration": program.iteration_found,
        "parent_id": program.parent_id,
        "score": get_score(program.metrics or {}),
        "prompt_keys": sorted((program.prompts or {}).keys()),
        "has_prompts": bool(program.prompts),
    }


def _write_trace_artifacts(output_dir: Path, lineage: List[Program]) -> None:
    trace = [_trace_entry(program) for program in lineage]
    with (output_dir / "trace.json").open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=_json_default)

    lines = ["# Trace", ""]
    for entry in trace:
        lines.append(
            f"- iter {entry['iteration']}: {entry['id'][:12]} "
            f"(score={entry['score']:.4f}, parent={entry['parent_id'][:12] if entry['parent_id'] else 'None'})"
        )
    _write_text(output_dir / "trace.md", "\n".join(lines) + "\n")


def _aggregate_run_summary(included_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [node["summary"] for node in included_nodes if node.get("summary")]
    if not summaries:
        return {
            "included_node_count": 0,
            "successful_replay_node_count": 0,
            "mean_exact_solution_match_rate": None,
            "mean_parse_success_rate": None,
            "mean_evaluation_success_rate": None,
            "mean_replay_score": None,
        }

    score_means = [s["score_mean"] for s in summaries if s.get("score_mean") is not None]
    exact_solution_rates = [
        s["exact_solution_match_rate"]
        for s in summaries
        if s.get("exact_solution_match_rate") is not None
    ]
    parse_success_rates = [
        s["parse_success_rate"] for s in summaries if s.get("parse_success_rate") is not None
    ]
    eval_success_rates = [
        s["evaluation_success_rate"]
        for s in summaries
        if s.get("evaluation_success_rate") is not None
    ]
    return {
        "included_node_count": len(included_nodes),
        "successful_replay_node_count": sum(1 for node in included_nodes if node.get("status") == "ok"),
        "mean_exact_solution_match_rate": (
            statistics.mean(exact_solution_rates) if exact_solution_rates else None
        ),
        "mean_parse_success_rate": (
            statistics.mean(parse_success_rates) if parse_success_rates else None
        ),
        "mean_evaluation_success_rate": (
            statistics.mean(eval_success_rates) if eval_success_rates else None
        ),
        "mean_replay_score": statistics.mean(score_means) if score_means else None,
    }


def _summarize_trials(
    *,
    original_program: Program,
    original_response: Optional[str],
    trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    response_matches = 0
    solution_matches = 0
    parse_successes = 0
    eval_successes = 0
    scores: List[float] = []

    for trial in trials:
        if original_response is not None and trial.get("response") == original_response:
            response_matches += 1
        if trial.get("solution") == original_program.solution:
            solution_matches += 1
        if trial.get("parse_error") is None:
            parse_successes += 1
        if trial.get("eval_error") is None and trial.get("score") is not None:
            eval_successes += 1
            scores.append(trial["score"])

    summary: Dict[str, Any] = {
        "original_program_id": original_program.id,
        "original_score": get_score(original_program.metrics or {}),
        "trial_count": len(trials),
        "exact_response_match_rate": response_matches / len(trials) if trials else 0.0,
        "exact_solution_match_rate": solution_matches / len(trials) if trials else 0.0,
        "parse_success_rate": parse_successes / len(trials) if trials else 0.0,
        "evaluation_success_rate": eval_successes / len(trials) if trials else 0.0,
    }

    if scores:
        summary.update(
            {
                "score_min": min(scores),
                "score_max": max(scores),
                "score_mean": statistics.mean(scores),
                "score_median": statistics.median(scores),
                "score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            }
        )
    else:
        summary.update(
            {
                "score_min": None,
                "score_max": None,
                "score_mean": None,
                "score_median": None,
                "score_std": None,
            }
        )
    return summary


async def _replay_program(
    *,
    output_dir: Path,
    program: Program,
    programs: Dict[str, Program],
    llm_pool: LLMPool,
    evaluator: Any,
    repeats: int,
    prompt_key_override: Optional[str],
    language: str,
) -> Dict[str, Any]:
    node_dir = output_dir / f"node_{program.iteration_found:03d}_{program.id[:12]}"
    node_dir.mkdir(parents=True, exist_ok=True)

    prompt_key, prompt_entry = _select_prompt_entry(program, prompt_key_override)
    original_response = None
    if prompt_entry.get("responses"):
        first = prompt_entry["responses"][0]
        if isinstance(first, str):
            original_response = first

    _write_text(node_dir / "system.txt", prompt_entry["system"])
    _write_text(node_dir / "user.txt", prompt_entry["user"])

    trial_results: List[Dict[str, Any]] = []

    for trial_idx in range(1, repeats + 1):
        response: LLMResponse = await llm_pool.generate(
            prompt_entry["system"],
            [{"role": "user", "content": prompt_entry["user"]}],
        )
        response_text = response.text or ""

        trial_dir = node_dir / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        _write_text(trial_dir / "response.txt", response_text)

        solution, parse_error = _parse_replay_response(
            program=program,
            programs=programs,
            prompt_key=prompt_key,
            llm_response=response_text,
            language=language,
        )

        eval_metrics: Optional[Dict[str, Any]] = None
        eval_error: Optional[str] = None
        score: Optional[float] = None
        if solution is not None:
            _write_text(trial_dir / f"solution{_suffix_for_language(language)}", solution)
            try:
                eval_result = await evaluator.evaluate_program(
                    solution,
                    f"{program.id}-replay-{trial_idx}-{uuid.uuid4()}",
                )
                eval_metrics = eval_result.metrics
                score = get_score(eval_metrics)
                with (trial_dir / "metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(eval_metrics, f, indent=2, default=_json_default)
            except Exception as exc:  # pragma: no cover - evaluator implementations vary
                eval_error = str(exc)
        trial_results.append(
            {
                "trial": trial_idx,
                "response": response_text,
                "solution": solution,
                "parse_error": parse_error,
                "eval_error": eval_error,
                "score": score,
                "metrics": eval_metrics,
                "model_name": response.model_name,
                "usage": response.usage,
            }
        )

    summary = _summarize_trials(
        original_program=program,
        original_response=original_response,
        trials=trial_results,
    )
    summary.update({"prompt_key": prompt_key})

    with (node_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    return {
        "program_id": program.id,
        "iteration": program.iteration_found,
        "prompt_key": prompt_key,
        "summary": summary,
        "status": "ok",
    }


def _suffix_for_language(language: str) -> str:
    mapping = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java": ".java",
        "cpp": ".cpp",
        "rust": ".rs",
        "sql": ".sql",
    }
    return mapping.get(language, ".txt")


def parse_replay_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="skydiscover-replay",
        description="Replay saved prompt context from a run to measure stability",
    )
    parser.add_argument("run_or_checkpoint_path", help="Run dir or checkpoint dir to analyze")
    parser.add_argument("evaluator", help="Evaluator file used for replay evaluation")
    parser.add_argument("--node-id", help="Program ID or unique prefix (default: best node)")
    parser.add_argument(
        "--scope",
        choices=["node", "lineage"],
        default="lineage",
        help="Replay the selected node only or the full lineage to it",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Replay count per node")
    parser.add_argument("--prompt-key", help="Prompt key to replay when a node stores multiple")
    parser.add_argument("--config", "-c", help="YAML config file")
    parser.add_argument("--model", help="Override LLM model for replay")
    parser.add_argument("--backend", choices=["openai", "litellm"], help="LLM backend")
    parser.add_argument("--api-base", help="Override API base URL")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def _default_output_dir(target: Program) -> str:
    base = build_output_dir("replay", target.id[:8])
    return base


async def main_async(argv: Optional[List[str]] = None) -> int:
    args = parse_replay_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    apply_overrides(
        config,
        model=args.model,
        api_base=args.api_base,
        backend=args.backend,
    )
    config.evaluator.evaluation_file = args.evaluator
    config.evaluator.file_suffix = config.file_suffix

    checkpoint_dir, programs, best_id, _ = _load_checkpoint(args.run_or_checkpoint_path)
    target = _resolve_program(programs, best_id, args.node_id)
    lineage = _build_lineage(programs, target)

    output_dir = Path(args.output or _default_output_dir(target))
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_trace_artifacts(output_dir, lineage)

    replay_targets = [target] if args.scope == "node" else [program for program in lineage if program.parent_id]
    llm_pool = LLMPool(config.llm.models) if args.repeats > 0 else None
    evaluator = create_evaluator(config.evaluator) if args.repeats > 0 else None

    run_summary: Dict[str, Any] = {
        "checkpoint_dir": checkpoint_dir,
        "target_program_id": target.id,
        "scope": args.scope,
        "repeats": args.repeats,
        "replayed_nodes": [],
        "excluded_nodes": [],
    }

    try:
        for program in replay_targets:
            eligible, exclusion_reason, prompt_key = _get_replay_eligibility(program, args.prompt_key)
            if not eligible:
                run_summary["excluded_nodes"].append(
                    {
                        "program_id": program.id,
                        "iteration": program.iteration_found,
                        "reason": exclusion_reason,
                    }
                )
                continue
            try:
                if args.repeats <= 0:
                    node_summary = {
                        "program_id": program.id,
                        "iteration": program.iteration_found,
                        "prompt_key": prompt_key,
                        "summary": {
                            "original_program_id": program.id,
                            "original_score": get_score(program.metrics or {}),
                            "trial_count": 0,
                            "exact_response_match_rate": None,
                            "exact_solution_match_rate": None,
                            "parse_success_rate": None,
                            "evaluation_success_rate": None,
                            "score_min": None,
                            "score_max": None,
                            "score_mean": None,
                            "score_median": None,
                            "score_std": None,
                        },
                        "status": "ok",
                    }
                else:
                    node_summary = await _replay_program(
                        output_dir=output_dir,
                        program=program,
                        programs=programs,
                        llm_pool=llm_pool,
                        evaluator=evaluator,
                        repeats=args.repeats,
                        prompt_key_override=args.prompt_key,
                        language=config.language or program.language or "python",
                    )
            except Exception as exc:
                node_summary = {
                    "program_id": program.id,
                    "iteration": program.iteration_found,
                    "status": "error",
                    "error": str(exc),
                }
            run_summary["replayed_nodes"].append(node_summary)
    finally:
        if evaluator is not None and hasattr(evaluator, "close"):
            evaluator.close()

    run_summary["aggregate"] = _aggregate_run_summary(run_summary["replayed_nodes"])
    run_summary["excluded_node_count"] = len(run_summary["excluded_nodes"])

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, default=_json_default)

    logger.info("Trace written to %s", output_dir / "trace.json")
    logger.info("Replay summary written to %s", output_dir / "summary.json")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
