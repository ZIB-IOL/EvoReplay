"""Counterfactual replay for breakthrough events in completed runs.

This tool identifies prompt-bearing best-so-far improvements, replays their
saved prompts through one or more LiteLLM-backed models, and evaluates whether
the counterfactual generations reproduce, improve on, or lose the original
score gain.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import re
import statistics
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from skydiscover.config import (
    Config,
    LLMModelConfig,
    apply_overrides,
    build_output_dir,
    load_config,
)
from skydiscover.evaluation import create_evaluator
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.llm_pool import LLMPool
from evo_replay.breakthrough_replay.replay import (
    _get_replay_eligibility,
    _parse_replay_response,
    _select_prompt_entry,
    _suffix_for_language,
)
from skydiscover.search.base_database import Program
from skydiscover.utils.metrics import get_score

logger = logging.getLogger("skydiscover.counterfactuals")

BUILTIN_PROMPTS = {
    "exact",
    "strict_diff",
    "no_other_context",
    "no_history",
    "alternate_context",
}
DEFAULT_PROMPT = "exact"
DEFAULT_CONTEXTS = "exact,no_other_context,no_history,alternate_context"
STRICT_DIFF_SUFFIX = """

# Output Contract
Return only one or more SEARCH/REPLACE blocks.
Do not include markdown fences.
Do not include explanations, bullet points, or prose before or after the blocks.
Every SEARCH block must be copied exactly from "# Current Solution".
""".rstrip()


@dataclass
class ProgramHistory:
    programs: Dict[str, Program]
    source_paths: Dict[str, str]
    checkpoint_dirs: List[str]
    last_iteration: int
    best_program_id: Optional[str]


@dataclass
class BreakthroughEvent:
    program: Program
    prompt_key: str
    previous_best_score: Optional[float]
    best_delta: Optional[float]
    parent_score: Optional[float]
    child_parent_delta: Optional[float]
    original_model: str


@dataclass
class ContextVariant:
    name: str
    system: str
    user: str
    excluded_reason: Optional[str] = None


@dataclass
class CounterfactualConditionSpec:
    """One explicitly requested counterfactual condition."""

    name: str
    prompt: str = DEFAULT_PROMPT
    model: Optional[str] = None
    repeats: Optional[int] = None
    attempts: Optional[int] = None
    api_base: Optional[str] = None
    backend: Optional[str] = None
    system: Optional[str] = None
    user: Optional[str] = None
    system_file: Optional[str] = None
    user_file: Optional[str] = None
    base_dir: Optional[str] = None


def _ckpt_num(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except (IndexError, ValueError):
        return 0


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "programs").is_dir() and (
        (path / "metadata.json").exists() or any((path / "programs").glob("*.json"))
    )


def _iter_checkpoint_dirs(path: str) -> List[Path]:
    root = Path(path)
    if _is_checkpoint_dir(root):
        return [root]

    candidates: List[Path] = []
    for base in (root / "checkpoints", root):
        if base.is_dir():
            candidates.extend(p for p in base.glob("checkpoint_*") if _is_checkpoint_dir(p))
    deduped = sorted({p.resolve(): p for p in candidates}.values(), key=_ckpt_num)
    if deduped:
        return deduped
    raise FileNotFoundError(f"No checkpoint directories found under {path}")


def _prompt_count(program: Program) -> int:
    prompts = program.prompts or {}
    return sum(
        1
        for value in prompts.values()
        if isinstance(value, dict) and "system" in value and "user" in value
    )


def _is_migrant(program: Program) -> bool:
    return bool((program.metadata or {}).get("migrant"))


def _prefer_program_record(existing: Program, candidate: Program) -> bool:
    existing_prompts = _prompt_count(existing)
    candidate_prompts = _prompt_count(candidate)
    if candidate_prompts != existing_prompts:
        return candidate_prompts > existing_prompts
    if _is_migrant(existing) != _is_migrant(candidate):
        return _is_migrant(existing) and not _is_migrant(candidate)

    existing_model = bool((existing.metadata or {}).get("model_name"))
    candidate_model = bool((candidate.metadata or {}).get("model_name"))
    if candidate_model != existing_model:
        return candidate_model

    # Prefer the earliest non-migration record when all useful payloads tie.
    return candidate.iteration_found < existing.iteration_found


def load_program_history(path: str) -> ProgramHistory:
    """Load and deduplicate programs from all checkpoints under a run."""
    checkpoint_dirs = _iter_checkpoint_dirs(path)
    programs: Dict[str, Program] = {}
    source_paths: Dict[str, str] = {}
    best_program_id: Optional[str] = None
    last_iteration = 0

    for checkpoint_dir in checkpoint_dirs:
        metadata_path = checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            last_iteration = max(last_iteration, int(metadata.get("last_iteration") or 0))
            if metadata.get("best_program_id"):
                best_program_id = metadata["best_program_id"]

        for program_path in sorted((checkpoint_dir / "programs").glob("*.json")):
            try:
                with program_path.open("r", encoding="utf-8") as f:
                    program = Program.from_dict(json.load(f))
            except Exception as exc:
                logger.warning("Skipping unreadable program %s: %s", program_path, exc)
                continue

            current = programs.get(program.id)
            if current is None or _prefer_program_record(current, program):
                programs[program.id] = program
                source_paths[program.id] = str(program_path)

    if not programs:
        raise ValueError(f"No programs found in checkpoints under {path}")

    return ProgramHistory(
        programs=programs,
        source_paths=source_paths,
        checkpoint_dirs=[str(p) for p in checkpoint_dirs],
        last_iteration=last_iteration,
        best_program_id=best_program_id,
    )


def _is_chronological_program(program: Program) -> bool:
    if _is_migrant(program):
        return False
    # Migration snapshots can reuse iteration 0 with a parent; those are not
    # true initial nodes and would distort best-so-far ordering.
    return not (program.iteration_found == 0 and program.parent_id)


def _event_cap(last_iteration: int, max_events_per_100: int) -> int:
    scaled = math.ceil(max_events_per_100 * max(last_iteration, 1) / 100)
    return max(1, scaled)


def _parent_score(program: Program, programs: Dict[str, Program]) -> Optional[float]:
    parent_metrics = (program.metadata or {}).get("parent_metrics") or {}
    if isinstance(parent_metrics, dict) and parent_metrics:
        return get_score(parent_metrics)
    if program.parent_id and program.parent_id in programs:
        return get_score(programs[program.parent_id].metrics or {})
    return None


def select_breakthrough_events(
    programs: Dict[str, Program],
    *,
    last_iteration: int,
    max_events_per_100: int = 15,
    prompt_key: Optional[str] = None,
    original_model_override: Optional[str] = None,
    log_model: Optional[str] = None,
    config_model: Optional[str] = None,
) -> Tuple[List[BreakthroughEvent], List[Dict[str, Any]]]:
    """Select LLM-generated best-so-far events with replayable prompts."""
    selected: List[BreakthroughEvent] = []
    excluded: List[Dict[str, Any]] = []
    best_score = float("-inf")

    chronological = sorted(
        (p for p in programs.values() if _is_chronological_program(p)),
        key=lambda p: (p.iteration_found, p.timestamp, p.id),
    )

    for program in chronological:
        score = get_score(program.metrics or {})
        if score <= best_score + 1e-12:
            continue

        previous_best = None if best_score == float("-inf") else best_score
        best_delta = None if previous_best is None else score - previous_best
        if program.parent_id:
            eligible, reason, selected_prompt_key = _get_replay_eligibility(program, prompt_key)
            if eligible and selected_prompt_key is not None:
                parent_score = _parent_score(program, programs)
                model = infer_original_model(
                    program,
                    original_model_override=original_model_override,
                    log_model=log_model,
                    config_model=config_model,
                )
                selected.append(
                    BreakthroughEvent(
                        program=program,
                        prompt_key=selected_prompt_key,
                        previous_best_score=previous_best,
                        best_delta=best_delta,
                        parent_score=parent_score,
                        child_parent_delta=(
                            score - parent_score if parent_score is not None else None
                        ),
                        original_model=model,
                    )
                )
            else:
                excluded.append(
                    {
                        "program_id": program.id,
                        "iteration": program.iteration_found,
                        "score": score,
                        "reason": reason or "not replayable",
                    }
                )
        best_score = score

    cap = _event_cap(last_iteration, max_events_per_100)
    if len(selected) > cap:
        final_best = max(selected, key=lambda event: event.program.iteration_found)
        by_delta = sorted(
            selected,
            key=lambda event: (
                event.best_delta if event.best_delta is not None else float("-inf"),
                event.program.iteration_found,
            ),
            reverse=True,
        )
        kept: Dict[str, BreakthroughEvent] = {final_best.program.id: final_best}
        for event in by_delta:
            if len(kept) >= cap:
                break
            kept[event.program.id] = event
        excluded.extend(
            {
                "program_id": event.program.id,
                "iteration": event.program.iteration_found,
                "score": get_score(event.program.metrics or {}),
                "reason": f"best-so-far event beyond cap {cap}",
            }
            for event in selected
            if event.program.id not in kept
        )
        selected = sorted(kept.values(), key=lambda event: event.program.iteration_found)

    return selected, excluded


def _candidate_log_dirs(run_or_checkpoint_path: str) -> List[Path]:
    path = Path(run_or_checkpoint_path)
    candidates = [path / "logs"]
    if path.name.startswith("checkpoint_") and path.parent.name == "checkpoints":
        candidates.append(path.parent.parent / "logs")
    if path.name == "checkpoints":
        candidates.append(path.parent / "logs")
    return candidates


def infer_log_model(run_or_checkpoint_path: str) -> Optional[str]:
    """Infer a unique original model name from run logs, if possible."""
    patterns = [
        re.compile(r"\bOpenAI LLM:\s*([^\s]+)"),
        re.compile(r"\bLiteLLM:\s*([^\s]+)"),
    ]
    models: List[str] = []
    for log_dir in _candidate_log_dirs(run_or_checkpoint_path):
        if not log_dir.is_dir():
            continue
        for log_path in sorted(log_dir.glob("*.log")):
            try:
                text = log_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in patterns:
                models.extend(match.group(1) for match in pattern.finditer(text))
    unique = sorted(set(models))
    return unique[0] if len(unique) == 1 else None


def infer_original_model(
    program: Program,
    *,
    original_model_override: Optional[str] = None,
    log_model: Optional[str] = None,
    config_model: Optional[str] = None,
) -> str:
    metadata_model = (program.metadata or {}).get("model_name")
    if isinstance(metadata_model, str) and metadata_model:
        return metadata_model
    if original_model_override:
        return original_model_override
    if log_model:
        return log_model
    if config_model:
        return config_model
    return "unknown"


def _remove_between(
    text: str,
    start_marker: str,
    end_marker: str,
) -> Tuple[Optional[str], Optional[str]]:
    start = text.find(start_marker)
    if start < 0:
        return None, f"Marker not found: {start_marker}"
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        return None, f"End marker not found after {start_marker}: {end_marker}"
    return text[:start].rstrip() + "\n\n" + text[end:].lstrip("\n"), None


def _find_other_context_span(text: str) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    marker = "\n## Other Context Solutions"
    start = text.find(marker)
    if start < 0 and text.startswith("## Other Context Solutions"):
        start = 0
    if start < 0:
        return None, "No other-context section found"
    end = text.find("\n# Current Solution", start + 1)
    if end < 0:
        return None, "Could not find current-solution marker after other-context section"
    return (start, end), None


def _format_context_section(programs: List[Program], language: str) -> str:
    lines = [
        "\n## Other Context Solutions\n",
        "These programs represent diverse approaches and creative solutions that may be relevant to the current task:\n\n",
    ]
    for index, program in enumerate(programs, start=1):
        metrics = program.metrics or {}
        combined = metrics.get("combined_score")
        if isinstance(combined, (int, float)):
            lines.append(f"### Program {index} (combined_score: {combined:.4f})\n")
        else:
            lines.append(f"### Program {index}\n")
        if metrics:
            lines.append("Score breakdown:\n")
            for key, value in metrics.items():
                if key == "combined_score":
                    continue
                if isinstance(value, float):
                    lines.append(f"  - {key}: {value:.4f}\n")
                elif isinstance(value, (int, str, bool)):
                    lines.append(f"  - {key}: {value}\n")
            lines.append("\n")
        lines.append(f"\n```{language}\n{program.solution}\n```\n\n")
    return "".join(lines)


def _alternate_context_programs(
    event: BreakthroughEvent,
    programs: Dict[str, Program],
    language: str,
) -> Tuple[Optional[str], Optional[str]]:
    original_context_ids = list(event.program.other_context_ids or [])
    if not original_context_ids:
        return None, "Program has no recorded other_context_ids"
    if event.parent_score is None:
        return None, "Parent score is unavailable"

    excluded_ids = set(original_context_ids)
    excluded_ids.add(event.program.id)
    if event.program.parent_id:
        excluded_ids.add(event.program.parent_id)

    candidates = [
        program
        for program in programs.values()
        if program.id not in excluded_ids
        and _is_chronological_program(program)
        and program.solution
        and program.iteration_found < event.program.iteration_found
    ]
    candidates.sort(
        key=lambda program: (
            abs(get_score(program.metrics or {}) - event.parent_score),
            program.iteration_found,
            program.id,
        )
    )
    if len(candidates) < len(original_context_ids):
        return None, (
            f"Need {len(original_context_ids)} alternate context programs; "
            f"found {len(candidates)}"
        )
    return _format_context_section(candidates[: len(original_context_ids)], language), None


def build_context_variants(
    event: BreakthroughEvent,
    programs: Dict[str, Program],
    *,
    requested_contexts: Iterable[str],
    language: str,
) -> List[ContextVariant]:
    prompt_key, prompt_entry = _select_prompt_entry(event.program, event.prompt_key)
    del prompt_key
    system = str(prompt_entry["system"])
    user = str(prompt_entry["user"])
    variants: List[ContextVariant] = []

    for context_name in requested_contexts:
        if context_name == "exact":
            variants.append(ContextVariant(name=context_name, system=system, user=user))
        elif context_name == "strict_diff":
            variants.append(
                ContextVariant(
                    name=context_name,
                    system=system,
                    user=user.rstrip() + "\n\n" + STRICT_DIFF_SUFFIX,
                )
            )
        elif context_name == "no_other_context":
            new_user, reason = _remove_other_context(user)
            variants.append(
                ContextVariant(
                    name=context_name,
                    system=system,
                    user=new_user or user,
                    excluded_reason=reason,
                )
            )
        elif context_name == "no_history":
            new_user, reason = _remove_between(
                user,
                "# Program Generation History",
                "# Current Solution",
            )
            variants.append(
                ContextVariant(
                    name=context_name,
                    system=system,
                    user=new_user or user,
                    excluded_reason=reason,
                )
            )
        elif context_name == "alternate_context":
            section, section_reason = _alternate_context_programs(event, programs, language)
            if section_reason:
                variants.append(
                    ContextVariant(
                        name=context_name,
                        system=system,
                        user=user,
                        excluded_reason=section_reason,
                    )
                )
                continue
            span, span_reason = _find_other_context_span(user)
            if span is None:
                variants.append(
                    ContextVariant(
                        name=context_name,
                        system=system,
                        user=user,
                        excluded_reason=span_reason,
                    )
                )
                continue
            start, end = span
            variants.append(
                ContextVariant(
                    name=context_name,
                    system=system,
                    user=user[:start] + section + user[end:],
                )
            )
        else:
            variants.append(
                ContextVariant(
                    name=context_name,
                    system=system,
                    user=user,
                    excluded_reason=f"Unknown context variant: {context_name}",
                )
            )
    return variants


def _condition_path(path: str, base_dir: Optional[str]) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir:
        resolved = Path(base_dir) / resolved
    return resolved


def _read_prompt_part(
    *,
    literal: Optional[str],
    file_path: Optional[str],
    base_dir: Optional[str],
    label: str,
) -> Optional[str]:
    if literal is not None and file_path is not None:
        raise ValueError(f"Use either {label} or {label}-file, not both")
    if literal is not None:
        return literal
    if file_path is None:
        return None
    return _condition_path(file_path, base_dir).read_text(encoding="utf-8")


def build_prompt_variant(
    event: BreakthroughEvent,
    programs: Dict[str, Program],
    *,
    spec: CounterfactualConditionSpec,
    language: str,
) -> ContextVariant:
    """Build the prompt for one condition spec.

    Built-in prompt names reuse the saved prompt with deterministic transforms.
    system/user overrides then replace the selected prompt parts wholesale.
    """
    base_prompt = spec.prompt if spec.prompt in BUILTIN_PROMPTS else DEFAULT_PROMPT
    variant = build_context_variants(
        event,
        programs,
        requested_contexts=[base_prompt],
        language=language,
    )[0]

    system_override = _read_prompt_part(
        literal=spec.system,
        file_path=spec.system_file,
        base_dir=spec.base_dir,
        label="system",
    )
    user_override = _read_prompt_part(
        literal=spec.user,
        file_path=spec.user_file,
        base_dir=spec.base_dir,
        label="user",
    )
    has_override = system_override is not None or user_override is not None

    if spec.prompt not in BUILTIN_PROMPTS and not has_override:
        return ContextVariant(
            name=spec.prompt,
            system=variant.system,
            user=variant.user,
            excluded_reason=f"Unknown prompt variant: {spec.prompt}",
        )

    return ContextVariant(
        name=spec.prompt,
        system=system_override if system_override is not None else variant.system,
        user=user_override if user_override is not None else variant.user,
        excluded_reason=variant.excluded_reason,
    )


def _remove_other_context(text: str) -> Tuple[Optional[str], Optional[str]]:
    span, reason = _find_other_context_span(text)
    if span is None:
        return None, reason
    start, end = span
    return text[:start].rstrip() + "\n\n" + text[end:].lstrip("\n"), None


def classify_score(
    *,
    score: Optional[float],
    original_score: float,
    parent_score: Optional[float],
    similar_tolerance: float,
    parse_error: Optional[str] = None,
    eval_error: Optional[str] = None,
) -> str:
    if parse_error:
        return "parse_failed"
    if eval_error or score is None:
        return "eval_failed"
    if score > original_score + similar_tolerance:
        return "better"
    if abs(score - original_score) <= similar_tolerance:
        return "similar"
    if parent_score is not None and score > parent_score + similar_tolerance:
        return "still_improves_parent"
    return "worse"


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug[:96] or "item"


def _truncate_for_prompt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def _format_retry_failed_attempts(errors: List[Dict[str, Any]], language: str) -> str:
    """Format failed retry attempts using the same shape as search prompts."""
    lines = ["\n## Previous Failed Attempts (this retry):\n"]
    lines.append("The following attempts failed. Avoid these errors:\n\n")
    for attempt in errors:
        metadata = attempt.get("metadata", {}) or {}
        err_msg = metadata.get("error", "Unknown error")
        attempt_num = metadata.get("attempt_number", "?")
        lines.append(f"### Attempt {attempt_num}:\n")
        lines.append(f"**Error:** {err_msg}\n")

        failed_solution = attempt.get("solution", "") or ""
        llm_response = attempt.get("llm_response", "") or ""

        if llm_response and (
            "SEARCH" in str(err_msg)
            or "diff" in str(err_msg).lower()
            or "visible response" in str(err_msg).lower()
        ):
            llm_response = _truncate_for_prompt(llm_response, 1500)
            lines.append(f"**Your response that failed:**\n```\n{llm_response}\n```\n\n")
        elif failed_solution:
            failed_solution = _truncate_for_prompt(failed_solution, 1500)
            lines.append(
                f"**Generated solution that failed:**\n```{language}\n{failed_solution}\n```\n"
            )
            metrics = attempt.get("metrics") or {}
            if metrics:
                metrics_text = json.dumps(metrics, indent=2, default=_json_default)
                lines.append(f"**Metrics:**\n```json\n{metrics_text}\n```\n\n")
            else:
                lines.append("\n")
    return "".join(lines)


def _with_retry_failed_attempts(
    user_prompt: str, failed_attempts: List[Dict[str, Any]], language: str
) -> str:
    if not failed_attempts:
        return user_prompt

    retry_section = _format_retry_failed_attempts(failed_attempts, language).rstrip() + "\n\n"
    current_solution_match = re.search(r"(?m)^# Current Solution[ \t]*$", user_prompt)
    if current_solution_match:
        return (
            user_prompt[: current_solution_match.start()].rstrip()
            + "\n\n"
            + retry_section
            + user_prompt[current_solution_match.start() :].lstrip("\n")
        )
    return user_prompt.rstrip() + "\n\n" + retry_section.rstrip()


def _evaluation_failure_message(metrics: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(metrics, dict):
        return None

    validity = metrics.get("validity")
    if validity in (0, -1):
        return str(
            metrics.get("error")
            or metrics.get("error_message")
            or f"Evaluation failed (validity={validity})"
        )
    if metrics.get("timeout") is True and metrics.get("validity") is None:
        return str(metrics.get("error") or metrics.get("error_message") or "Evaluation timed out")
    if metrics.get("combined_score") == 0 and metrics.get("error") is not None:
        return str(metrics.get("error"))
    return None


def _write_response_artifacts(
    attempt_dir: Path, response: LLMResponse, model_name: str
) -> Tuple[str, str, Dict[str, Any]]:
    response_text = response.text or ""
    response_file = attempt_dir / "response.txt"
    _write_text(response_file, response_text)

    reasoning_content = response.reasoning_content or ""
    reasoning_path: Optional[str] = None
    if reasoning_content:
        reasoning_file = attempt_dir / "reasoning_content.txt"
        _write_text(reasoning_file, reasoning_content)
        reasoning_path = str(reasoning_file)

    response_metadata = dict(response.response_metadata or {})
    response_metadata.update(
        {
            "model_requested": model_name,
            "model_name": response.model_name,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
            "response_text_chars": len(response_text),
            "reasoning_content_chars": len(reasoning_content),
            "reasoning_content_path": reasoning_path,
        }
    )
    response_metadata_path = attempt_dir / "response_metadata.json"
    with response_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(response_metadata, f, indent=2, default=_json_default)

    response_fields = {
        "model_requested": model_name,
        "model_name": response.model_name,
        "finish_reason": response.finish_reason,
        "usage": response.usage,
        "response_path": str(response_file),
        "response_metadata_path": str(response_metadata_path),
        "response_text_chars": len(response_text),
        "reasoning_content_path": reasoning_path,
        "reasoning_content_chars": len(reasoning_content),
    }
    return response_text, reasoning_content, response_fields


def _event_to_manifest(
    event: BreakthroughEvent, source_path: Optional[str] = None
) -> Dict[str, Any]:
    program = event.program
    return {
        "program_id": program.id,
        "iteration": program.iteration_found,
        "parent_id": program.parent_id,
        "other_context_ids": program.other_context_ids or [],
        "score": get_score(program.metrics or {}),
        "metrics": program.metrics,
        "previous_best_score": event.previous_best_score,
        "best_delta": event.best_delta,
        "parent_score": event.parent_score,
        "child_parent_delta": event.child_parent_delta,
        "changes": (program.metadata or {}).get("changes"),
        "prompt_key": event.prompt_key,
        "original_model": event.original_model,
        "source_path": source_path,
    }


def _split_csv_items(values: Optional[Iterable[str] | str]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    items: List[str] = []
    for value in values:
        items.extend(part.strip() for part in str(value).split(",") if part.strip())
    return items


def _match_iteration(events: List[BreakthroughEvent], iteration: int) -> List[BreakthroughEvent]:
    return [event for event in events if event.program.iteration_found == iteration]


def _match_program_prefix(events: List[BreakthroughEvent], prefix: str) -> List[BreakthroughEvent]:
    return [event for event in events if event.program.id.startswith(prefix)]


def _resolve_one_event_selector(
    events: List[BreakthroughEvent], selector: str
) -> List[BreakthroughEvent]:
    selector = selector.strip()
    if selector in {"*", "all"}:
        return list(events)
    if not selector:
        raise ValueError("Empty event selector")

    key = ""
    value = selector
    if ":" in selector:
        key, value = selector.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

    if key in {"idx", "index"}:
        index = int(value)
        if index < 1 or index > len(events):
            raise ValueError(f"Event index {index} is outside 1..{len(events)}")
        return [events[index - 1]]

    if key in {"iter", "iteration"}:
        matches = _match_iteration(events, int(value))
        if not matches:
            raise ValueError(f"No breakthrough event found at iteration {value}")
        return matches

    if key in {"id", "program", "program_id"}:
        matches = _match_program_prefix(events, value)
        if not matches:
            raise ValueError(f"No breakthrough event matches program prefix {value!r}")
        if len(matches) > 1:
            raise ValueError(f"Program prefix {value!r} matches multiple breakthrough events")
        return matches

    if key:
        raise ValueError(f"Unknown event selector prefix {key!r}")

    if selector.isdigit():
        iteration_matches = _match_iteration(events, int(selector))
        if iteration_matches:
            return iteration_matches
        index = int(selector)
        if 1 <= index <= len(events):
            return [events[index - 1]]

    matches = _match_program_prefix(events, selector)
    if not matches:
        raise ValueError(f"No breakthrough event matches selector {selector!r}")
    if len(matches) > 1:
        raise ValueError(f"Selector {selector!r} matches multiple breakthrough events")
    return matches


def resolve_event_selectors(
    events: List[BreakthroughEvent], selectors: Iterable[str]
) -> List[BreakthroughEvent]:
    """Resolve selectors to events while preserving breakthrough order."""
    requested = _split_csv_items(selectors)
    if not requested:
        return list(events)

    selected_by_id: Dict[str, BreakthroughEvent] = {}
    for selector in requested:
        for event in _resolve_one_event_selector(events, selector):
            selected_by_id[event.program.id] = event
    return [event for event in events if event.program.id in selected_by_id]


def _event_table(events: List[BreakthroughEvent]) -> str:
    lines = ["idx iteration program_id score previous_best best_delta parent_score original_model"]
    for index, event in enumerate(events, start=1):
        score = get_score(event.program.metrics or {})
        previous = event.previous_best_score
        delta = event.best_delta
        parent = event.parent_score
        previous_text = f"{previous:.6g}" if previous is not None else "None"
        delta_text = f"{delta:.6g}" if delta is not None else "None"
        parent_text = f"{parent:.6g}" if parent is not None else "None"
        lines.append(
            f"{index} {event.program.iteration_found} {event.program.id} "
            f"{score:.6g} {previous_text} {delta_text} {parent_text} "
            f"{event.original_model}"
        )
    return "\n".join(lines)


def _condition_spec_from_mapping(
    data: Dict[str, Any],
    *,
    base_dir: Optional[str],
    index: int,
) -> CounterfactualConditionSpec:
    if not isinstance(data, dict):
        raise ValueError("Each condition must be a JSON object")
    prompt = str(data.get("prompt") or data.get("context") or DEFAULT_PROMPT)
    model = data.get("model")
    name = str(data.get("name") or f"{prompt}_{index}")
    return CounterfactualConditionSpec(
        name=name,
        prompt=prompt,
        model=str(model) if model is not None else None,
        repeats=int(data["repeats"]) if data.get("repeats") is not None else None,
        attempts=int(data["attempts"]) if data.get("attempts") is not None else None,
        api_base=str(data["api_base"]) if data.get("api_base") is not None else None,
        backend=str(data["backend"]) if data.get("backend") is not None else None,
        system=str(data["system"]) if data.get("system") is not None else None,
        user=str(data["user"]) if data.get("user") is not None else None,
        system_file=(str(data["system_file"]) if data.get("system_file") is not None else None),
        user_file=str(data["user_file"]) if data.get("user_file") is not None else None,
        base_dir=base_dir,
    )


def load_condition_file(path: str) -> Tuple[List[CounterfactualConditionSpec], List[str]]:
    """Load named condition specs and optional event selectors from JSON."""
    condition_path = Path(path)
    with condition_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    event_selectors: List[str] = []
    if isinstance(payload, dict):
        raw_conditions = payload.get("conditions", [])
        event_selectors = _split_csv_items(payload.get("events"))
    elif isinstance(payload, list):
        raw_conditions = payload
    else:
        raise ValueError("Condition file must be a JSON object or array")

    if not isinstance(raw_conditions, list):
        raise ValueError("condition file field 'conditions' must be a list")

    base_dir = str(condition_path.parent)
    specs = [
        _condition_spec_from_mapping(item, base_dir=base_dir, index=index)
        for index, item in enumerate(raw_conditions, start=1)
    ]
    return specs, event_selectors


def _condition_spec_to_dict(spec: CounterfactualConditionSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "prompt": spec.prompt,
        "model": spec.model,
        "repeats": spec.repeats,
        "attempts": spec.attempts,
        "api_base": spec.api_base,
        "backend": spec.backend,
        "system_file": spec.system_file,
        "user_file": spec.user_file,
        "has_system_override": spec.system is not None,
        "has_user_override": spec.user is not None,
    }


def _summarize_trials(trials: List[Dict[str, Any]], original_score: float) -> Dict[str, Any]:
    scores = [trial["score"] for trial in trials if trial.get("score") is not None]
    classes: Dict[str, int] = {}
    for trial in trials:
        classification = trial.get("classification")
        if classification:
            classes[classification] = classes.get(classification, 0) + 1

    summary: Dict[str, Any] = {
        "trial_count": len(trials),
        "classes": classes,
        "parse_success_rate": (
            sum(1 for trial in trials if trial.get("parse_error") is None) / len(trials)
            if trials
            else 0.0
        ),
        "evaluation_success_rate": (
            sum(
                1
                for trial in trials
                if trial.get("eval_error") is None and trial.get("score") is not None
            )
            / len(trials)
            if trials
            else 0.0
        ),
        "better_rate": classes.get("better", 0) / len(trials) if trials else 0.0,
        "similar_or_better_rate": (
            (classes.get("better", 0) + classes.get("similar", 0)) / len(trials) if trials else 0.0
        ),
        "still_improves_parent_rate": (
            (
                classes.get("better", 0)
                + classes.get("similar", 0)
                + classes.get("still_improves_parent", 0)
            )
            / len(trials)
            if trials
            else 0.0
        ),
    }
    if scores:
        summary.update(
            {
                "score_min": min(scores),
                "score_max": max(scores),
                "score_mean": statistics.mean(scores),
                "score_median": statistics.median(scores),
                "score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "mean_delta_vs_original": statistics.mean(
                    score - original_score for score in scores
                ),
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
                "mean_delta_vs_original": None,
            }
        )
    return summary


async def _run_condition(
    *,
    condition_dir: Path,
    event: BreakthroughEvent,
    programs: Dict[str, Program],
    spec: CounterfactualConditionSpec,
    variant: ContextVariant,
    llm_pool: LLMPool,
    evaluator: Any,
    model_name: str,
    repeats: int,
    attempts: int,
    language: str,
    similar_tolerance: float,
) -> Dict[str, Any]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    _write_text(condition_dir / "system.txt", variant.system)
    _write_text(condition_dir / "user.txt", variant.user)

    trials: List[Dict[str, Any]] = []
    original_score = get_score(event.program.metrics or {})

    for trial_idx in range(1, repeats + 1):
        trial_dir = condition_dir / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        failed_attempts: List[Dict[str, Any]] = []
        attempt_payloads: List[Dict[str, Any]] = []
        final_attempt: Optional[Dict[str, Any]] = None

        for attempt_idx in range(1, attempts + 1):
            attempt_dir = trial_dir / f"attempt_{attempt_idx:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            attempt_user = _with_retry_failed_attempts(
                variant.user,
                failed_attempts,
                language,
            )
            _write_text(attempt_dir / "system.txt", variant.system)
            _write_text(attempt_dir / "user.txt", attempt_user)

            response: LLMResponse = await llm_pool.generate(
                variant.system,
                [{"role": "user", "content": attempt_user}],
            )
            response_text, reasoning_content, response_fields = _write_response_artifacts(
                attempt_dir,
                response,
                model_name,
            )

            if not response_text.strip():
                solution = None
                if reasoning_content:
                    parse_error = "Empty visible response content; reasoning_content was captured"
                else:
                    parse_error = "Empty visible response content"
            else:
                solution, parse_error = _parse_replay_response(
                    program=event.program,
                    programs=programs,
                    prompt_key=event.prompt_key,
                    llm_response=response_text,
                    language=language,
                )

            solution_path: Optional[str] = None
            eval_metrics: Optional[Dict[str, Any]] = None
            eval_error: Optional[str] = None
            score: Optional[float] = None
            if solution is not None:
                solution_file = attempt_dir / f"solution{_suffix_for_language(language)}"
                _write_text(solution_file, solution)
                solution_path = str(solution_file)
                try:
                    eval_result = await evaluator.evaluate_program(
                        solution,
                        (
                            f"{event.program.id}-counterfactual-{trial_idx}-"
                            f"attempt-{attempt_idx}-{uuid.uuid4()}"
                        ),
                    )
                    eval_metrics = eval_result.metrics
                    score = get_score(eval_metrics)
                    eval_error = _evaluation_failure_message(eval_metrics)
                    with (attempt_dir / "metrics.json").open("w", encoding="utf-8") as f:
                        json.dump(eval_metrics, f, indent=2, default=_json_default)
                except Exception as exc:  # pragma: no cover - evaluator implementations vary
                    eval_error = str(exc)

            classification = classify_score(
                score=score,
                original_score=original_score,
                parent_score=event.parent_score,
                similar_tolerance=similar_tolerance,
                parse_error=parse_error,
                eval_error=eval_error,
            )
            attempt_payload = {
                "attempt": attempt_idx,
                "system_path": str(attempt_dir / "system.txt"),
                "user_path": str(attempt_dir / "user.txt"),
                **response_fields,
                "solution_path": solution_path,
                "parse_error": parse_error,
                "eval_error": eval_error,
                "score": score,
                "metrics": eval_metrics,
                "classification": classification,
            }
            with (attempt_dir / "attempt.json").open("w", encoding="utf-8") as f:
                json.dump(attempt_payload, f, indent=2, default=_json_default)
            attempt_payloads.append(attempt_payload)
            final_attempt = attempt_payload

            if parse_error is None and eval_error is None:
                break

            if attempt_idx < attempts:
                failed_attempts.append(
                    {
                        "solution": solution or "",
                        "llm_response": response_text,
                        "metrics": eval_metrics or {},
                        "metadata": {
                            "error": parse_error or eval_error or "Unknown error",
                            "attempt_number": attempt_idx,
                        },
                    }
                )

        if final_attempt is None:
            raise RuntimeError("Counterfactual trial completed without any attempts")

        trial = {
            "trial": trial_idx,
            "attempts_configured": attempts,
            "attempts_used": final_attempt["attempt"],
            "attempts": attempt_payloads,
            "model_requested": final_attempt["model_requested"],
            "model_name": final_attempt["model_name"],
            "finish_reason": final_attempt["finish_reason"],
            "usage": final_attempt["usage"],
            "response_path": final_attempt["response_path"],
            "response_metadata_path": final_attempt["response_metadata_path"],
            "response_text_chars": final_attempt["response_text_chars"],
            "reasoning_content_path": final_attempt["reasoning_content_path"],
            "reasoning_content_chars": final_attempt["reasoning_content_chars"],
            "solution_path": final_attempt["solution_path"],
            "parse_error": final_attempt["parse_error"],
            "eval_error": final_attempt["eval_error"],
            "score": final_attempt["score"],
            "metrics": final_attempt["metrics"],
            "classification": final_attempt["classification"],
        }
        with (trial_dir / "trial.json").open("w", encoding="utf-8") as f:
            json.dump(trial, f, indent=2, default=_json_default)
        trials.append(trial)

    summary = _summarize_trials(trials, original_score)
    payload = {
        "condition": spec.name,
        "model": model_name,
        "prompt": variant.name,
        "context": variant.name,
        "condition_spec": _condition_spec_to_dict(spec),
        "status": "ok",
        "repeats": repeats,
        "attempts": attempts,
        "summary": summary,
        "trials": trials,
    }
    with (condition_dir / "condition_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return payload


def _aggregate_conditions(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    conditions = [
        condition
        for event in events
        for condition in event.get("conditions", [])
        if condition.get("status") == "ok"
    ]
    trials = [trial for condition in conditions for trial in condition.get("trials", [])]
    classes: Dict[str, int] = {}
    for trial in trials:
        classification = trial.get("classification")
        if classification:
            classes[classification] = classes.get(classification, 0) + 1
    scores = [trial["score"] for trial in trials if trial.get("score") is not None]
    return {
        "event_count": len(events),
        "condition_count": len(conditions),
        "trial_count": len(trials),
        "classes": classes,
        "score_mean": statistics.mean(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "parse_success_rate": (
            sum(1 for trial in trials if trial.get("parse_error") is None) / len(trials)
            if trials
            else 0.0
        ),
        "evaluation_success_rate": (
            sum(
                1
                for trial in trials
                if trial.get("eval_error") is None and trial.get("score") is not None
            )
            / len(trials)
            if trials
            else 0.0
        ),
    }


def _model_pool(model_cfg: LLMModelConfig) -> LLMPool:
    return LLMPool([copy.deepcopy(model_cfg)])


def _configured_model_names(config: Config) -> List[str]:
    return [model.name or "unknown_model" for model in config.llm.models]


def _expand_specs_across_models(
    specs: List[CounterfactualConditionSpec],
    model_names: List[str],
) -> List[CounterfactualConditionSpec]:
    expanded: List[CounterfactualConditionSpec] = []
    for spec in specs:
        if spec.model:
            expanded.append(spec)
            continue
        if not model_names:
            expanded.append(spec)
            continue
        for model_name in model_names:
            expanded.append(
                replace(
                    spec,
                    model=model_name,
                    name=(
                        spec.name
                        if len(model_names) == 1
                        else f"{spec.name}__{_safe_slug(model_name)}"
                    ),
                )
            )
    return expanded


def build_condition_specs(
    args: argparse.Namespace,
    config: Config,
    file_specs: Optional[List[CounterfactualConditionSpec]] = None,
) -> List[CounterfactualConditionSpec]:
    """Build explicit conditions from a condition file or CLI flags."""
    model_names = _configured_model_names(config)
    if file_specs:
        return _expand_specs_across_models(file_specs, model_names)

    has_prompt_override = any(
        [
            args.system is not None,
            args.user is not None,
            args.system_file is not None,
            args.user_file is not None,
        ]
    )
    prompts = _split_csv_items(args.prompt)
    if not prompts:
        prompts = _split_csv_items(args.contexts)
    if not prompts:
        prompts = ["custom" if has_prompt_override else DEFAULT_PROMPT]

    if args.condition_name and (len(prompts) > 1 or len(model_names) > 1):
        raise ValueError("--condition-name can only be used with one model and one prompt")

    specs: List[CounterfactualConditionSpec] = []
    for prompt in prompts:
        for model_name in model_names or [None]:
            default_name = prompt
            if model_name is not None:
                default_name = f"{prompt}__{_safe_slug(model_name)}"
            specs.append(
                CounterfactualConditionSpec(
                    name=args.condition_name or default_name,
                    prompt=prompt,
                    model=model_name,
                    repeats=args.repeats,
                    attempts=args.attempts,
                    system=args.system,
                    user=args.user,
                    system_file=args.system_file,
                    user_file=args.user_file,
                )
            )
    return specs


def _model_cfg_for_condition(
    *,
    spec: CounterfactualConditionSpec,
    config: Config,
) -> LLMModelConfig:
    for model_cfg in config.llm.models:
        if (model_cfg.name or "unknown_model") == spec.model:
            cfg = copy.deepcopy(model_cfg)
            break
    else:
        cfg = copy.deepcopy(config.llm.models[0]) if config.llm.models else LLMModelConfig()
        cfg.name = spec.model

    if spec.api_base:
        cfg.api_base = spec.api_base
    if spec.backend:
        cfg.backend = spec.backend
    if cfg.backend is None:
        cfg.backend = "litellm"
    if cfg.api_base is None:
        cfg.api_base = config.llm.api_base
    if cfg.api_key is None:
        cfg.api_key = config.llm.api_key or os.environ.get("OPENAI_API_KEY")

    shared_params = {
        "temperature": config.llm.temperature,
        "top_p": config.llm.top_p,
        "max_tokens": config.llm.max_tokens,
        "timeout": config.llm.timeout,
        "retries": config.llm.retries,
        "retry_delay": config.llm.retry_delay,
        "reasoning_effort": config.llm.reasoning_effort,
    }
    for key, value in shared_params.items():
        if getattr(cfg, key, None) is None:
            setattr(cfg, key, value)
    return cfg


def parse_counterfactual_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="skydiscover-counterfactuals",
        description="Extract breakthrough events and run explicit counterfactual conditions",
    )
    parser.add_argument("run_or_checkpoint_path", help="Run directory or checkpoint directory")
    parser.add_argument(
        "evaluator",
        nargs="?",
        help="Evaluator file used for replay evaluation; optional for --list-events or --repeats 0",
    )
    parser.add_argument("--config", "-c", help="YAML config file")
    parser.add_argument("--model", help="Counterfactual model(s), comma-separated")
    parser.add_argument("--original-model", help="Override inferred original model label")
    parser.add_argument("--backend", choices=["openai", "litellm"], default="litellm")
    parser.add_argument("--api-base", help="Override API base URL")
    parser.add_argument("--repeats", type=int, default=3, help="Default repeats per condition")
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help=(
            "LLM attempts per repeat. Failed parse/eval attempts are added to the next "
            "attempt prompt."
        ),
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        help="Only extract and list replayable breakthrough events",
    )
    parser.add_argument(
        "--event",
        action="append",
        help=(
            "Event selector to run or extract; repeatable. Supports idx:N, iter:N, "
            "program:<id-prefix>, all, or bare iteration/id-prefix."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help=(
            "Prompt variant for a condition; repeatable/comma-separated. Built-ins: "
            "exact,strict_diff,no_other_context,no_history,alternate_context. Defaults to exact."
        ),
    )
    parser.add_argument(
        "--contexts",
        help="Deprecated alias for --prompt with built-in prompt variants",
    )
    parser.add_argument("--condition-file", help="JSON file with explicit condition specs")
    parser.add_argument("--condition-name", help="Name for a single CLI-defined condition")
    parser.add_argument("--system", help="Override system prompt text for a CLI condition")
    parser.add_argument("--user", help="Override user prompt text for a CLI condition")
    parser.add_argument("--system-file", help="Read system prompt override from a file")
    parser.add_argument("--user-file", help="Read user prompt override from a file")
    parser.add_argument(
        "--max-events",
        type=int,
        default=15,
        help="Maximum best-so-far events per 100 iterations",
    )
    parser.add_argument("--prompt-key", help="Prompt key to replay when multiple are logged")
    parser.add_argument(
        "--similar-tolerance",
        type=float,
        default=0.001,
        help="Score tolerance for classifying a replay as similar",
    )
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _default_output_dir(target_path: str) -> str:
    name = Path(target_path).name or "run"
    return build_output_dir("counterfactuals", name)


async def main_async(argv: Optional[List[str]] = None) -> int:
    args = parse_counterfactual_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.repeats < 0:
        raise ValueError("--repeats must be non-negative")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.max_events <= 0:
        raise ValueError("--max-events must be positive")

    config: Config = load_config(args.config)
    apply_overrides(
        config,
        model=args.model,
        api_base=args.api_base,
        backend=args.backend,
    )
    if args.evaluator:
        config.evaluator.evaluation_file = args.evaluator
    config.evaluator.file_suffix = config.file_suffix

    history = load_program_history(args.run_or_checkpoint_path)
    log_model = infer_log_model(args.run_or_checkpoint_path)
    config_model = config.llm.models[0].name if config.llm.models else None
    events, excluded_events = select_breakthrough_events(
        history.programs,
        last_iteration=history.last_iteration,
        max_events_per_100=args.max_events,
        prompt_key=args.prompt_key,
        original_model_override=args.original_model,
        log_model=log_model,
        config_model=config_model,
    )

    file_specs: List[CounterfactualConditionSpec] = []
    file_event_selectors: List[str] = []
    if args.condition_file:
        file_specs, file_event_selectors = load_condition_file(args.condition_file)

    event_selectors = _split_csv_items(args.event) + file_event_selectors
    run_requested = args.repeats > 0 and not args.list_events
    if run_requested and not event_selectors:
        raise ValueError(
            "Counterfactual execution requires --event. Use --list-events or "
            "--repeats 0 to inspect candidate breakthrough events first."
        )
    if run_requested and not config.evaluator.evaluation_file:
        raise ValueError("Counterfactual execution requires an evaluator file")

    selected_events = resolve_event_selectors(events, event_selectors)
    output_dir = Path(args.output or _default_output_dir(args.run_or_checkpoint_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    events_root = output_dir / "events"
    events_root.mkdir(parents=True, exist_ok=True)

    if args.list_events:
        print(_event_table(selected_events))

    condition_specs: List[CounterfactualConditionSpec] = []
    if run_requested:
        condition_specs = build_condition_specs(args, config, file_specs=file_specs)
    if run_requested and not condition_specs:
        raise ValueError("No counterfactual conditions were configured")
    if run_requested:
        for spec in condition_specs:
            if spec.repeats is not None and spec.repeats < 0:
                raise ValueError(f"Condition {spec.name!r} has negative repeats")
            if spec.attempts is not None and spec.attempts <= 0:
                raise ValueError(f"Condition {spec.name!r} has non-positive attempts")
    if run_requested and any(not spec.model for spec in condition_specs) and not config.llm.models:
        raise ValueError(
            "No LLM models configured; pass --model or provide a config with llm.models"
        )

    llm_pools: Dict[str, LLMPool] = {}
    if run_requested:
        for spec in condition_specs:
            model_cfg = _model_cfg_for_condition(spec=spec, config=config)
            model_key = spec.model or model_cfg.name or "unknown_model"
            pool_key = json.dumps(
                {
                    "model": model_key,
                    "api_base": spec.api_base or model_cfg.api_base,
                    "backend": spec.backend or model_cfg.backend,
                },
                sort_keys=True,
            )
            if pool_key not in llm_pools:
                llm_pools[pool_key] = _model_pool(model_cfg)

    evaluator = create_evaluator(config.evaluator) if run_requested else None

    event_payloads: List[Dict[str, Any]] = []
    try:
        for event in selected_events:
            event_dir = (
                events_root / f"event_{event.program.iteration_found:04d}_{event.program.id[:12]}"
            )
            event_dir.mkdir(parents=True, exist_ok=True)
            prompt_key, prompt_entry = _select_prompt_entry(event.program, event.prompt_key)
            del prompt_key
            _write_text(event_dir / "original_system.txt", str(prompt_entry["system"]))
            _write_text(event_dir / "original_user.txt", str(prompt_entry["user"]))
            original_response = ""
            if prompt_entry.get("responses") and isinstance(prompt_entry["responses"][0], str):
                original_response = prompt_entry["responses"][0]
            _write_text(event_dir / "original_response.txt", original_response)

            event_manifest = _event_to_manifest(
                event,
                source_path=history.source_paths.get(event.program.id),
            )
            event_summary = {
                **event_manifest,
                "event_dir": str(event_dir),
                "conditions_requested": [_condition_spec_to_dict(spec) for spec in condition_specs],
                "conditions": [],
            }
            with (event_dir / "event_manifest.json").open("w", encoding="utf-8") as f:
                json.dump(event_summary, f, indent=2, default=_json_default)

            if run_requested and evaluator is not None:
                language = config.language or event.program.language or "python"
                for spec in condition_specs:
                    model_cfg = _model_cfg_for_condition(spec=spec, config=config)
                    model_name = spec.model or model_cfg.name or "unknown_model"
                    pool_key = json.dumps(
                        {
                            "model": model_name,
                            "api_base": spec.api_base or model_cfg.api_base,
                            "backend": spec.backend or model_cfg.backend,
                        },
                        sort_keys=True,
                    )
                    llm_pool = llm_pools[pool_key]
                    variant = build_prompt_variant(
                        event,
                        history.programs,
                        spec=spec,
                        language=language,
                    )
                    condition_dir = event_dir / f"condition_{_safe_slug(spec.name)}"
                    if variant.excluded_reason:
                        condition = {
                            "condition": spec.name,
                            "model": model_name,
                            "prompt": variant.name,
                            "context": variant.name,
                            "status": "excluded",
                            "reason": variant.excluded_reason,
                            "condition_spec": _condition_spec_to_dict(spec),
                        }
                        condition_dir.mkdir(parents=True, exist_ok=True)
                        with (condition_dir / "condition_summary.json").open(
                            "w", encoding="utf-8"
                        ) as f:
                            json.dump(condition, f, indent=2, default=_json_default)
                        event_summary["conditions"].append(condition)
                        continue

                    condition = await _run_condition(
                        condition_dir=condition_dir,
                        event=event,
                        programs=history.programs,
                        spec=spec,
                        variant=variant,
                        llm_pool=llm_pool,
                        evaluator=evaluator,
                        model_name=model_name,
                        repeats=spec.repeats if spec.repeats is not None else args.repeats,
                        attempts=(
                            spec.attempts if spec.attempts is not None else args.attempts
                        ),
                        language=language,
                        similar_tolerance=args.similar_tolerance,
                    )
                    event_summary["conditions"].append(condition)

            with (event_dir / "event_manifest.json").open("w", encoding="utf-8") as f:
                json.dump(event_summary, f, indent=2, default=_json_default)
            event_payloads.append(event_summary)
    finally:
        if evaluator is not None and hasattr(evaluator, "close"):
            evaluator.close()

    with (output_dir / "events.json").open("w", encoding="utf-8") as f:
        json.dump(event_payloads, f, indent=2, default=_json_default)

    summary = {
        "run_or_checkpoint_path": args.run_or_checkpoint_path,
        "checkpoint_dirs": history.checkpoint_dirs,
        "last_iteration": history.last_iteration,
        "best_program_id": history.best_program_id,
        "discovered_event_count": len(events),
        "selected_event_count": len(selected_events),
        "event_selectors": event_selectors,
        "excluded_events": excluded_events,
        "model_names": sorted({spec.model for spec in condition_specs if spec.model is not None}),
        "original_log_model": log_model,
        "conditions": [_condition_spec_to_dict(spec) for spec in condition_specs],
        "repeats": args.repeats,
        "attempts": args.attempts,
        "similar_tolerance": args.similar_tolerance,
        "aggregate": _aggregate_conditions(event_payloads),
        "created_at": time.time(),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    logger.info("Counterfactual events written to %s", output_dir / "events.json")
    logger.info("Counterfactual summary written to %s", output_dir / "summary.json")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
