"""Single LLM call: classify one parent->child diff into multi-label categories.

Mirrors ``agentic_tuning/llm_propose.py`` for the OpenAI-compat client and env
vars. Adds a content-addressed on-disk cache so re-runs and shared parents
across runs do not re-pay.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evo_replay.edit_taxonomy.rubric import (
    CATEGORY_NAMES,
    SYSTEM_PROMPT,
    build_user_prompt,
)


DEFAULT_MODEL = "deepseek/deepseek-chat"
CACHE_ROOT = Path(
    os.environ.get(
        "EVO_REPLAY_TAXONOMY_CACHE",
        str(Path.home() / ".cache" / "evo_replay" / "edit_taxonomy"),
    )
)


# ---------------------------------------------------------------------------
# Diff construction
# ---------------------------------------------------------------------------


def make_unified_diff(parent: str, child: str, *, n: int = 3) -> str:
    """Unified diff with `n` lines of context. No filenames in the header."""
    lines = list(
        difflib.unified_diff(
            parent.splitlines(),
            child.splitlines(),
            n=n,
            lineterm="",
        )
    )
    if lines and lines[0].startswith("---"):
        lines = lines[2:]  # drop the empty `--- ` / `+++ ` header lines
    return "\n".join(lines)


def truncate_diff(diff: str, *, max_chars: int = 16000) -> str:
    """Hard cap. Long diffs are usually architectural changes anyway and the
    judge does not need the entire file to recognise that."""
    if len(diff) <= max_chars:
        return diff
    head = diff[: max_chars // 2]
    tail = diff[-max_chars // 2 :]
    return (
        f"{head}\n"
        f"... [diff truncated, original length {len(diff)} chars] ...\n"
        f"{tail}"
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    labels: List[str]
    rationale: str
    raw_response: str = ""
    cache_hit: bool = False
    parse_error: Optional[str] = None
    diff_sha: str = ""
    model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": list(self.labels),
            "rationale": self.rationale,
            "raw_response": self.raw_response,
            "cache_hit": self.cache_hit,
            "parse_error": self.parse_error,
            "diff_sha": self.diff_sha,
            "model": self.model,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_key(diff: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(diff.encode("utf-8"))
    return h.hexdigest()


def _cache_path(key: str, root: Path = CACHE_ROOT) -> Path:
    # Two-level fanout to keep directories small.
    return root / key[:2] / f"{key}.json"


def _cache_get(key: str, root: Path = CACHE_ROOT) -> Optional[Dict[str, Any]]:
    p = _cache_path(key, root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _cache_put(key: str, value: Dict[str, Any], root: Path = CACHE_ROOT) -> None:
    p = _cache_path(key, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(p)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(
            l for l in text.splitlines() if not l.startswith("```")
        )
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(m.group(0))


def _normalise_labels(raw: Any) -> List[str]:
    """Drop unknown labels, dedupe, preserve order."""
    if not isinstance(raw, list):
        return []
    seen = set()
    out = []
    for item in raw:
        if not isinstance(item, str):
            continue
        key = item.strip()
        if key not in CATEGORY_NAMES:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _call_llm(
    system: str,
    user: str,
    *,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int = 800,
    timeout: int = 120,
) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    msg = resp.choices[0].message
    text = msg.content or getattr(msg, "reasoning_content", "") or ""
    return text


def classify_diff(
    diff: str,
    *,
    language: str = "python",
    model: str = DEFAULT_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    cache: bool = True,
    cache_root: Path = CACHE_ROOT,
    max_diff_chars: int = 16000,
) -> JudgeResult:
    """Classify one diff. Returns a JudgeResult; does not raise on parse errors."""
    diff = truncate_diff(diff, max_chars=max_diff_chars)
    key = _cache_key(diff, model)

    if cache:
        hit = _cache_get(key, cache_root)
        if hit is not None:
            hit = dict(hit)
            hit["cache_hit"] = True
            # Refresh fields that newer code may add but old caches lack.
            hit.setdefault("labels", [])
            hit.setdefault("rationale", "")
            hit.setdefault("raw_response", "")
            hit.setdefault("parse_error", None)
            hit.setdefault("diff_sha", key)
            hit.setdefault("model", model)
            return JudgeResult(
                labels=_normalise_labels(hit["labels"]),
                rationale=hit["rationale"],
                raw_response=hit["raw_response"],
                cache_hit=True,
                parse_error=hit["parse_error"],
                diff_sha=hit["diff_sha"],
                model=hit["model"],
            )

    api_base = api_base or os.environ.get("EVO_REPLAY_API_BASE")
    if not api_base:
        raise RuntimeError(
            "Set EVO_REPLAY_API_BASE to an OpenAI-compatible endpoint "
            "or pass api_base= to classify_diff()."
        )
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass api_key=.")

    user = build_user_prompt(diff, language=language)
    text = _call_llm(
        SYSTEM_PROMPT, user, model=model, api_base=api_base, api_key=api_key
    )

    parse_error: Optional[str] = None
    labels: List[str] = []
    rationale = ""
    try:
        data = _extract_json(text)
        labels = _normalise_labels(data.get("labels"))
        rationale = str(data.get("rationale", "") or "")[:500]
    except (ValueError, json.JSONDecodeError) as exc:
        parse_error = str(exc)

    result = JudgeResult(
        labels=labels,
        rationale=rationale,
        raw_response=text,
        cache_hit=False,
        parse_error=parse_error,
        diff_sha=key,
        model=model,
    )

    if cache and parse_error is None:
        _cache_put(key, result.to_dict(), cache_root)

    return result


__all__ = [
    "DEFAULT_MODEL",
    "JudgeResult",
    "classify_diff",
    "make_unified_diff",
    "truncate_diff",
]
