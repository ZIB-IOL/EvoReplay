"""LLM-as-a-judge multi-label classifier for parent->child evolutionary edits.

The nine categories live in ``rubric.py``. The judge call lives in ``judge.py``.
The CLI entrypoint that walks a run directory and writes per-edit labels lives
in ``run_classify.py``. Gold-set utilities live in ``gold.py``.
"""
from evo_replay.edit_taxonomy.rubric import (
    CATEGORIES,
    CATEGORY_NAMES,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from evo_replay.edit_taxonomy.judge import (
    JudgeResult,
    classify_diff,
)

__all__ = [
    "CATEGORIES",
    "CATEGORY_NAMES",
    "SYSTEM_PROMPT",
    "JudgeResult",
    "build_user_prompt",
    "classify_diff",
]
