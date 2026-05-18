"""Language-aware tunable-literal extractors.

The function `count_literal_candidates(source, language)` is a thin
language-dispatch helper used by `evo_replay.static.hyperparameter_counts`.
"""
from __future__ import annotations

from typing import Any, List, Optional

from .python_literals import (
    LiteralCandidate as PyLiteralCandidate,
    extract_python_literal_candidates,
)
from .cpp_literals import (
    LiteralCandidate as CppLiteralCandidate,
    extract_cpp_literal_candidates,
)


def extract_literal_candidates(source: str, language: str) -> List[Any]:
    lang = (language or "").lower()
    if lang in {"py", "python"}:
        return list(extract_python_literal_candidates(source))
    if lang in {"cpp", "c++", "cxx", "cc", "c"}:
        return list(extract_cpp_literal_candidates(source))
    return []


__all__ = [
    "PyLiteralCandidate",
    "CppLiteralCandidate",
    "extract_python_literal_candidates",
    "extract_cpp_literal_candidates",
    "extract_literal_candidates",
]
