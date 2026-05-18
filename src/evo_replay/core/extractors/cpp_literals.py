"""Regex-based extractor for tunable numeric literals in C/C++ source.

Mirrors the API of `python_literals.extract_python_literal_candidates`. The
patterns are the structurally-tested set used by `evo_replay.cycling`'s
`_is_hyperparam`, plus a few extras for `for(int i=0; i<NUM; ...)` and
function-call kw-arg-style numbers.

Strategy: line-by-line scan, but skip lines inside string literals and
multi-line block comments (``/* ... */``). For each kept line, run regexes
to classify and extract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class LiteralCandidate:
    candidate_id: str
    kind: str        # "int" | "float"
    value: Any
    lineno: int
    col_offset: int
    context: str     # "assignment" | "threshold" | "bitmask" | "for_bound" | "binop_factor"
    name_hint: Optional[str] = None


_NUM_RE = re.compile(r"(?<![A-Za-z_0-9.])(-?[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?|-?[0-9]+(?:[eE][+-]?[0-9]+)?)(?![A-Za-z_0-9.])")
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')

_ASSIGN_RE = re.compile(
    r"^\s*(?:const\s+|static\s+|constexpr\s+)*"
    r"(?:bool|char|short|int|long|long\s+long|unsigned|signed|float|double|size_t|int\d+_t|uint\d+_t|auto)"
    r"\s+(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;]+);"
)
_REASSIGN_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;]+);"
)
_THRESHOLD_RE = re.compile(
    r"\b(?P<name>[A-Za-z_]\w*)\s*(?:[<>]=?|==|!=)\s*(?P<num>-?[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)"
)
_BITMASK_RE = re.compile(
    r"\(\s*(?P<name>[A-Za-z_]\w*)\s*&\s*(?P<mask>[0-9]+)\s*\)\s*==\s*(?P<eq>[0-9]+)"
)
_FOR_RE = re.compile(
    r"\bfor\s*\(\s*[^;]*?(?P<name>[A-Za-z_]\w*)\s*[<>]=?\s*(?P<num>-?[0-9]+);"
)
_BINOP_RE = re.compile(
    r"(?P<num>-?[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?|-?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*[*/]\s*[A-Za-z_(]"
    r"|[A-Za-z_)]\s*[*/]\s*(?P<num2>-?[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?|-?[0-9]+(?:[eE][+-]?[0-9]+)?)"
)


def _strip_comments_and_strings(source: str) -> List[Tuple[int, str]]:
    """Return [(1-based lineno, sanitised line)]; drops lines fully inside block comments."""
    out: List[Tuple[int, str]] = []
    in_block = False
    for lineno, raw in enumerate(source.splitlines(), start=1):
        line = raw
        # Block comments
        if in_block:
            end = line.find("*/")
            if end == -1:
                continue
            line = line[end + 2:]
            in_block = False
        # Iteratively strip block comments that open and close on the same line
        while True:
            start = line.find("/*")
            if start == -1:
                break
            end = line.find("*/", start + 2)
            if end == -1:
                line = line[:start]
                in_block = True
                break
            line = line[:start] + line[end + 2:]
        # Line comment
        sl = line.find("//")
        if sl != -1:
            line = line[:sl]
        # Strings → empty placeholder
        line = _STRING_RE.sub('""', line)
        if line.strip():
            out.append((lineno, line))
    return out


def _parse_num(token: str) -> Tuple[str, Any]:
    """Return ('int'|'float', value)."""
    if "." in token or "e" in token or "E" in token:
        return ("float", float(token))
    return ("int", int(token))


def extract_cpp_literal_candidates(program_source: str) -> List[LiteralCandidate]:
    """Conservative list of tunable numeric literals from C/C++ source."""
    cleaned = _strip_comments_and_strings(program_source)
    candidates: List[LiteralCandidate] = []
    seen: set[Tuple[Any, int, int, str, Optional[str]]] = set()

    def _add(lineno: int, col: int, raw: str, context: str, name_hint: Optional[str]) -> None:
        try:
            kind, value = _parse_num(raw)
        except ValueError:
            return
        # Skip degenerate flags (0/1) when used as bools — but only in threshold/bitmask
        # contexts; assignments to 0/1 still matter for many heuristics so we keep them.
        key = (value, lineno, col, context, name_hint)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            LiteralCandidate(
                candidate_id=f"cand_{len(candidates) + 1:03d}",
                kind=kind,
                value=value,
                lineno=lineno,
                col_offset=col,
                context=context,
                name_hint=name_hint,
            )
        )

    for lineno, line in cleaned:
        # 1. Typed assignment:   double end_temp = 0.01;
        m = _ASSIGN_RE.match(line)
        if m:
            for nm in _NUM_RE.finditer(m.group("rhs")):
                _add(lineno, m.start("rhs") + nm.start(1), nm.group(1),
                     "assignment", m.group("name"))
            continue

        # 2. Bare reassignment:   end_temp = 0.01;
        m = _REASSIGN_RE.match(line)
        if m:
            for nm in _NUM_RE.finditer(m.group("rhs")):
                _add(lineno, m.start("rhs") + nm.start(1), nm.group(1),
                     "assignment", m.group("name"))
            # Don't `continue` — also let threshold etc. catch RHS comparisons

        # 3. Threshold:           if (op_choice_val < 10) ...
        for m in _THRESHOLD_RE.finditer(line):
            _add(lineno, m.start("num"), m.group("num"),
                 "threshold", m.group("name"))

        # 4. Bitmask check:       if ((flags & 7) == 0)
        for m in _BITMASK_RE.finditer(line):
            _add(lineno, m.start("mask"), m.group("mask"),
                 "bitmask", m.group("name"))
            _add(lineno, m.start("eq"), m.group("eq"),
                 "bitmask", m.group("name"))

        # 5. For-loop bound:      for (int i = 0; i < 100; ...)
        for m in _FOR_RE.finditer(line):
            _add(lineno, m.start("num"), m.group("num"),
                 "for_bound", m.group("name"))

        # 6. Multiplicative factor:   x * 0.3   or   0.3 * x
        for m in _BINOP_RE.finditer(line):
            tok = m.group("num") or m.group("num2")
            if tok is not None:
                _add(lineno, m.start(), tok, "binop_factor", None)

    return candidates
