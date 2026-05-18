"""Python AST-based extractor for tunable literal candidates.

Lifted verbatim from `skydiscover.agentic_analysis.extract_python_literal_candidates`
so evo_replay has no hard dependency on the active skydiscover install.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class LiteralCandidate:
    candidate_id: str
    kind: str
    value: Any
    lineno: int
    col_offset: int
    context: str
    name_hint: Optional[str] = None


def extract_python_literal_candidates(program_source: str) -> List[LiteralCandidate]:
    """Return a conservative list of potential tunable literals from Python code."""
    try:
        tree = ast.parse(program_source)
    except SyntaxError:
        return []

    candidates: List[LiteralCandidate] = []

    def _const_kind(value: Any) -> Optional[str]:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        return None

    def _add_candidate(node: ast.AST, value: Any, context: str, name_hint: Optional[str]) -> None:
        kind = _const_kind(value)
        if kind is None:
            return
        if kind == "str" and value in {"__main__", "python", "javascript", "typescript"}:
            return
        candidate_id = f"cand_{len(candidates) + 1:03d}"
        candidates.append(
            LiteralCandidate(
                candidate_id=candidate_id,
                kind=kind,
                value=value,
                lineno=getattr(node, "lineno", 0),
                col_offset=getattr(node, "col_offset", 0),
                context=context,
                name_hint=name_hint,
            )
        )

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            if isinstance(node.value, ast.Constant):
                name_hint = None
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name_hint = node.targets[0].id
                _add_candidate(node.value, node.value.value, "assignment", name_hint)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.value, ast.Constant):
                name_hint = node.target.id if isinstance(node.target, ast.Name) else None
                _add_candidate(node.value, node.value.value, "annotated_assignment", name_hint)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                for index, arg in enumerate(node.iter.args):
                    if isinstance(arg, ast.Constant):
                        _add_candidate(arg, arg.value, f"range_arg_{index}", None)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                if isinstance(keyword.value, ast.Constant):
                    _add_candidate(
                        keyword.value,
                        keyword.value.value,
                        f"call_kw:{func_name or 'unknown'}",
                        keyword.arg,
                    )
                elif isinstance(keyword.value, ast.Dict):
                    for key_node, value_node in zip(keyword.value.keys, keyword.value.values):
                        if (
                            isinstance(key_node, ast.Constant)
                            and isinstance(value_node, ast.Constant)
                            and isinstance(key_node.value, str)
                        ):
                            _add_candidate(
                                value_node,
                                value_node.value,
                                f"call_kw_dict:{func_name or 'unknown'}",
                                key_node.value,
                            )

            interesting = {"uniform", "normal", "randint", "linspace", "default_rng", "clip"}
            if func_name in interesting:
                for index, arg in enumerate(node.args):
                    if isinstance(arg, ast.Constant):
                        _add_candidate(arg, arg.value, f"call_arg:{func_name}", f"arg_{index}")
            self.generic_visit(node)

        def visit_BinOp(self, node: ast.BinOp) -> None:
            if isinstance(node.op, (ast.Mult, ast.Div)):
                for operand in (node.left, node.right):
                    if isinstance(operand, ast.Constant):
                        _add_candidate(operand, operand.value, "binop_factor", None)
            self.generic_visit(node)

    Visitor().visit(tree)

    unique: List[LiteralCandidate] = []
    seen_keys: set[tuple[Any, int, int, str, Optional[str]]] = set()
    for candidate in candidates:
        key = (
            candidate.value,
            candidate.lineno,
            candidate.col_offset,
            candidate.context,
            candidate.name_hint,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(candidate)
    return unique
