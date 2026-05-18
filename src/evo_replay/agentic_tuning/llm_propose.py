"""Single LLM call to identify tunable hparams + intervals in a program.

No codex, no agentic loop — just one structured-output call:
  1. Send the program source to an LLM.
  2. Ask it to identify numeric constants that look like tunable knobs and
     propose `(low, high, scale)` for each.
  3. Parse the JSON, validate, rewrite the program with a `PARAMS = {...}`
     block + literal replacements.
  4. Return the rewritten source + a `List[HparamSpec]` ready to feed into BO.

The LLM is told what NOT to tune (problem constants like `n=26`, evaluator-
facing shapes, mathematical identities). It's also told to be conservative —
default ≤ 8 knobs, since BO budget scales poorly past that.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


SYSTEM_PROMPT = """\
You are an expert at identifying tunable hyperparameters in heuristic optimisation code.

Your task: read a program and return a JSON list of numeric constants that
are good candidates for Bayesian-optimisation tuning, WITH SENSIBLE RANGES.

GOOD candidates:
  - Solver tolerances / convergence thresholds (e.g. 1e-6, 0.001)
  - Iteration / restart / sample budgets (e.g. 1000, 50, 10)
  - Step sizes, perturbation scales, learning rates
  - Soft penalty / reward weights (e.g. 0.5, 2.0)
  - Cooling / annealing rates (e.g. 0.95, 0.99)
  - Threshold dispatches (e.g. `if depth < 30: return X else return Y`)

BAD candidates (do NOT include):
  - Problem constants (n=26, n_circles=26, dim=2, GRID_SIZE=10)
  - Mathematical identities (sqrt(2), pi, e)
  - Array shapes / index bounds
  - Evaluator-facing constants (timeout values that the harness reads)
  - Constants inside identities the model should preserve (vertices, basis vectors)
  - True boolean flags (use=True)

For EACH selected knob, return:
  - name           : a unique snake_case identifier (e.g. "step_size")
  - source_literal : the EXACT numeric literal as it appears in the source
                     (e.g. "0.1", "1e-6", "0.95"). Must be byte-identical.
  - context_line   : the line of code where the literal appears
                     (verbatim, used as a disambiguator for replacement)
  - default        : the default value (= source_literal as a number)
  - low / high     : interval bounds for BO (see RANGE GUIDELINES)
  - scale          : "linear" or "log" (see RANGE GUIDELINES)
  - kind           : "int" or "float"
  - rationale      : one short sentence

RANGE GUIDELINES — cover orders of magnitude when the parameter family
warrants it. Don't propose timid ±20% intervals: BO can only discover what
the search space contains. Use these defaults:

  parameter family            scale   range relative to default          example
  --------------------------  ------  -------------------------------    -------------------
  learning rate / step size   log     [default/100, default*100]         0.01 → [1e-4, 1.0]
  temperature / amplitude     log     [default/100, default*100]         0.005 → [5e-5, 0.5]
  tolerance / threshold       log     [default/1000, default*1000]       1e-6 → [1e-9, 1e-3]
  penalty / reward weight     log     [default/100, default*100]         5.0 → [0.05, 500]
  cooling factor (0<x<1)      linear  [0.5, 0.99999]   (always wide)     0.9998 → [0.5, 0.99999]
  acceptance prob / sigmoid   linear  [0.0, 1.0]                         0.7 → [0.0, 1.0]
  restart count / pop size    log     [max(1, default/10), default*10]   10 → [1, 100]
  iteration budget            log     [default/5, default*5]             35000 → [7000, 175000]
  sample count                log     [max(1, default/10), default*10]   100 → [10, 1000]
  small-int categorical       linear  [1, max(default*5, 20)]            depth 3 → [1, 20]

If a parameter doesn't fit a family above, default to:
  - log scale if default is a positive non-integer < 1 or > 100
  - linear scale otherwise
  - range that spans at least one order of magnitude (high/low >= 10)

HARD RULES:
  - If high/low >= 100, scale MUST be "log" (BO converges much faster on
    log-scale priors over wide intervals).
  - Cooling factors and probabilities stay within their natural [0, 1] range
    even if that constrains low/high.
  - For integers: low = max(1, floor(low_proposed)); high = ceil(high_proposed).

CONSERVATIVE RULES:
  - Pick at most 8 knobs. Fewer is fine — quality over quantity.
  - If a literal appears multiple times in the source, skip it (ambiguous).
  - If unsure whether a constant is tunable, skip it.

Return ONLY a JSON object of the form:
    {"hparams": [ { ...one knob... }, ... ]}
No commentary, no markdown fences.
"""


USER_TEMPLATE = """\
Program ({language}):

```{language}
{source}
```

Return the JSON now.
"""


@dataclass
class HparamSpec:
    name: str
    source_literal: str
    context_line: str
    default: Any
    low: float
    high: float
    scale: str
    kind: str
    rationale: str


def _call_llm(system: str, user: str, *, model: str, api_base: str,
              api_key: str, max_tokens: int = 8000, timeout: int = 600) -> str:
    """One synchronous OpenAI-compat call. Falls back to reasoning_content if content is empty."""
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    text = msg.content
    if not text:
        text = getattr(msg, "reasoning_content", "") or ""
    return text or ""


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _extract_json(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from possibly-markdown-wrapped LLM output."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```"))
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in LLM response (head): {text[:200]!r}")
    return json.loads(m.group(0))


def propose_hparams(
    source: str,
    *,
    language: str = "python",
    model: str = "deepseek/deepseek-reasoner",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[HparamSpec]:
    """One LLM call → list of validated HparamSpec proposals."""
    api_base = api_base or os.environ.get("EVO_REPLAY_API_BASE")
    if not api_base:
        raise RuntimeError(
            "Set EVO_REPLAY_API_BASE or pass api_base= to point at an "
            "OpenAI-compatible endpoint.")
    api_key = api_key or os.environ["OPENAI_API_KEY"]

    user = USER_TEMPLATE.format(language=language, source=source)
    text = _call_llm(SYSTEM_PROMPT, user, model=model,
                     api_base=api_base, api_key=api_key)
    data = _extract_json(text)
    raw = data.get("hparams") or []
    out: List[HparamSpec] = []
    for h in raw:
        try:
            spec = HparamSpec(
                name=h["name"],
                source_literal=str(h["source_literal"]),
                context_line=h["context_line"],
                default=h["default"],
                low=float(h["low"]),
                high=float(h["high"]),
                scale=h.get("scale", "linear"),
                kind=h.get("kind", "float"),
                rationale=h.get("rationale", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  [skip malformed proposal: {exc}] -> {h}")
            continue
        # Sanity: the literal should appear (uniquely or not) in the source.
        if spec.source_literal not in source:
            print(f"  [skip {spec.name}: literal {spec.source_literal!r} "
                  f"not found in source]")
            continue
        out.append(spec)
    return out


def _replace_literal_in_line(line: str, literal: str, replacement: str) -> Tuple[str, bool]:
    """Replace the FIRST occurrence of literal in line, ensuring it's not part of a longer number."""
    # Find with word-ish boundaries: not preceded/followed by [0-9.eE+-]
    pat = re.compile(r"(?<![\w.])" + re.escape(literal) + r"(?![\w.])")
    new, n = pat.subn(replacement, line, count=1)
    return new, (n > 0)


def _cpp_macro(name: str) -> str:
    return "_BO_" + name.upper()


def _format_cpp_value(v: Any) -> str:
    """Render a default/BO-trial value as a C++ literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Keep scientific notation compact; C++ accepts both `1e-06` and `0.000001`.
        s = repr(v)
        return s
    return str(v)


def rewrite_program(source: str, specs: List[HparamSpec],
                    language: str = "python") -> Tuple[str, List[HparamSpec]]:
    """Replace each spec's literal in its context line with a placeholder, and
    inject a parameter block at the top of the file.

    Returns (rewritten_source, specs_actually_applied). Specs whose literal
    couldn't be found in their context_line are dropped.

    Python: placeholder = `PARAMS["name"]`; block = `PARAMS = {...}` dict.
    C++:    placeholder = `_BO_NAME` macro; block = `#define _BO_NAME value` lines.
    """
    lines = source.splitlines(keepends=True)
    applied: List[HparamSpec] = []
    is_cpp = language == "cpp"

    for spec in specs:
        target_idx = None
        ctx = spec.context_line.strip()
        for i, line in enumerate(lines):
            if ctx and ctx in line:
                target_idx = i
                break
        if target_idx is None:
            print(f"  [skip {spec.name}: context line not found]")
            continue
        replacement = _cpp_macro(spec.name) if is_cpp else f'PARAMS["{spec.name}"]'
        new_line, ok = _replace_literal_in_line(
            lines[target_idx], spec.source_literal, replacement,
        )
        if not ok:
            print(f"  [skip {spec.name}: literal {spec.source_literal!r} "
                  f"not found in context line]")
            continue
        lines[target_idx] = new_line
        applied.append(spec)

    if is_cpp:
        # Insert #define block AFTER the last #include line so headers come first.
        defines = "// === BO-injected hyperparameters ===\n" + "".join(
            f"#define {_cpp_macro(s.name)} {_format_cpp_value(s.default)}\n"
            for s in applied
        ) + "// === end BO injection ===\n\n"

        insert_at = 0
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#include"):
                insert_at = i + 1
        new_source = "".join(lines[:insert_at]) + defines + "".join(lines[insert_at:])
        return new_source, applied

    # Python path: PARAMS dict at top, after shebang/encoding/docstring.
    params = "PARAMS = {\n" + "".join(
        f'    {s.name!r}: {s.default!r},  # {s.rationale}\n' for s in applied
    ) + "}\n\n"

    insert_at = 0
    for i, line in enumerate(lines):
        st = line.lstrip()
        if st.startswith("#!") or st.startswith("# -*-") or st.startswith('"""'):
            continue
        insert_at = i
        break

    new_source = "".join(lines[:insert_at]) + params + "".join(lines[insert_at:])
    return new_source, applied


def render_with_params(rewritten: str, params: Dict[str, Any],
                       language: str = "python") -> str:
    """Replace the parameter block with the given values.

    Python: replaces the `PARAMS = {...}` dict.
    C++:    rewrites each `#define _BO_<NAME> <value>` line in the BO block.
    """
    if language == "cpp":
        out = rewritten
        for name, value in params.items():
            macro = _cpp_macro(name)
            new_line = f"#define {macro} {_format_cpp_value(value)}"
            pat = re.compile(rf"^#define\s+{re.escape(macro)}\s+.*$", flags=re.MULTILINE)
            out, n = pat.subn(new_line, out, count=1)
            if n == 0:
                raise ValueError(f"No `#define {macro}` line in rewritten source")
        return out

    # Python path: replace PARAMS = {...} block (greedy brace walk).
    m = re.search(r"^PARAMS\s*=\s*\{", rewritten, flags=re.MULTILINE)
    if not m:
        raise ValueError("No PARAMS block in rewritten source.")
    start = m.start()
    depth = 0
    i = m.end() - 1
    while i < len(rewritten):
        if rewritten[i] == "{":
            depth += 1
        elif rewritten[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    else:
        raise ValueError("PARAMS block has no matching close brace.")
    new_block = "PARAMS = " + repr(params)
    return rewritten[:start] + new_block + rewritten[end:]


__all__ = ["HparamSpec", "propose_hparams", "rewrite_program", "render_with_params"]
