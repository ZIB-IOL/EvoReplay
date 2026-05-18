"""Multi-label edit-taxonomy rubric.

Nine categories illustrated in ``evolutionary_edits_taxonomy_two_col.html``.
Each parent->child diff may receive zero, one, or several labels — labels are
*independent*, NOT exclusive, and there is NO precedence ordering: the judge
picks every category that genuinely characterises the edit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Category:
    key: str               # snake_case identifier used in JSON output
    display: str           # human-readable name
    one_liner: str         # short definition
    positive_cues: List[str]
    negative_cues: List[str]
    canonical_example: str  # verbatim excerpt from the taxonomy figure
    score_signal: str       # what score-delta pattern is typical of this category;
                            # a *prior* the judge can lean on when the diff is ambiguous


CATEGORIES: List[Category] = [
    Category(
        key="bug_fix",
        display="Bug fix",
        one_liner=(
            "Corrects faulty behaviour in pre-existing logic — a degeneracy "
            "guard, an off-by-one fix, a sign correction, an empty-input "
            "check, a typo in a constant, a wrong index."
        ),
        positive_cues=[
            "Adds a guard around code that already existed (e.g. `if x <= 0: continue`).",
            "Fixes a wrong constant, wrong index, or wrong comparison operator.",
            "Handles an edge case the parent crashed or silently mis-handled on.",
            "Net change is small AND localised AND the surrounding logic is unchanged.",
        ],
        negative_cues=[
            "A guard that only protects new code is not a bug fix — it is part of that new code's category.",
            "Replacing a numeric constant with a different numeric constant is hyperparameter_tuning, not bug_fix.",
            "Restructuring how the algorithm works is architectural_change, even if the parent had a latent bug.",
        ],
        canonical_example=(
            "+ if cur_min <= 1e-12:\n"
            "+     continue"
        ),
        score_signal=(
            "Often produces a LARGE positive score jump: the parent was "
            "silently broken on some inputs and the fix unblocks them. A "
            "guard with no score impact is more likely defensive coding "
            "around new logic — i.e. NOT a bug_fix."
        ),
    ),
    Category(
        key="external_dependency",
        display="External dependency",
        one_liner=(
            "Adds or removes an import of an external library, or starts/"
            "stops calling into one."
        ),
        positive_cues=[
            "New `import X` / `from X import Y` line for a library not previously used.",
            "Removal of an import that is no longer referenced.",
            "First call to a library function (e.g. `scipy.spatial.ConvexHull`).",
        ],
        negative_cues=[
            "Re-importing a stdlib module that was already used elsewhere is not a meaningful new dependency.",
            "Importing a module that is then never referenced does not count.",
        ],
        canonical_example=(
            "+ from scipy.spatial import ConvexHull\n"
            "...\n"
            "- return min_area(pts) / hull_const\n"
            "+ hull = ConvexHull(pts)\n"
            "+ return min_area(pts) / hull.volume"
        ),
        score_signal=(
            "Score impact varies — the import alone is neutral; what matters "
            "is the algorithm change it enables. Often co-occurs with "
            "architectural_change or local_refinement when the new library "
            "supplants hand-written logic."
        ),
    ),
    Category(
        key="architectural_change",
        display="Architectural change",
        one_liner=(
            "Substantively changes the algorithmic approach or the data "
            "representation — a different family of solver, a different "
            "geometry, a different state representation."
        ),
        positive_cues=[
            "Replaces one algorithm with a fundamentally different one (greedy -> DP, exhaustive -> heuristic).",
            "Changes the geometric construction (single 14-gon -> two concentric 7-gons).",
            "Changes the loop / search structure (single-pass -> nested two-phase).",
        ],
        negative_cues=[
            "Adjusting numeric knobs of the same algorithm is hyperparameter_tuning.",
            "Adding a second phase that *augments* an existing approach is composition.",
            "Pure code reorganisation with identical behaviour is refactor.",
        ],
        canonical_example=(
            "- # regular 14-gon\n"
            "- angles = np.arange(14) * 2*pi/14\n"
            "+ # two concentric heptagons\n"
            "+ outer = R * heptagon(angles_7)\n"
            "+ inner = f*R * heptagon(angles_7 + pi/7)"
        ),
        score_signal=(
            "High variance: can be the largest positive jump in a run "
            "(breakthrough) or a regression. Look at the magnitude of the "
            "diff and the surrounding rationale rather than the score alone."
        ),
    ),
    Category(
        key="composition",
        display="Composition",
        one_liner=(
            "Combines two existing strategies into a hybrid or multi-stage "
            "pipeline — adds a phase rather than replacing the approach."
        ),
        positive_cues=[
            "Hill-climb augmented with simulated-annealing acceptance.",
            "Coarse pass followed by a fine-refinement pass over the same state.",
            "Two-stage solver where the second stage operates on the first stage's output.",
        ],
        negative_cues=[
            "If the new stage REPLACES the old logic rather than augmenting it, that is architectural_change.",
            "Refining an existing single-stage formula in place is local_refinement.",
        ],
        canonical_example=(
            "+ start_temp = max(cur * 0.5, 1e-8)\n"
            "  for step in range(n_iters):\n"
            "+     temp = start_temp * (1 - step/n_iters)\n"
            "      if cand > cur: cur = cand\n"
            "+     else:\n"
            "+         p = exp((cand - cur) / temp)\n"
            "+         if rng() < p: cur = cand"
        ),
        score_signal=(
            "Usually a moderate positive jump: the second strategy patches "
            "the first one's failure mode (escapes a local optimum, finishes "
            "an unfinished search). Pure regressions here are rare."
        ),
    ),
    Category(
        key="local_refinement",
        display="Local refinement",
        one_liner=(
            "Small targeted change to the FORM of an existing expression or "
            "step — a better normalisation, a better initialisation formula, "
            "a tighter loss — without changing the algorithm."
        ),
        positive_cues=[
            "Rescales an existing quantity using a different formula (unit radius -> unit hull area).",
            "Replaces a hand-tuned constant with a closed-form expression.",
            "Tweaks an objective / loss / projection step within the same algorithm.",
        ],
        negative_cues=[
            "Pure numeric-literal change with no formula change is hyperparameter_tuning.",
            "Wholly replacing the algorithm is architectural_change.",
        ],
        canonical_example=(
            "+ # scale to unit hull area\n"
            "+ hull_area = (n/2) * sin(2*pi/n)\n"
            "+ points *= sqrt(1.0 / hull_area)"
        ),
        score_signal=(
            "Typically a small-to-moderate positive jump. Larger jumps point "
            "toward architectural_change or bug_fix instead."
        ),
    ),
    Category(
        key="pruning",
        display="Pruning",
        one_liner=(
            "Removes substantial code, often delegating to a baseline or "
            "dropping an unused feature — net deletion is the dominant signal."
        ),
        positive_cues=[
            "Large net deletion of lines.",
            "Replacement of an elaborate implementation with a single call to an existing baseline.",
            "Removal of a code path or feature that the parent had but is no longer reached.",
        ],
        negative_cues=[
            "Net deletion that comes from extracting a helper into a separate function is refactor, not pruning.",
            "Removing a guard while keeping the rest is more likely a bug-fix reversal or hyperparameter_tuning.",
        ],
        canonical_example=(
            "- cols = ceil(sqrt(n))\n"
            "- # ... 40+ lines of placement\n"
            "- return centers, radii, sum_radii\n"
            "+ return baseline_packing(instance)"
        ),
        score_signal=(
            "Score usually drops to a baseline level or stays flat. A score "
            "JUMP after pure pruning suggests the deleted code was actively "
            "harmful — flag bug_fix as a co-label in that case."
        ),
    ),
    Category(
        key="refactor",
        display="Refactor",
        one_liner=(
            "Restructures code without changing externally observable "
            "behaviour — extract function, rename, reorder."
        ),
        positive_cues=[
            "Extracts a block of code into a new helper function.",
            "Renames a variable or function consistently across the file.",
            "Splits a large block into smaller named steps with no semantic change.",
        ],
        negative_cues=[
            "If the extracted helper is also faster, it is BOTH refactor AND efficiency.",
            "If the extracted code now does something different, the change is not pure refactor.",
        ],
        canonical_example=(
            "- def heilbronn_convex14():\n"
            "-     # 80-line monolithic body\n"
            "+ def _heptagon_layout():\n"
            "+     ... # extracted\n"
            "+ def heilbronn_convex14():\n"
            "+     return _heptagon_layout()"
        ),
        score_signal=(
            "Zero score change BY DEFINITION. A score change after a "
            "supposed refactor means it wasn't a pure refactor — pick the "
            "category that actually changed behaviour instead."
        ),
    ),
    Category(
        key="efficiency",
        display="Efficiency",
        one_liner=(
            "Speeds up code without changing what it computes — hoisting, "
            "vectorisation, caching, asymptotic improvement on the same task."
        ),
        positive_cues=[
            "Hoists invariant work out of a loop.",
            "Replaces Python-level loops with NumPy vectorisation.",
            "Adds memoisation / a precomputed table that does not change outputs.",
        ],
        negative_cues=[
            "If the change also alters the result (different points, different ordering), it is not pure efficiency — likely also local_refinement or architectural_change.",
            "Reducing iteration count to be faster is hyperparameter_tuning, not efficiency.",
        ],
        canonical_example=(
            "+ COMBOS = np.array(combinations(range(11), 3))\n"
            "  def min_area(pts):\n"
            "-     idx = np.array(combinations(...))\n"
            "+     p1,p2,p3 = pts[COMBOS[:,0]], ..."
        ),
        score_signal=(
            "Outputs are the same, so direct score impact is zero. But on "
            "time-bounded problems, the freed budget often lets MORE "
            "iterations run, producing a moderate POSITIVE jump indirectly."
        ),
    ),
    Category(
        key="hyperparameter_tuning",
        display="Hyperparameter tuning",
        one_liner=(
            "Changes the value of one or more numeric literals — temperatures, "
            "iteration counts, step sizes, time limits — without changing the "
            "surrounding code structure."
        ),
        positive_cues=[
            "`x = 1.85` -> `x = 1.90` style edits.",
            "Several numeric-literal changes in a config block, structure unchanged.",
            "Edit lines whose number-collapsed skeleton matches between parent and child.",
        ],
        negative_cues=[
            "If the formula AROUND the literal changed, that is local_refinement.",
            "If a literal moved into a new code path, the dominant label is whatever introduced that path.",
        ],
        canonical_example=(
            "- double time_limit_seconds = 1.85;\n"
            "+ double time_limit_seconds = 1.90;"
        ),
        score_signal=(
            "Small, often noisy score swings in either direction. Repeated "
            "tuning that ratchets the score up by tiny amounts is the "
            "characteristic signal."
        ),
    ),
]

CATEGORY_NAMES: List[str] = [c.key for c in CATEGORIES]


def _format_category_block(cat: Category) -> str:
    pos = "\n".join(f"      - {c}" for c in cat.positive_cues)
    neg = "\n".join(f"      - {c}" for c in cat.negative_cues)
    ex = "\n".join(f"      {ln}" for ln in cat.canonical_example.splitlines())
    return (
        f"  * `{cat.key}` ({cat.display})\n"
        f"    definition: {cat.one_liner}\n"
        f"    positive cues:\n{pos}\n"
        f"    negative cues:\n{neg}\n"
        f"    canonical diff:\n{ex}\n"
        f"    typical score impact: {cat.score_signal}"
    )


SYSTEM_PROMPT = """\
You are an expert reviewer of evolutionary code edits. You receive ONE unified
diff representing a single parent -> child mutation produced by an LLM-driven
code-search system. Your job is to assign every category from the taxonomy
that genuinely characterises that edit.

Output a JSON object of the form:
    {{"labels": ["<key1>", "<key2>", ...], "rationale": "<one short sentence>"}}

RULES:
  - The categories are INDEPENDENT and NON-EXCLUSIVE. Apply every category
    that genuinely fits. Most edits will receive 1-3 labels; pure single-aspect
    edits receive exactly one.
  - There is NO precedence ordering. Do not pick a "primary" label.
  - Be parsimonious. Do not tag a category for a tiny incidental aspect of
    the edit; tag it only if a non-trivial portion of the diff is explained
    by that category.
  - Use the EXACT keys below. Do not invent new categories. Do not output
    display names.
  - If the diff is empty or only touches comments / whitespace, return
    `{{"labels": [], "rationale": "..."}}`.
  - Output ONLY the JSON object, no markdown fences, no commentary.

EACH CATEGORY BELOW INCLUDES A `typical score impact` LINE — this is a *prior*
about what score-delta pattern usually accompanies that category. Use it as
soft evidence when the diff is ambiguous (e.g. a single guard line could be
defensive coding OR a real bug fix; the bug-fix prior says real bug fixes
typically produce large positive jumps, so weight that against the diff
content). DO NOT invent a score delta if you don't see one — just use the
prior to disambiguate when the diff alone is genuinely ambiguous.

THE NINE CATEGORIES:

{categories}
""".format(
    categories="\n\n".join(_format_category_block(c) for c in CATEGORIES)
)


USER_TEMPLATE = """\
Parent -> child unified diff (language: {language}):

```diff
{diff}
```

Return the JSON now.
"""


def build_user_prompt(diff: str, *, language: str = "python") -> str:
    return USER_TEMPLATE.format(language=language, diff=diff)


__all__ = [
    "Category",
    "CATEGORIES",
    "CATEGORY_NAMES",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "build_user_prompt",
]
