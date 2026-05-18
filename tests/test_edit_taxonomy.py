"""Tests for evo_replay.edit_taxonomy.

No network: the LLM call is monkey-patched. We verify rubric structure, prompt
assembly, JSON parsing including malformed responses, the on-disk cache, and
the multi-label scoring math.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evo_replay.edit_taxonomy import judge as judge_mod
from evo_replay.edit_taxonomy import gold as gold_mod
from evo_replay.edit_taxonomy.gold import GoldExample, score_predictions
from evo_replay.edit_taxonomy.judge import (
    JudgeResult,
    classify_diff,
    make_unified_diff,
    truncate_diff,
)
from evo_replay.edit_taxonomy.rubric import (
    CATEGORIES,
    CATEGORY_NAMES,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def test_categories_have_unique_keys() -> None:
    assert len(CATEGORIES) == 9
    assert len(set(CATEGORY_NAMES)) == 9
    expected = {
        "bug_fix", "external_dependency", "architectural_change", "composition",
        "local_refinement", "pruning", "refactor", "efficiency",
        "hyperparameter_tuning",
    }
    assert set(CATEGORY_NAMES) == expected


def test_system_prompt_mentions_each_category() -> None:
    for cat in CATEGORIES:
        assert cat.key in SYSTEM_PROMPT, f"{cat.key} missing from prompt"


def test_build_user_prompt_embeds_diff() -> None:
    diff = "@@ -1,1 +1,1 @@\n-old\n+new"
    out = build_user_prompt(diff, language="python")
    assert "```diff" in out
    assert diff in out
    assert "(language: python)" in out


def test_make_unified_diff_basic() -> None:
    a = "line1\nline2\nline3\n"
    b = "line1\nLINE2\nline3\n"
    d = make_unified_diff(a, b, n=1)
    assert "-line2" in d
    assert "+LINE2" in d
    # No bare `--- ` / `+++ ` filename headers (we strip them).
    assert not d.startswith("---")


def test_truncate_diff_long_input() -> None:
    big = "x" * 50000
    out = truncate_diff(big, max_chars=1000)
    assert len(out) < len(big)
    assert "diff truncated" in out


# ---------------------------------------------------------------------------
# LLM-call patching
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_llm(monkeypatch, tmp_path):
    """Replace _call_llm with a stub and redirect the cache to a tmp dir."""
    calls = []

    def _stub(system, user, *, model, api_base, api_key, max_tokens=800, timeout=120):
        calls.append({"system": system, "user": user, "model": model})
        return json.dumps(
            {
                "labels": ["hyperparameter_tuning", "pruning", "not_a_real_label"],
                "rationale": "stubbed",
            }
        )

    monkeypatch.setattr(judge_mod, "_call_llm", _stub)
    monkeypatch.setattr(judge_mod, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setenv("EVO_REPLAY_API_BASE", "http://stub")
    monkeypatch.setenv("OPENAI_API_KEY", "stub-key")
    return calls


def test_classify_diff_drops_unknown_labels(patched_llm, tmp_path) -> None:
    res = classify_diff("dummy diff", cache_root=tmp_path / "cache")
    assert isinstance(res, JudgeResult)
    assert res.labels == ["hyperparameter_tuning", "pruning"]
    assert res.rationale == "stubbed"
    assert res.cache_hit is False
    assert res.parse_error is None


def test_classify_diff_cache_hit_skips_llm(patched_llm, tmp_path) -> None:
    cache = tmp_path / "cache"
    classify_diff("same diff", cache_root=cache)
    assert len(patched_llm) == 1
    classify_diff("same diff", cache_root=cache)
    assert len(patched_llm) == 1, "second call should hit the cache"


def test_classify_diff_handles_unparseable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        judge_mod, "_call_llm",
        lambda *a, **k: "not even close to JSON",
    )
    monkeypatch.setattr(judge_mod, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setenv("EVO_REPLAY_API_BASE", "http://stub")
    monkeypatch.setenv("OPENAI_API_KEY", "stub-key")
    res = classify_diff("dummy", cache_root=tmp_path / "cache")
    assert res.parse_error is not None
    assert res.labels == []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _ex(labels):
    return GoldExample(diff="x", labels=labels)


def test_score_perfect_predictions() -> None:
    examples = [_ex(["bug_fix"]), _ex(["refactor", "efficiency"]), _ex([])]
    preds = [["bug_fix"], ["refactor", "efficiency"], []]
    s = score_predictions(examples, preds)
    assert s["exact_match_accuracy"] == 1.0
    assert s["mean_jaccard"] == 1.0
    assert s["micro_f1"] == 1.0


def test_score_partial_match_jaccard() -> None:
    examples = [_ex(["bug_fix", "refactor"])]
    preds = [["bug_fix"]]
    s = score_predictions(examples, preds)
    # |gold ∩ pred| / |gold ∪ pred| = 1/2
    assert s["mean_jaccard"] == 0.5
    assert s["exact_match_accuracy"] == 0.0
    assert s["per_category"]["refactor"]["fn"] == 1
    assert s["per_category"]["bug_fix"]["tp"] == 1


def test_score_handles_empty_gold_and_pred() -> None:
    s = score_predictions([_ex([])], [[]])
    assert s["exact_match_accuracy"] == 1.0
    assert s["mean_jaccard"] == 1.0


# ---------------------------------------------------------------------------
# Gold-set on disk
# ---------------------------------------------------------------------------


def test_default_gold_set_is_loadable() -> None:
    examples = gold_mod.load_gold()
    assert len(examples) >= 9, "gold set should cover most categories"
    seen = set()
    for ex in examples:
        for l in ex.labels:
            assert l in CATEGORY_NAMES, f"{l!r} is not a real category"
            seen.add(l)
    # We expect coverage of at least most categories.
    assert len(seen) >= 6, f"only {seen} represented"


# ---------------------------------------------------------------------------
# Agreement: sampling + Cohen's kappa
# ---------------------------------------------------------------------------


from evo_replay.edit_taxonomy.agreement import (
    LABEL_KEYS,
    cohens_kappa,
    score_pairs,
    stratified_sample,
)


def test_label_keys_cover_every_category() -> None:
    assert set(LABEL_KEYS.values()) == set(CATEGORY_NAMES)
    assert len(set(LABEL_KEYS)) == len(LABEL_KEYS), "duplicate single-key bindings"


def test_kappa_perfect_agreement() -> None:
    assert cohens_kappa(tp=10, fp=0, fn=0, tn=10) == pytest.approx(1.0)


def test_kappa_chance_agreement_is_zero() -> None:
    # Both raters say "yes" 50% of the time, independently.
    # Expected by chance = 0.5*0.5 + 0.5*0.5 = 0.5; observed = 0.5; kappa = 0.
    assert cohens_kappa(tp=25, fp=25, fn=25, tn=25) == pytest.approx(0.0)


def test_kappa_perfect_disagreement_is_negative() -> None:
    # Raters always disagree.
    assert cohens_kappa(tp=0, fp=10, fn=10, tn=0) < 0


def test_kappa_all_no_returns_one() -> None:
    # Both raters say "no" to every item — degenerate but defined.
    assert cohens_kappa(tp=0, fp=0, fn=0, tn=10) == pytest.approx(1.0)


def test_score_pairs_per_category() -> None:
    judge = {
        "a": ["bug_fix", "hyperparameter_tuning"],
        "b": ["refactor"],
        "c": [],
    }
    human = {
        "a": ["bug_fix"],
        "b": ["refactor"],
        "c": ["pruning"],
    }
    m = score_pairs(judge, human)
    assert m["n"] == 3
    bf = m["per_category"]["bug_fix"]
    assert bf["tp"] == 1 and bf["fp"] == 0 and bf["fn"] == 0
    hp = m["per_category"]["hyperparameter_tuning"]
    assert hp["tp"] == 0 and hp["fp"] == 1, "judge over-applied hp"
    pr = m["per_category"]["pruning"]
    assert pr["fn"] == 1, "judge missed pruning that the human flagged"


def test_stratified_sample_balances_strata() -> None:
    rows = (
        [{"labels": ["hyperparameter_tuning"]}] * 100
        + [{"labels": ["bug_fix"]}] * 5
        + [{"labels": ["refactor", "efficiency"]}] * 3
        + [{"labels": []}] * 2
    )
    sample = stratified_sample(rows, n=12, seed=0)
    assert len(sample) == 12
    # Round-robin should pull from every stratum before re-pulling from any.
    strata_seen = set()
    for r in sample[:4]:
        strata_seen.add(tuple(sorted(r["labels"] or [])) or ("<none>",))
    assert len(strata_seen) == 4, f"first 4 picks should hit all 4 strata: {strata_seen}"
