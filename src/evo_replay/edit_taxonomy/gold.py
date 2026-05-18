"""Score the LLM judge against a hand-labelled gold set.

Default gold set lives at ``gold/examples.jsonl`` next to this file. Each row
is a JSON object with at least ``diff`` (string) and ``labels`` (list[str]).

Scoring is multi-label: per-category precision/recall/F1, micro-averaged
F1 (treating each (edit, category) cell as a binary decision), and Jaccard
similarity per edit (averaged) plus subset (exact-match) accuracy.

Usage:
    uv run python -m evo_replay.edit_taxonomy.gold score [--model M] [--gold PATH]
    uv run python -m evo_replay.edit_taxonomy.gold show  [--gold PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evo_replay.edit_taxonomy.judge import DEFAULT_MODEL, classify_diff
from evo_replay.edit_taxonomy.rubric import CATEGORY_NAMES


DEFAULT_GOLD = Path(__file__).parent / "gold" / "examples.jsonl"


@dataclass
class GoldExample:
    diff: str
    labels: List[str]
    note: str = ""
    run: str = ""
    child_id: str = ""
    parent_id: str = ""


def load_gold(path: Path = DEFAULT_GOLD) -> List[GoldExample]:
    if not path.exists():
        raise SystemExit(f"Gold set not found: {path}")
    out: List[GoldExample] = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{i}: {exc}")
        out.append(
            GoldExample(
                diff=d["diff"],
                labels=list(d.get("labels") or []),
                note=str(d.get("note", "")),
                run=str(d.get("run", "")),
                child_id=str(d.get("child_id", "")),
                parent_id=str(d.get("parent_id", "")),
            )
        )
    return out


def _f1(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


@dataclass
class PerCategoryStats:
    category: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)

    @property
    def support(self) -> int:
        return self.tp + self.fn


def score_predictions(
    examples: List[GoldExample],
    predictions: List[List[str]],
) -> Dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must align 1:1")

    per_cat = {k: PerCategoryStats(category=k) for k in CATEGORY_NAMES}
    micro_tp = micro_fp = micro_fn = 0
    jaccards: List[float] = []
    exact = 0

    for ex, pred in zip(examples, predictions):
        gold: Set[str] = set(ex.labels)
        got: Set[str] = set(pred)
        for k in CATEGORY_NAMES:
            in_gold = k in gold
            in_pred = k in got
            if in_gold and in_pred:
                per_cat[k].tp += 1
                micro_tp += 1
            elif in_pred and not in_gold:
                per_cat[k].fp += 1
                micro_fp += 1
            elif in_gold and not in_pred:
                per_cat[k].fn += 1
                micro_fn += 1
        union = gold | got
        inter = gold & got
        if not union and not gold:  # both empty
            jaccards.append(1.0)
        else:
            jaccards.append(len(inter) / len(union) if union else 0.0)
        if gold == got:
            exact += 1

    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    macro_f1 = (
        sum(per_cat[k].f1 for k in CATEGORY_NAMES) / len(CATEGORY_NAMES)
        if CATEGORY_NAMES
        else 0.0
    )

    return {
        "n": len(examples),
        "exact_match_accuracy": exact / len(examples) if examples else 0.0,
        "mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": _f1(micro_prec, micro_rec),
        "macro_f1": macro_f1,
        "per_category": {
            k: {
                "support": per_cat[k].support,
                "precision": per_cat[k].precision,
                "recall": per_cat[k].recall,
                "f1": per_cat[k].f1,
                "tp": per_cat[k].tp,
                "fp": per_cat[k].fp,
                "fn": per_cat[k].fn,
            }
            for k in CATEGORY_NAMES
        },
    }


def print_scores(scores: Dict[str, Any]) -> None:
    print(
        f"n={scores['n']}  "
        f"exact_match={scores['exact_match_accuracy']:.2%}  "
        f"mean_jaccard={scores['mean_jaccard']:.3f}  "
        f"micro_F1={scores['micro_f1']:.3f}  "
        f"macro_F1={scores['macro_f1']:.3f}"
    )
    print(f"  micro precision={scores['micro_precision']:.3f}  recall={scores['micro_recall']:.3f}")
    print(f"  per-category (support / P / R / F1 / TP-FP-FN):")
    for k, v in scores["per_category"].items():
        print(
            f"    {k:<24} "
            f"sup={v['support']:>2}  "
            f"P={v['precision']:.2f}  R={v['recall']:.2f}  F1={v['f1']:.2f}  "
            f"({v['tp']}-{v['fp']}-{v['fn']})"
        )


def run_judge_on_gold(
    examples: List[GoldExample],
    *,
    model: str = DEFAULT_MODEL,
    cache: bool = True,
) -> List[Dict[str, Any]]:
    out = []
    for i, ex in enumerate(examples, 1):
        result = classify_diff(ex.diff, model=model, cache=cache)
        out.append(
            {
                "i": i,
                "gold": ex.labels,
                "pred": result.labels,
                "rationale": result.rationale,
                "cache_hit": result.cache_hit,
                "parse_error": result.parse_error,
                "note": ex.note,
                "child_id": ex.child_id,
            }
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_score(args: argparse.Namespace) -> None:
    examples = load_gold(args.gold)
    rows = run_judge_on_gold(examples, model=args.model, cache=not args.no_cache)
    predictions = [r["pred"] for r in rows]
    scores = score_predictions(examples, predictions)
    print_scores(scores)
    print()
    print("per-edit:")
    for r in rows:
        gold = sorted(r["gold"])
        pred = sorted(r["pred"])
        ok = "✓" if set(gold) == set(pred) else "✗"
        print(
            f"  {ok} #{r['i']:>2} {r['child_id'][:8]} "
            f"gold={gold}  pred={pred}"
            + (f"  err={r['parse_error']}" if r["parse_error"] else "")
        )
        if r["rationale"]:
            print(f"      rationale: {r['rationale']}")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"scores": scores, "rows": rows}, indent=2)
        )
        print(f"Wrote {args.out}")


def _cmd_show(args: argparse.Namespace) -> None:
    examples = load_gold(args.gold)
    label_counts: Dict[str, int] = defaultdict(int)
    for ex in examples:
        if not ex.labels:
            label_counts["<none>"] += 1
        for l in ex.labels:
            label_counts[l] += 1
    print(f"{len(examples)} gold examples in {args.gold}")
    for k in CATEGORY_NAMES + ["<none>"]:
        if k in label_counts:
            print(f"  {k:<24} {label_counts[k]}")
    print()
    for i, ex in enumerate(examples, 1):
        print(
            f"#{i:>2} {ex.child_id[:8]:<10} {ex.labels}  ({len(ex.diff)} chars)"
        )
        print(f"     note: {ex.note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("score", help="run the judge on the gold set and report metrics")
    sp.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    sp.add_argument("--model", default=DEFAULT_MODEL)
    sp.add_argument("--no-cache", action="store_true")
    sp.add_argument("--out", type=Path, default=None)
    sp.set_defaults(func=_cmd_score)

    sp = sub.add_parser("show", help="print gold-set summary")
    sp.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    sp.set_defaults(func=_cmd_show)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
