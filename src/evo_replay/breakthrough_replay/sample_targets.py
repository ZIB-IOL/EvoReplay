"""Pick replay-eligible target programs for stability experiments.

A target is eligible if:
  - It has logged `prompts` (so replay can use the saved prompt)
  - Its parent is in the active database (so replay can apply the diff)
  - It has a numeric `combined_score`

Usage:
    uv run python -m evo_replay.breakthrough_replay.sample_targets \
        --run <run_dir> --n 4 --strategy stratified

Strategies:
    best          -- run's overall best (1 target)
    breakthroughs -- best-so-far events that satisfy the eligibility filter
    stratified    -- mix: 1 best + 1 mid-trajectory + 1 regression + 1 random
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def _load_active(run_dir: Path) -> Dict[str, dict]:
    """Programs in the latest checkpoint (post-pruning)."""
    ckpts = sorted((run_dir / "checkpoints").glob("checkpoint_*"),
                   key=lambda p: int(p.name.split("_")[-1]))
    if not ckpts:
        return {}
    last = ckpts[-1]
    out = {}
    for p in last.glob("programs/*.json"):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out[d["id"]] = d
    return out


def _eligible(d: dict, active: Dict[str, dict]) -> bool:
    if not d.get("prompts"):
        return False
    par = d.get("parent_id")
    if not par or par not in active:
        return False
    sc = (d.get("metrics") or {}).get("combined_score")
    return isinstance(sc, (int, float))


def _score(d: dict) -> float:
    return float((d.get("metrics") or {}).get("combined_score") or 0.0)


def _delta(d: dict, active: Dict[str, dict]) -> float:
    par = active.get(d.get("parent_id") or "")
    return _score(d) - _score(par) if par else 0.0


def pick(run_dir: Path, n: int, strategy: str) -> List[dict]:
    active = _load_active(run_dir)
    eligible = [d for d in active.values() if _eligible(d, active)]
    if not eligible:
        return []
    eligible.sort(key=lambda d: int(d.get("iteration_found") or 0))

    if strategy == "best":
        return [max(eligible, key=_score)]

    if strategy == "breakthroughs":
        # Score-improving edits, ranked by score-delta
        improvers = [d for d in eligible if _delta(d, active) > 0]
        improvers.sort(key=lambda d: -_delta(d, active))
        return improvers[:n]

    if strategy == "stratified":
        out: List[dict] = []
        # 1 best
        best = max(eligible, key=_score)
        out.append(best)
        # 1 biggest score-improving edit (different from best)
        improvers = [d for d in eligible if _delta(d, active) > 0 and d["id"] != best["id"]]
        improvers.sort(key=lambda d: -_delta(d, active))
        if improvers:
            out.append(improvers[0])
        # 1 biggest regression
        regressions = [d for d in eligible if _delta(d, active) < 0]
        regressions.sort(key=lambda d: _delta(d, active))
        if regressions:
            out.append(regressions[0])
        # remaining slots: random non-improving edits in mid-trajectory
        mid_iter = sorted(int(d.get("iteration_found") or 0) for d in eligible)
        mid = mid_iter[len(mid_iter)//2] if mid_iter else 0
        candidates = sorted(
            [d for d in eligible if d["id"] not in {x["id"] for x in out}],
            key=lambda d: abs(int(d.get("iteration_found") or 0) - mid),
        )
        for d in candidates:
            if len(out) >= n:
                break
            out.append(d)
        return out[:n]

    raise ValueError(f"unknown strategy: {strategy}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--strategy",
                    choices=("best", "breakthroughs", "stratified"),
                    default="stratified")
    args = ap.parse_args()

    chosen = pick(args.run, args.n, args.strategy)
    if not chosen:
        sys.exit(f"No eligible targets in {args.run}")
    active = _load_active(args.run)
    for d in chosen:
        delta = _delta(d, active)
        kind = "improve" if delta > 0 else ("regress" if delta < 0 else "neutral")
        print(f"{d['id'][:12]}\t{d.get('iteration_found','?')}\t"
              f"{_score(d):.4f}\t{delta:+.4f}\t{kind}")


if __name__ == "__main__":
    main()
