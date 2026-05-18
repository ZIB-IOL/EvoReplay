"""Aggregate BO ceiling results across an experiment dir.

Reads each `<exp>/<label>/summary.json` produced by `run_bo`, writes
`<exp>/aggregate.{json,csv}` and prints a comparison table.

Usage:
    uv run python -m evo_replay.agentic_tuning.aggregate_bo <experiment_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _read_summary(label_dir: Path) -> Dict[str, Any] | None:
    sp = label_dir / "summary.json"
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text())
    except json.JSONDecodeError:
        return None


def aggregate(exp_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label_dir in sorted(exp_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        s = _read_summary(label_dir)
        if s is None:
            rows.append({"label": label_dir.name, "status": "missing"})
            continue
        tgt = s.get("target") or {}
        applied = s.get("applied_specs") or []
        rows.append({
            "label": label_dir.name,
            "status": "ok",
            "target_iter": tgt.get("iteration"),
            "original_score": tgt.get("original_score"),
            "baseline_score": tgt.get("baseline_score"),
            "bo_best": s.get("best_score"),
            "delta": s.get("bo_minus_original"),
            "n_calls": s.get("n_calls"),
            "n_knobs": len(applied),
            "best_params": s.get("best_params"),
            "knobs": [
                {
                    "name": a["name"],
                    "default": a["default"],
                    "low": a["low"], "high": a["high"],
                    "scale": a["scale"], "kind": a["kind"],
                }
                for a in applied
            ],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path)
    args = ap.parse_args()
    exp = args.exp_dir.resolve()
    if not exp.exists():
        sys.exit(f"Not found: {exp}")

    rows = aggregate(exp)

    json_path = exp / "aggregate.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str))

    csv_path = exp / "aggregate.csv"
    keys = ["label", "status", "target_iter", "original_score", "baseline_score",
            "bo_best", "delta", "n_calls", "n_knobs"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    print(f"\n{'label':<28} {'status':<8} {'iter':>4} "
          f"{'orig':>8} {'baseline':>8} {'BO best':>8} {'Δ':>8} "
          f"{'knobs':>5}")
    print("-" * 92)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['label']:<28} {r['status']:<8}")
            continue
        d = r.get("delta") or 0.0
        sign = "+" if d > 0 else ""
        print(f"{r['label']:<28} {'ok':<8} {r['target_iter']:>4} "
              f"{r['original_score']:>8.4f} {r['baseline_score']:>8.4f} "
              f"{r['bo_best']:>8.4f} {sign}{d:>7.4f} {r['n_knobs']:>5}")

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
