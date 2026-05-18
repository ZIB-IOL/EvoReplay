"""Aggregate edit-taxonomy outputs across completed runs and emit plots.

Scans one or more dataset roots for completed edit-taxonomy outputs
(`analysis/llm_edit_taxonomy.{jsonl,summary.json}`), then produces:

  - overall label-effect bars
  - stage-conditioned heatmaps for edit impact
  - co-occurrence heatmaps
  - CSV/JSON summaries for downstream analysis

The script is incremental by design: it only uses runs that already have a
completed `llm_edit_taxonomy.summary.json`, so it can be rerun while
classification is still ongoing.

Usage:
    uv run python -m evo_replay.edit_taxonomy.aggregate_plots \
        /path/to/evo_trace_anon --out /tmp/edit_taxonomy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from evo_replay.core.checkpoints import best_so_far_targets, lineage, load_programs
from evo_replay.edit_taxonomy.rubric import CATEGORIES
from evo_replay.static.aggregate_loc import backend_of, domain_of


CATEGORY_KEYS = [c.key for c in CATEGORIES]
CATEGORY_INDEX = {key: idx for idx, key in enumerate(CATEGORY_KEYS)}
CATEGORY_DISPLAY = {c.key: c.display for c in CATEGORIES}
BACKEND_DISPLAY = {
    "openevolve_native": "OpenEvolve",
    "evox": "EvoX",
    "gepa_native": "GEPA",
    "shinkaevolve": "ShinkaEvolve",
}
SUCCESS_SUBSETS = (
    ("final_best_lineage", "Final-best lineage"),
    ("best_so_far", "Best-so-far updates"),
)
LABEL_COLORS = {
    "bug_fix": "#4e79a7",
    "external_dependency": "#f28e2b",
    "architectural_change": "#e15759",
    "composition": "#76b7b2",
    "local_refinement": "#59a14f",
    "pruning": "#edc948",
    "refactor": "#b07aa1",
    "efficiency": "#ff9da7",
    "hyperparameter_tuning": "#9c755f",
}
LABEL_COUNT_BUCKETS = ("0", "1", "2", "3", "4+")
_MODEL_HEAD_TOKENS = {
    "deepseek",
    "dpsk",
    "dschat",
    "claude",
    "gemini",
    "gpt",
    "local",
    "gflash",
}


def stage_names(n_stages: int) -> list[str]:
    edges = np.linspace(0.0, 1.0, n_stages + 1)
    names: list[str] = []
    for i in range(n_stages):
        lo = int(round(100 * edges[i]))
        hi = int(round(100 * edges[i + 1]))
        names.append(f"{lo}-{hi}%\nof run")
    return names


def problem_of(name: str) -> str:
    parts = name.split("_")
    out: list[str] = []
    for part in parts:
        head = part.split("-", 1)[0]
        if head in _MODEL_HEAD_TOKENS:
            break
        if part.isdigit() and out:
            break
        out.append(part)
    return "_".join(out) if out else name


def pretty_group_value(group_field: str, value: str) -> str:
    if group_field == "backend":
        return BACKEND_DISPLAY.get(value, value)
    if group_field == "task":
        return value.replace("_", " ")
    return value


def find_completed_runs(roots: list[Path]) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for root in roots:
        for summary in sorted(root.rglob("llm_edit_taxonomy.summary.json")):
            jsonl = summary.with_name("llm_edit_taxonomy.jsonl")
            if not jsonl.exists():
                continue
            found.append((summary.parent.parent, jsonl))
    return sorted(set(found))


def _score_scale(records: list[dict[str, Any]]) -> float:
    abs_scores: list[float] = []
    for rec in records:
        for key in ("score", "parent_score"):
            value = rec.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                abs_scores.append(abs(float(value)))
    if not abs_scores:
        return 1.0
    return max(float(np.quantile(abs_scores, 0.95)), 1.0)


def _label_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item in CATEGORY_KEYS:
            out.append(item)
    return sorted(set(out))


def load_rows(
    roots: list[Path],
    *,
    n_stages: int,
    include_backends: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_pairs = find_completed_runs(roots)
    run_counts = Counter()
    skipped_runs: list[str] = []

    for run_dir, jsonl_path in run_pairs:
        backend = backend_of(run_dir)
        if include_backends is not None and backend not in include_backends:
            continue
        task = problem_of(run_dir.name)
        programs = load_programs(run_dir)
        best_targets = best_so_far_targets(programs)
        best_so_far_ids = {str(program.get("id") or "") for program in best_targets}
        final_lineage_ids: set[str] = set()
        if best_targets:
            final_lineage_ids = {
                node.program_id
                for node in lineage(programs, str(best_targets[-1].get("id") or ""))
            }

        try:
            raw_records = [
                json.loads(line)
                for line in jsonl_path.read_text().splitlines()
                if line.strip()
            ]
        except (OSError, ValueError, json.JSONDecodeError):
            skipped_runs.append(str(run_dir))
            continue

        if not raw_records:
            skipped_runs.append(str(run_dir))
            continue

        max_iteration = max(int(rec.get("iteration") or 0) for rec in raw_records)
        max_iteration = max(max_iteration, 1)
        scale = _score_scale(raw_records)
        run_counts[backend] += 1

        for rec in raw_records:
            delta = rec.get("score_delta")
            delta_value = (
                float(delta)
                if isinstance(delta, (int, float)) and math.isfinite(delta)
                else None
            )
            iteration = int(rec.get("iteration") or 0)
            x_norm = iteration / max_iteration if max_iteration > 0 else 0.0
            stage_idx = min(int(x_norm * n_stages), n_stages - 1)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "backend": backend,
                    "domain": domain_of(run_dir),
                    "task": task,
                    "iteration": iteration,
                    "parent_iteration": int(rec.get("parent_iteration") or 0),
                    "x_norm": x_norm,
                    "stage_idx": stage_idx,
                    "stage": stage_names(n_stages)[stage_idx],
                    "score_delta": delta_value,
                    "score_delta_norm": (
                        delta_value / scale if delta_value is not None else None
                    ),
                    "labels": _label_list(rec.get("labels")),
                    "is_best_so_far": str(rec.get("program_id") or "") in best_so_far_ids,
                    "is_final_best_lineage": (
                        str(rec.get("program_id") or "") in final_lineage_ids
                    ),
                }
            )

    meta = {
        "n_completed_runs": sum(run_counts.values()),
        "runs_by_backend": dict(sorted(run_counts.items())),
        "skipped_runs": skipped_runs,
    }
    return rows, meta


def _label_values(
    rows: list[dict[str, Any]],
    label: str,
    *,
    stage_idx: int | None = None,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if label not in row["labels"]:
            continue
        if stage_idx is not None and row["stage_idx"] != stage_idx:
            continue
        value = row["score_delta_norm"]
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def label_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in CATEGORY_KEYS:
        values = _label_values(rows, label)
        positives = [v for v in values if v > 0]
        out.append(
            {
                "label": label,
                "display": CATEGORY_DISPLAY[label],
                "n_edits": len(values),
                "positive_rate": (
                    len(positives) / len(values) if values else float("nan")
                ),
                "mean_delta_norm": float(np.mean(values)) if values else float("nan"),
                "median_delta_norm": (
                    float(np.median(values)) if values else float("nan")
                ),
                "median_positive_delta_norm": (
                    float(np.median(positives)) if positives else float("nan")
                ),
            }
        )
    return out


def label_stage_rows(rows: list[dict[str, Any]], n_stages: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    names = stage_names(n_stages)
    for label in CATEGORY_KEYS:
        for stage_idx in range(n_stages):
            values = _label_values(rows, label, stage_idx=stage_idx)
            positives = [v for v in values if v > 0]
            out.append(
                {
                    "label": label,
                    "display": CATEGORY_DISPLAY[label],
                    "stage_idx": stage_idx,
                    "stage": names[stage_idx],
                    "n_edits": len(values),
                    "positive_rate": (
                        len(positives) / len(values) if values else float("nan")
                    ),
                    "mean_delta_norm": (
                        float(np.mean(values)) if values else float("nan")
                    ),
                    "median_delta_norm": (
                        float(np.median(values)) if values else float("nan")
                    ),
                }
            )
    return out


def cooccurrence_matrices(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(CATEGORY_KEYS)
    counts = np.zeros((n, n), dtype=float)
    for row in rows:
        labels = row["labels"]
        for a in labels:
            i = CATEGORY_INDEX[a]
            counts[i, i] += 1
        for a in labels:
            i = CATEGORY_INDEX[a]
            for b in labels:
                j = CATEGORY_INDEX[b]
                if i == j:
                    continue
                counts[i, j] += 1
    conditional = np.full_like(counts, np.nan, dtype=float)
    for i in range(n):
        if counts[i, i] <= 0:
            continue
        conditional[i, :] = counts[i, :] / counts[i, i]
        conditional[i, i] = 1.0
    return counts, conditional


def _annotate_matrix(
    ax: Any,
    matrix: np.ndarray,
    *,
    fmt: str,
    threshold: float | None = None,
    use_abs: bool = False,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    source = np.abs(finite) if use_abs else finite
    auto_threshold = float(np.nanmedian(source))
    use_threshold = auto_threshold if threshold is None else threshold
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            metric = abs(value) if use_abs else value
            color = "white" if metric >= use_threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )


def _matrix_labels() -> list[str]:
    return [CATEGORY_DISPLAY[key] for key in CATEGORY_KEYS]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *CATEGORY_KEYS])
        for idx, label in enumerate(CATEGORY_KEYS):
            writer.writerow([label, *matrix[idx].tolist()])


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    label_matrix = np.zeros((len(rows), len(CATEGORY_KEYS)), dtype=float)
    deltas = np.full(len(rows), np.nan, dtype=float)
    for row_idx, row in enumerate(rows):
        for label in row["labels"]:
            label_matrix[row_idx, CATEGORY_INDEX[label]] = 1.0
        delta = row["score_delta_norm"]
        if isinstance(delta, (int, float)) and math.isfinite(delta):
            deltas[row_idx] = float(delta)

    phi = np.full((len(CATEGORY_KEYS), len(CATEGORY_KEYS)), np.nan, dtype=float)
    for i in range(len(CATEGORY_KEYS)):
        for j in range(len(CATEGORY_KEYS)):
            phi[i, j] = _safe_corr(label_matrix[:, i], label_matrix[:, j])

    valid = np.isfinite(deltas)
    score_corr_rows: list[dict[str, Any]] = []
    for idx, label in enumerate(CATEGORY_KEYS):
        present = label_matrix[:, idx] > 0.5
        n_present = int(np.sum(present))
        corr = _safe_corr(label_matrix[valid, idx], deltas[valid])
        score_corr_rows.append(
            {
                "label": label,
                "display": CATEGORY_DISPLAY[label],
                "n_edits": n_present,
                "score_delta_correlation": corr,
            }
        )
    return phi, score_corr_rows


def helpfulness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_rows = [
        row
        for row in rows
        if isinstance(row["score_delta_norm"], (int, float))
        and math.isfinite(float(row["score_delta_norm"]))
    ]
    n_total = len(scored_rows)
    n_positive = sum(float(row["score_delta_norm"]) > 0 for row in scored_rows)
    baseline_positive_rate = n_positive / n_total if n_total else float("nan")

    out: list[dict[str, Any]] = []
    for label in CATEGORY_KEYS:
        present_rows = [row for row in scored_rows if label in row["labels"]]
        n_present = len(present_rows)
        n_present_positive = sum(float(row["score_delta_norm"]) > 0 for row in present_rows)
        n_present_nonpositive = n_present - n_present_positive
        n_absent_positive = n_positive - n_present_positive
        n_absent_nonpositive = (n_total - n_positive) - n_present_nonpositive
        present_positive_rate = (
            n_present_positive / n_present if n_present else float("nan")
        )
        odds_ratio = (
            ((n_present_positive + 0.5) * (n_absent_nonpositive + 0.5))
            / ((n_present_nonpositive + 0.5) * (n_absent_positive + 0.5))
        )
        out.append(
            {
                "label": label,
                "display": CATEGORY_DISPLAY[label],
                "n_edits": n_present,
                "baseline_positive_rate": baseline_positive_rate,
                "present_positive_rate": present_positive_rate,
                "positive_rate_uplift": (
                    present_positive_rate - baseline_positive_rate
                    if np.isfinite(present_positive_rate)
                    and np.isfinite(baseline_positive_rate)
                    else float("nan")
                ),
                "odds_ratio": odds_ratio,
            }
        )
    return out


def _label_count_bucket(n_labels: int) -> str:
    return "4+" if n_labels >= 4 else str(n_labels)


def label_count_distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(_label_count_bucket(len(row["labels"])) for row in rows)
    total = len(rows)
    out: list[dict[str, Any]] = []
    for bucket in LABEL_COUNT_BUCKETS:
        n_edits = int(counts.get(bucket, 0))
        out.append(
            {
                "label_count_bucket": bucket,
                "n_edits": n_edits,
                "share_edits": n_edits / total if total else float("nan"),
            }
        )
    return out


def label_multilabel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in CATEGORY_KEYS:
        label_rows = [row for row in rows if label in row["labels"]]
        n_label_edits = len(label_rows)
        n_single = sum(len(row["labels"]) == 1 for row in label_rows)
        n_multi = sum(len(row["labels"]) >= 2 for row in label_rows)
        out.append(
            {
                "label": label,
                "display": CATEGORY_DISPLAY[label],
                "n_label_edits": n_label_edits,
                "n_single_label_edits": n_single,
                "n_multi_label_edits": n_multi,
                "single_label_rate": (
                    n_single / n_label_edits if n_label_edits else float("nan")
                ),
                "multi_label_rate": (
                    n_multi / n_label_edits if n_label_edits else float("nan")
                ),
            }
        )
    return out


def top_combination_rows(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, ...]] = Counter()
    n_labeled_edits = 0
    n_multilabel_edits = 0
    for row in rows:
        labels = tuple(sorted(row["labels"]))
        if labels:
            n_labeled_edits += 1
        if len(labels) < 2:
            continue
        n_multilabel_edits += 1
        counter[labels] += 1

    out: list[dict[str, Any]] = []
    for labels, n_edits in counter.most_common(top_n):
        display = " + ".join(CATEGORY_DISPLAY[label] for label in labels)
        out.append(
            {
                "label_combo": "|".join(labels),
                "display": display,
                "n_labels": len(labels),
                "n_edits": int(n_edits),
                "share_labeled_edits": (
                    n_edits / n_labeled_edits if n_labeled_edits else float("nan")
                ),
                "share_multilabel_edits": (
                    n_edits / n_multilabel_edits if n_multilabel_edits else float("nan")
                ),
            }
        )
    return out


def helpfulness_rows_by_group(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    group_values: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_value in group_values:
        group_rows = [row for row in rows if str(row[group_field]) == group_value]
        for metric_row in helpfulness_rows(group_rows):
            out.append(
                {
                    **metric_row,
                    "group_field": group_field,
                    "group": group_value,
                    "group_display": pretty_group_value(group_field, group_value),
                }
            )
    return out


def enrichment_rows(rows: list[dict[str, Any]], *, subset: str) -> list[dict[str, Any]]:
    subset_rows = [row for row in rows if subset_match(row, subset)]
    n_all = len(rows)
    n_subset = len(subset_rows)
    out: list[dict[str, Any]] = []
    for label in CATEGORY_KEYS:
        n_all_label = sum(label in row["labels"] for row in rows)
        n_subset_label = sum(label in row["labels"] for row in subset_rows)
        share_all = n_all_label / n_all if n_all else float("nan")
        share_subset = n_subset_label / n_subset if n_subset else float("nan")
        out.append(
            {
                "subset": subset,
                "subset_display": dict(SUCCESS_SUBSETS).get(subset, subset),
                "label": label,
                "display": CATEGORY_DISPLAY[label],
                "n_all_edits": n_all,
                "n_subset_edits": n_subset,
                "n_all_label": n_all_label,
                "n_subset_label": n_subset_label,
                "share_all": share_all,
                "share_subset": share_subset,
                "enrichment_ratio": (
                    share_subset / share_all if share_all and np.isfinite(share_all) else float("nan")
                ),
            }
        )
    return out


def subset_match(row: dict[str, Any], subset: str) -> bool:
    if subset == "all":
        return True
    if subset == "final_best_lineage":
        return bool(row["is_final_best_lineage"])
    if subset == "best_so_far":
        return bool(row["is_best_so_far"])
    raise ValueError(f"Unknown subset: {subset}")


def group_order(rows: list[dict[str, Any]], group_field: str) -> list[str]:
    if group_field == "backend":
        discovered = {str(row["backend"]) for row in rows}
        ordered = [
            key
            for key in ("openevolve_native", "evox", "gepa_native", "shinkaevolve")
            if key in discovered
        ]
        return ordered + sorted(discovered - set(ordered))

    counts = Counter(str(row[group_field]) for row in rows)
    return [
        value
        for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def label_share_rows(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    group_values: list[str],
    subset: str,
) -> list[dict[str, Any]]:
    subset_rows = [row for row in rows if subset_match(row, subset)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset_rows:
        grouped[str(row[group_field])].append(row)

    out: list[dict[str, Any]] = []
    for label in CATEGORY_KEYS:
        for group_value in group_values:
            bucket = grouped.get(group_value, [])
            n_subset_edits = len(bucket)
            n_label_edits = sum(label in row["labels"] for row in bucket)
            out.append(
                {
                    "subset": subset,
                    "subset_display": dict(SUCCESS_SUBSETS).get(subset, subset),
                    "group_field": group_field,
                    "group": group_value,
                    "group_display": pretty_group_value(group_field, group_value),
                    "label": label,
                    "display": CATEGORY_DISPLAY[label],
                    "n_subset_edits": n_subset_edits,
                    "n_label_edits": n_label_edits,
                    "label_share": (
                        n_label_edits / n_subset_edits if n_subset_edits else float("nan")
                    ),
                }
            )
    return out


def _share_matrix(
    share_rows: list[dict[str, Any]],
    *,
    subset: str,
    group_values: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    share = np.full((len(CATEGORY_KEYS), len(group_values)), np.nan, dtype=float)
    counts = np.zeros((len(CATEGORY_KEYS), len(group_values)), dtype=float)
    for row in share_rows:
        if row["subset"] != subset:
            continue
        i = CATEGORY_INDEX[row["label"]]
        j = group_values.index(row["group"])
        share[i, j] = row["label_share"]
        counts[i, j] = row["n_label_edits"]
    return share, counts


def plot_overall_effects(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metric: str,
) -> None:
    labels = [row["display"] for row in rows]
    median_delta = [row["median_delta_norm"] for row in rows]
    positive_rate = [row["positive_rate"] for row in rows]
    counts = [row["n_edits"] for row in rows]
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7.5))

    if metric == "median_delta_norm":
        values = median_delta
        colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in values]
        ax.barh(ypos, values, color=colors)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("median normalized score delta")
        xpad = 0.01 * max(1.0, np.nanmax(np.abs(values)))
        for idx, (value, count) in enumerate(zip(values, counts)):
            if not np.isfinite(value):
                continue
            ha = "left" if value >= 0 else "right"
            xpos = value + xpad if ha == "left" else value - xpad
            ax.text(xpos, idx, f"n={count}", va="center", ha=ha, fontsize=9)
    elif metric == "positive_rate":
        values = positive_rate
        ax.barh(ypos, values, color="#4c78a8")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("positive-rate")
        for idx, (value, count) in enumerate(zip(values, counts)):
            if not np.isfinite(value):
                continue
            ax.text(value + 0.02, idx, f"n={count}", va="center", ha="left", fontsize=9)
    else:
        raise ValueError(f"Unknown overall metric: {metric}")

    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_ylabel("edit label")
    fig.subplots_adjust(left=0.32, right=0.97, bottom=0.12)
    fig.savefig(path)
    plt.close(fig)


def _delta_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1e-6
    if lo > 0:
        lo = 0.0
    if hi < 0:
        hi = 0.0
    return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)


def _centered_norm(values: np.ndarray, *, center: float) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TwoSlopeNorm(vmin=center - 1.0, vcenter=center, vmax=center + 1.0)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1e-6
    lo = min(lo, center)
    hi = max(hi, center)
    return TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)


def plot_stage_heatmap(
    path: Path,
    rows: list[dict[str, Any]],
    n_stages: int,
    *,
    metric: str,
) -> None:
    delta_matrix = np.full((len(CATEGORY_KEYS), n_stages), np.nan, dtype=float)
    rate_matrix = np.full((len(CATEGORY_KEYS), n_stages), np.nan, dtype=float)
    for i, label in enumerate(CATEGORY_KEYS):
        for stage_idx in range(n_stages):
            values = _label_values(rows, label, stage_idx=stage_idx)
            if not values:
                continue
            delta_matrix[i, stage_idx] = float(np.median(values))
            rate_matrix[i, stage_idx] = sum(v > 0 for v in values) / len(values)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    stage_labels = stage_names(n_stages)
    label_names = _matrix_labels()
    if metric == "median_delta_norm":
        matrix = delta_matrix
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=_delta_norm(matrix))
        colorbar_label = "median normalized score delta"
    elif metric == "positive_rate":
        matrix = rate_matrix
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        colorbar_label = "positive-rate"
    else:
        raise ValueError(f"Unknown stage metric: {metric}")

    ax.set_xticks(np.arange(n_stages), stage_labels)
    ax.set_yticks(np.arange(len(label_names)), label_names)
    ax.set_xlabel("stage")
    ax.set_ylabel("edit label")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=colorbar_label)
    fig.subplots_adjust(left=0.32, right=0.96, bottom=0.18)
    fig.savefig(path)
    plt.close(fig)


def plot_cooccurrence(path: Path, matrix: np.ndarray, *, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    label_names = _matrix_labels()
    if kind == "counts":
        im = ax.imshow(matrix, aspect="equal", cmap="Blues")
        xlabel = "label B"
        colorbar_label = "edits containing both A and B"
    elif kind == "conditional":
        im = ax.imshow(matrix, aspect="equal", cmap="magma", vmin=0.0, vmax=1.0)
        xlabel = "P(label B | label A)"
        colorbar_label = "conditional probability"
    else:
        raise ValueError(f"Unknown cooccurrence kind: {kind}")

    ax.set_xticks(np.arange(len(label_names)), label_names, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(label_names)), label_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("label A")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=colorbar_label)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.24)
    fig.savefig(path)
    plt.close(fig)


def plot_correlation_heatmap(path: Path, phi: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    label_names = _matrix_labels()
    im = ax.imshow(phi, aspect="equal", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(label_names)), label_names, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(label_names)), label_names)
    ax.set_xlabel("edit label B")
    ax.set_ylabel("edit label A")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="phi correlation")
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.24)
    fig.savefig(path)
    plt.close(fig)


def plot_correlation_bars(path: Path, score_corr_rows: list[dict[str, Any]]) -> None:
    order = sorted(
        range(len(score_corr_rows)),
        key=lambda idx: (
            float("-inf")
            if not np.isfinite(score_corr_rows[idx]["score_delta_correlation"])
            else score_corr_rows[idx]["score_delta_correlation"]
        ),
        reverse=True,
    )
    ordered_labels = [score_corr_rows[idx]["display"] for idx in order]
    ordered_corrs = [score_corr_rows[idx]["score_delta_correlation"] for idx in order]
    ordered_counts = [score_corr_rows[idx]["n_edits"] for idx in order]
    fig, ax = plt.subplots(figsize=(10, 9))
    ypos = np.arange(len(ordered_labels))
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in ordered_corrs]
    ax.barh(ypos, ordered_corrs, color=colors)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(ypos, ordered_labels)
    ax.invert_yaxis()
    ax.set_xlabel("correlation with normalized score delta")
    for idx, (value, count) in enumerate(zip(ordered_corrs, ordered_counts)):
        if not np.isfinite(value):
            continue
        ha = "left" if value >= 0 else "right"
        xpos = value + 0.01 if value >= 0 else value - 0.01
        ax.text(xpos, idx, f"n={count}", va="center", ha=ha, fontsize=9)

    fig.subplots_adjust(left=0.32, right=0.96, bottom=0.12)
    fig.savefig(path)
    plt.close(fig)


def plot_helpfulness_bars(path: Path, helpful_rows: list[dict[str, Any]], *, metric: str) -> None:
    sorted_rows = sorted(
        helpful_rows,
        key=lambda row: float(row[metric]) if np.isfinite(float(row[metric])) else float("-inf"),
        reverse=True,
    )
    labels = [row["display"] for row in sorted_rows]
    values = [float(row[metric]) for row in sorted_rows]
    counts = [int(row["n_edits"]) for row in sorted_rows]
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 8))
    if metric == "positive_rate_uplift":
        colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in values]
        ax.barh(ypos, values, color=colors)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("positive-rate uplift over baseline")
    elif metric == "odds_ratio":
        colors = ["#4c78a8" if value >= 1.0 else "#e76f51" for value in values]
        ax.barh(ypos, values, color=colors)
        ax.set_xscale("log")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("odds ratio for positive score delta")
    else:
        raise ValueError(f"Unknown helpfulness metric: {metric}")
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_ylabel("edit label")
    fig.subplots_adjust(left=0.32, right=0.96, bottom=0.12)
    fig.savefig(path)
    plt.close(fig)


def plot_enrichment_bars(path: Path, enrich_rows: list[dict[str, Any]]) -> None:
    sorted_rows = sorted(
        enrich_rows,
        key=lambda row: float(row["enrichment_ratio"])
        if np.isfinite(float(row["enrichment_ratio"]))
        else float("-inf"),
        reverse=True,
    )
    labels = [row["display"] for row in sorted_rows]
    values = [float(row["enrichment_ratio"]) for row in sorted_rows]
    counts = [int(row["n_subset_label"]) for row in sorted_rows]
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2a9d8f" if value >= 1.0 else "#e76f51" for value in values]
    ax.barh(ypos, values, color=colors)
    ax.set_xscale("log")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("enrichment ratio vs all edits")
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_ylabel("edit label")
    fig.subplots_adjust(left=0.32, right=0.96, bottom=0.12)
    fig.savefig(path)
    plt.close(fig)


def plot_success_composition(
    path: Path,
    share_rows: list[dict[str, Any]],
    *,
    group_field: str,
    group_values: list[str],
    xlabel: str,
    subset: str,
) -> None:
    fig_width = 12 if len(group_values) <= 6 else max(16, 1.0 * len(group_values) + 7)
    fig, ax = plt.subplots(figsize=(fig_width, 8.5))
    label_names = _matrix_labels()
    display_groups = [pretty_group_value(group_field, value) for value in group_values]
    matrix, _ = _share_matrix(share_rows, subset=subset, group_values=group_values)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(display_groups)), display_groups, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(label_names)), label_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("edit label")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="label share within subset")
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.24)
    fig.savefig(path)
    plt.close(fig)


def plot_all_edit_composition(
    path: Path,
    share_rows: list[dict[str, Any]],
    *,
    group_values: list[str],
    group_field: str,
    xlabel: str,
) -> None:
    matrix, _ = _share_matrix(share_rows, subset="all", group_values=group_values)
    label_names = _matrix_labels()
    display_groups = [pretty_group_value(group_field, value) for value in group_values]
    fig_width = 12 if len(group_values) <= 6 else max(16, 1.0 * len(group_values) + 7)
    fig, ax = plt.subplots(figsize=(fig_width, 8.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(display_groups)), display_groups, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(label_names)), label_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("edit label")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="label share among all edits")
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.24)
    fig.savefig(path)
    plt.close(fig)


def time_distribution_rows(
    rows: list[dict[str, Any]],
    *,
    subset: str,
    n_bins: int,
) -> list[dict[str, Any]]:
    counts = np.zeros((n_bins, len(CATEGORY_KEYS)), dtype=float)
    edit_counts = np.zeros(n_bins, dtype=float)
    for row in rows:
        if not subset_match(row, subset):
            continue
        labels = row["labels"]
        if not labels:
            continue
        bin_idx = min(int(float(row["x_norm"]) * n_bins), n_bins - 1)
        edit_counts[bin_idx] += 1
        for label in labels:
            counts[bin_idx, CATEGORY_INDEX[label]] += 1

    out: list[dict[str, Any]] = []
    for bin_idx in range(n_bins):
        total_labels = float(np.sum(counts[bin_idx]))
        x_left = bin_idx / n_bins
        x_right = (bin_idx + 1) / n_bins
        x_mid = 0.5 * (x_left + x_right)
        for label_idx, label in enumerate(CATEGORY_KEYS):
            n_label = float(counts[bin_idx, label_idx])
            out.append(
                {
                    "subset": subset,
                    "subset_display": dict(SUCCESS_SUBSETS).get(subset, subset),
                    "bin_idx": bin_idx,
                    "x_left": x_left,
                    "x_right": x_right,
                    "x_mid": x_mid,
                    "label": label,
                    "display": CATEGORY_DISPLAY[label],
                    "n_edits": int(edit_counts[bin_idx]),
                    "n_label_occurrences": int(n_label),
                    "n_label_occurrences_total": int(total_labels),
                    "label_distribution": (
                        n_label / total_labels if total_labels > 0 else float("nan")
                    ),
                }
            )
    return out


def plot_time_distribution(
    path: Path,
    dist_rows: list[dict[str, Any]],
    *,
    subset: str,
    ylabel: str,
) -> None:
    subset_rows = [row for row in dist_rows if row["subset"] == subset]
    if not subset_rows:
        return
    bins = sorted({int(row["bin_idx"]) for row in subset_rows})
    xs = np.array(
        [next(row["x_mid"] for row in subset_rows if int(row["bin_idx"]) == bin_idx) for bin_idx in bins]
    )
    ys: list[np.ndarray] = []
    colors: list[str] = []
    labels: list[str] = []
    for label in CATEGORY_KEYS:
        label_rows = [row for row in subset_rows if row["label"] == label]
        label_rows.sort(key=lambda row: int(row["bin_idx"]))
        ys.append(
            np.array(
                [
                    float(row["label_distribution"])
                    if np.isfinite(float(row["label_distribution"]))
                    else 0.0
                    for row in label_rows
                ]
            )
        )
        colors.append(LABEL_COLORS[label])
        labels.append(CATEGORY_DISPLAY[label])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.stackplot(xs, ys, labels=labels, colors=colors, alpha=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("normalized iteration in run")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.28)
    fig.savefig(path)
    plt.close(fig)


def plot_time_heatmap(
    path: Path,
    dist_rows: list[dict[str, Any]],
    *,
    subset: str,
    colorbar_label: str,
) -> None:
    subset_rows = [row for row in dist_rows if row["subset"] == subset]
    if not subset_rows:
        return
    bins = sorted({int(row["bin_idx"]) for row in subset_rows})
    matrix = np.full((len(CATEGORY_KEYS), len(bins)), np.nan, dtype=float)
    for row in subset_rows:
        i = CATEGORY_INDEX[row["label"]]
        j = int(row["bin_idx"])
        value = float(row["label_distribution"])
        matrix[i, j] = value if np.isfinite(value) else np.nan

    fig, ax = plt.subplots(figsize=(12, 7.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=np.nanmax(matrix))
    xticks = np.linspace(0, len(bins) - 1, 5)
    ax.set_xticks(xticks, ["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(np.arange(len(CATEGORY_KEYS)), _matrix_labels())
    ax.set_xlabel("normalized iteration in run")
    ax.set_ylabel("edit label")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=colorbar_label)
    fig.subplots_adjust(left=0.26, right=0.96, bottom=0.14)
    fig.savefig(path)
    plt.close(fig)


def plot_prevalence_bars(path: Path, effect_rows: list[dict[str, Any]], *, n_total: int) -> None:
    rows = sorted(effect_rows, key=lambda row: int(row["n_edits"]), reverse=True)
    labels = [row["display"] for row in rows]
    shares = [int(row["n_edits"]) / n_total for row in rows]
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(ypos, shares, color="#4c78a8")
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("share of all edits with label")
    ax.set_ylabel("edit label")
    ax.set_xlim(0.0, max(shares) * 1.18 if shares else 1.0)
    fig.subplots_adjust(left=0.32, right=0.96, bottom=0.12)
    fig.savefig(path)
    plt.close(fig)


def plot_label_count_distribution(path: Path, dist_rows: list[dict[str, Any]]) -> None:
    order = list(LABEL_COUNT_BUCKETS)
    values_by_bucket = {
        str(row["label_count_bucket"]): float(row["share_edits"]) for row in dist_rows
    }
    values = [values_by_bucket.get(bucket, 0.0) for bucket in order]
    colors = ["#bab0ab", "#4c78a8", "#2a9d8f", "#2a9d8f", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(np.arange(len(order)), values, color=colors, width=0.75)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    ax.set_xlabel("labels assigned to edit")
    ax.set_ylabel("share of edits")
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.16)
    fig.savefig(path)
    plt.close(fig)


def plot_multilabel_rate_bars(path: Path, multilabel_rows: list[dict[str, Any]]) -> None:
    sorted_rows = sorted(
        multilabel_rows,
        key=lambda row: float(row["multi_label_rate"])
        if np.isfinite(float(row["multi_label_rate"]))
        else float("-inf"),
        reverse=True,
    )
    labels = [row["display"] for row in sorted_rows]
    values = [float(row["multi_label_rate"]) for row in sorted_rows]
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(ypos, values, color="#4c78a8")
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("share of label occurrences in multi-label edits")
    ax.set_ylabel("edit label")
    fig.subplots_adjust(left=0.32, right=0.96, bottom=0.12)
    fig.savefig(path)
    plt.close(fig)


def plot_top_combinations(path: Path, combo_rows: list[dict[str, Any]]) -> None:
    labels = [row["display"] for row in combo_rows]
    values = [float(row["share_multilabel_edits"]) for row in combo_rows]
    ypos = np.arange(len(labels))
    fig_height = max(6.5, 0.52 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(ypos, values, color="#76b7b2")
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("share of multi-label edits")
    ax.set_ylabel("exact label combination")
    fig.subplots_adjust(left=0.46, right=0.97, bottom=0.10)
    fig.savefig(path)
    plt.close(fig)


def plot_group_metric_heatmap(
    path: Path,
    grouped_rows: list[dict[str, Any]],
    *,
    group_values: list[str],
    group_field: str,
    metric: str,
    xlabel: str,
) -> None:
    matrix = np.full((len(CATEGORY_KEYS), len(group_values)), np.nan, dtype=float)
    for row in grouped_rows:
        i = CATEGORY_INDEX[row["label"]]
        j = group_values.index(str(row["group"]))
        value = float(row[metric])
        matrix[i, j] = value if np.isfinite(value) else np.nan

    if metric == "odds_ratio":
        cmap = "RdBu"
        norm = _centered_norm(matrix, center=1.0)
        colorbar_label = "odds ratio for positive score delta"
    elif metric == "positive_rate_uplift":
        cmap = "RdBu"
        norm = _centered_norm(matrix, center=0.0)
        colorbar_label = "positive-rate uplift over baseline"
    else:
        raise ValueError(f"Unknown grouped metric: {metric}")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(
        np.arange(len(group_values)),
        [pretty_group_value(group_field, value) for value in group_values],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(np.arange(len(CATEGORY_KEYS)), _matrix_labels())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("edit label")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=colorbar_label)
    fig.subplots_adjust(left=0.32, right=0.96, bottom=0.18)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    from evo_replay.paper_style import apply as _apply_paper_style

    _apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "figure.titlesize": 22,
        }
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", type=Path, nargs="+", help="Dataset roots to scan.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory.")
    ap.add_argument("--n-stages", type=int, default=4, help="Stage bins over normalized iteration.")
    ap.add_argument(
        "--n-time-bins",
        type=int,
        default=40,
        help="Number of normalized-iteration bins for continuous label-distribution plots.",
    )
    ap.add_argument(
        "--include-backend",
        nargs="*",
        default=None,
        help="Restrict aggregation to these backends.",
    )
    args = ap.parse_args(argv)

    include_backends = set(args.include_backend) if args.include_backend else None
    rows, meta = load_rows(
        args.roots,
        n_stages=args.n_stages,
        include_backends=include_backends,
    )
    if not rows:
        sys.exit("No completed llm_edit_taxonomy outputs found.")

    args.out.mkdir(parents=True, exist_ok=True)
    effect_rows = label_effect_rows(rows)
    stage_rows = label_stage_rows(rows, args.n_stages)
    counts, conditional = cooccurrence_matrices(rows)
    phi, score_corr_rows = correlation_rows(rows)
    helpful_rows = helpfulness_rows(rows)
    label_count_rows = label_count_distribution_rows(rows)
    multilabel_rows = label_multilabel_rows(rows)
    combo_rows = top_combination_rows(rows, top_n=12)
    bsf_enrichment_rows = enrichment_rows(rows, subset="best_so_far")
    lineage_enrichment_rows = enrichment_rows(rows, subset="final_best_lineage")
    stage_share_rows = []
    domain_helpfulness_rows: list[dict[str, Any]] = []
    domain_bsf_enrichment_rows: list[dict[str, Any]] = []
    domain_lineage_enrichment_rows: list[dict[str, Any]] = []
    domain_label_count_rows: list[dict[str, Any]] = []
    all_stage_share_rows = label_share_rows(
        rows,
        group_field="stage",
        group_values=stage_names(args.n_stages),
        subset="all",
    )
    for subset, _ in SUCCESS_SUBSETS:
        stage_share_rows.extend(
            label_share_rows(
                rows,
                group_field="stage",
                group_values=stage_names(args.n_stages),
                subset=subset,
            )
        )
    backend_share_rows = []
    backend_groups = group_order(rows, "backend")
    backend_helpfulness_rows = helpfulness_rows_by_group(
        rows,
        group_field="backend",
        group_values=backend_groups,
    )
    all_backend_share_rows = label_share_rows(
        rows,
        group_field="backend",
        group_values=backend_groups,
        subset="all",
    )
    time_dist_all_rows = time_distribution_rows(rows, subset="all", n_bins=args.n_time_bins)
    time_dist_subset_rows: list[dict[str, Any]] = []
    for subset, _ in SUCCESS_SUBSETS:
        time_dist_subset_rows.extend(
            time_distribution_rows(rows, subset=subset, n_bins=args.n_time_bins)
        )
    for subset, _ in SUCCESS_SUBSETS:
        backend_share_rows.extend(
            label_share_rows(
                rows,
                group_field="backend",
                group_values=backend_groups,
                subset=subset,
            )
        )
    domain_stage_all_rows: list[dict[str, Any]] = []
    domain_stage_subset_rows: list[dict[str, Any]] = []
    domain_backend_all_rows: list[dict[str, Any]] = []
    domain_backend_subset_rows: list[dict[str, Any]] = []
    domain_time_all_rows: list[dict[str, Any]] = []
    domain_time_subset_rows: list[dict[str, Any]] = []
    for domain in ("ale", "math"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        if not domain_rows:
            continue
        domain_helpfulness = helpfulness_rows(domain_rows)
        domain_bsf_enrichment = enrichment_rows(domain_rows, subset="best_so_far")
        domain_lineage_enrichment = enrichment_rows(
            domain_rows,
            subset="final_best_lineage",
        )
        domain_label_count = label_count_distribution_rows(domain_rows)
        domain_helpfulness_rows.extend(
            [{**row, "domain": domain} for row in domain_helpfulness]
        )
        domain_bsf_enrichment_rows.extend(
            [{**row, "domain": domain} for row in domain_bsf_enrichment]
        )
        domain_lineage_enrichment_rows.extend(
            [{**row, "domain": domain} for row in domain_lineage_enrichment]
        )
        domain_label_count_rows.extend(
            [{**row, "domain": domain} for row in domain_label_count]
        )
        stage_all = label_share_rows(
            domain_rows,
            group_field="stage",
            group_values=stage_names(args.n_stages),
            subset="all",
        )
        backend_groups_domain = group_order(domain_rows, "backend")
        backend_all = label_share_rows(
            domain_rows,
            group_field="backend",
            group_values=backend_groups_domain,
            subset="all",
        )
        time_all = time_distribution_rows(
            domain_rows,
            subset="all",
            n_bins=args.n_time_bins,
        )
        stage_subset: list[dict[str, Any]] = []
        backend_subset: list[dict[str, Any]] = []
        time_subset: list[dict[str, Any]] = []
        for subset, _ in SUCCESS_SUBSETS:
            stage_subset.extend(
                label_share_rows(
                    domain_rows,
                    group_field="stage",
                    group_values=stage_names(args.n_stages),
                    subset=subset,
                )
            )
            backend_subset.extend(
                label_share_rows(
                    domain_rows,
                    group_field="backend",
                    group_values=backend_groups_domain,
                    subset=subset,
                )
            )
            time_subset.extend(
                time_distribution_rows(
                    domain_rows,
                    subset=subset,
                    n_bins=args.n_time_bins,
                )
            )
        domain_stage_all_rows.extend([{**row, "domain": domain} for row in stage_all])
        domain_stage_subset_rows.extend([{**row, "domain": domain} for row in stage_subset])
        domain_backend_all_rows.extend([{**row, "domain": domain} for row in backend_all])
        domain_backend_subset_rows.extend([{**row, "domain": domain} for row in backend_subset])
        domain_time_all_rows.extend([{**row, "domain": domain} for row in time_all])
        domain_time_subset_rows.extend([{**row, "domain": domain} for row in time_subset])

    summary = {
        **meta,
        "n_edits": len(rows),
        "n_labeled_edits": sum(1 for row in rows if row["labels"]),
        "n_unlabeled_edits": sum(1 for row in rows if not row["labels"]),
        "n_best_so_far_updates": sum(1 for row in rows if row["is_best_so_far"]),
        "n_final_best_lineage_edits": sum(1 for row in rows if row["is_final_best_lineage"]),
        "n_time_bins": args.n_time_bins,
        "stages": stage_names(args.n_stages),
    }
    (args.out / "taxonomy_aggregate_summary.json").write_text(json.dumps(summary, indent=2))

    _write_csv(
        args.out / "label_effects.csv",
        effect_rows,
        [
            "label",
            "display",
            "n_edits",
            "positive_rate",
            "mean_delta_norm",
            "median_delta_norm",
            "median_positive_delta_norm",
        ],
    )
    _write_csv(
        args.out / "label_count_distribution.csv",
        label_count_rows,
        ["label_count_bucket", "n_edits", "share_edits"],
    )
    _write_csv(
        args.out / "label_count_distribution_by_domain.csv",
        domain_label_count_rows,
        ["domain", "label_count_bucket", "n_edits", "share_edits"],
    )
    _write_csv(
        args.out / "label_multilabel_rate.csv",
        multilabel_rows,
        [
            "label",
            "display",
            "n_label_edits",
            "n_single_label_edits",
            "n_multi_label_edits",
            "single_label_rate",
            "multi_label_rate",
        ],
    )
    _write_csv(
        args.out / "top_label_combinations.csv",
        combo_rows,
        [
            "label_combo",
            "display",
            "n_labels",
            "n_edits",
            "share_labeled_edits",
            "share_multilabel_edits",
        ],
    )
    _write_csv(
        args.out / "label_helpfulness.csv",
        helpful_rows,
        [
            "label",
            "display",
            "n_edits",
            "baseline_positive_rate",
            "present_positive_rate",
            "positive_rate_uplift",
            "odds_ratio",
        ],
    )
    _write_csv(
        args.out / "label_helpfulness_by_domain.csv",
        domain_helpfulness_rows,
        [
            "domain",
            "label",
            "display",
            "n_edits",
            "baseline_positive_rate",
            "present_positive_rate",
            "positive_rate_uplift",
            "odds_ratio",
        ],
    )
    _write_csv(
        args.out / "label_helpfulness_by_backend.csv",
        backend_helpfulness_rows,
        [
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_edits",
            "baseline_positive_rate",
            "present_positive_rate",
            "positive_rate_uplift",
            "odds_ratio",
        ],
    )
    _write_csv(
        args.out / "label_enrichment_best_so_far.csv",
        bsf_enrichment_rows,
        [
            "subset",
            "subset_display",
            "label",
            "display",
            "n_all_edits",
            "n_subset_edits",
            "n_all_label",
            "n_subset_label",
            "share_all",
            "share_subset",
            "enrichment_ratio",
        ],
    )
    _write_csv(
        args.out / "label_enrichment_best_so_far_by_domain.csv",
        domain_bsf_enrichment_rows,
        [
            "domain",
            "subset",
            "subset_display",
            "label",
            "display",
            "n_all_edits",
            "n_subset_edits",
            "n_all_label",
            "n_subset_label",
            "share_all",
            "share_subset",
            "enrichment_ratio",
        ],
    )
    _write_csv(
        args.out / "label_enrichment_final_best_lineage.csv",
        lineage_enrichment_rows,
        [
            "subset",
            "subset_display",
            "label",
            "display",
            "n_all_edits",
            "n_subset_edits",
            "n_all_label",
            "n_subset_label",
            "share_all",
            "share_subset",
            "enrichment_ratio",
        ],
    )
    _write_csv(
        args.out / "label_enrichment_final_best_lineage_by_domain.csv",
        domain_lineage_enrichment_rows,
        [
            "domain",
            "subset",
            "subset_display",
            "label",
            "display",
            "n_all_edits",
            "n_subset_edits",
            "n_all_label",
            "n_subset_label",
            "share_all",
            "share_subset",
            "enrichment_ratio",
        ],
    )
    _write_csv(
        args.out / "label_stage_effects.csv",
        stage_rows,
        [
            "label",
            "display",
            "stage_idx",
            "stage",
            "n_edits",
            "positive_rate",
            "mean_delta_norm",
            "median_delta_norm",
        ],
    )
    _write_csv(
        args.out / "label_share_by_stage_all_edits.csv",
        all_stage_share_rows,
        [
            "subset",
            "subset_display",
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_subset_edits",
            "n_label_edits",
            "label_share",
        ],
    )
    _write_csv(
        args.out / "label_share_by_stage_subset.csv",
        stage_share_rows,
        [
            "subset",
            "subset_display",
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_subset_edits",
            "n_label_edits",
            "label_share",
        ],
    )
    _write_csv(
        args.out / "label_share_by_backend_all_edits.csv",
        all_backend_share_rows,
        [
            "subset",
            "subset_display",
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_subset_edits",
            "n_label_edits",
            "label_share",
        ],
    )
    _write_csv(
        args.out / "label_share_by_backend_subset.csv",
        backend_share_rows,
        [
            "subset",
            "subset_display",
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_subset_edits",
            "n_label_edits",
            "label_share",
        ],
    )
    _write_csv(
        args.out / "label_distribution_over_time_all_edits.csv",
        time_dist_all_rows,
        [
            "subset",
            "subset_display",
            "bin_idx",
            "x_left",
            "x_right",
            "x_mid",
            "label",
            "display",
            "n_edits",
            "n_label_occurrences",
            "n_label_occurrences_total",
            "label_distribution",
        ],
    )
    _write_csv(
        args.out / "label_distribution_over_time_subset.csv",
        time_dist_subset_rows,
        [
            "subset",
            "subset_display",
            "bin_idx",
            "x_left",
            "x_right",
            "x_mid",
            "label",
            "display",
            "n_edits",
            "n_label_occurrences",
            "n_label_occurrences_total",
            "label_distribution",
        ],
    )
    _write_csv(
        args.out / "label_share_by_stage_all_edits_by_domain.csv",
        domain_stage_all_rows,
        [
            "domain",
            "subset",
            "subset_display",
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_subset_edits",
            "n_label_edits",
            "label_share",
        ],
    )
    _write_csv(
        args.out / "label_share_by_stage_subset_by_domain.csv",
        domain_stage_subset_rows,
        [
            "domain",
            "subset",
            "subset_display",
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_subset_edits",
            "n_label_edits",
            "label_share",
        ],
    )
    _write_csv(
        args.out / "label_share_by_backend_all_edits_by_domain.csv",
        domain_backend_all_rows,
        [
            "domain",
            "subset",
            "subset_display",
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_subset_edits",
            "n_label_edits",
            "label_share",
        ],
    )
    _write_csv(
        args.out / "label_share_by_backend_subset_by_domain.csv",
        domain_backend_subset_rows,
        [
            "domain",
            "subset",
            "subset_display",
            "group_field",
            "group",
            "group_display",
            "label",
            "display",
            "n_subset_edits",
            "n_label_edits",
            "label_share",
        ],
    )
    _write_csv(
        args.out / "label_distribution_over_time_all_edits_by_domain.csv",
        domain_time_all_rows,
        [
            "domain",
            "subset",
            "subset_display",
            "bin_idx",
            "x_left",
            "x_right",
            "x_mid",
            "label",
            "display",
            "n_edits",
            "n_label_occurrences",
            "n_label_occurrences_total",
            "label_distribution",
        ],
    )
    _write_csv(
        args.out / "label_distribution_over_time_subset_by_domain.csv",
        domain_time_subset_rows,
        [
            "domain",
            "subset",
            "subset_display",
            "bin_idx",
            "x_left",
            "x_right",
            "x_mid",
            "label",
            "display",
            "n_edits",
            "n_label_occurrences",
            "n_label_occurrences_total",
            "label_distribution",
        ],
    )
    _write_matrix_csv(args.out / "cooccurrence_counts.csv", counts)
    _write_matrix_csv(args.out / "cooccurrence_conditional.csv", conditional)
    _write_matrix_csv(args.out / "label_phi_correlation.csv", phi)
    _write_csv(
        args.out / "label_score_correlation.csv",
        score_corr_rows,
        ["label", "display", "n_edits", "score_delta_correlation"],
    )

    plot_overall_effects(
        args.out / "edit_progress_overall_median_delta.pdf",
        effect_rows,
        metric="median_delta_norm",
    )
    plot_overall_effects(
        args.out / "edit_progress_overall_positive_rate.pdf",
        effect_rows,
        metric="positive_rate",
    )
    plot_stage_heatmap(
        args.out / "edit_progress_by_stage_median_delta.pdf",
        rows,
        args.n_stages,
        metric="median_delta_norm",
    )
    plot_stage_heatmap(
        args.out / "edit_progress_by_stage_positive_rate.pdf",
        rows,
        args.n_stages,
        metric="positive_rate",
    )
    plot_cooccurrence(
        args.out / "edit_cooccurrence_counts.pdf",
        counts,
        kind="counts",
    )
    plot_cooccurrence(
        args.out / "edit_cooccurrence_conditional.pdf",
        conditional,
        kind="conditional",
    )
    plot_correlation_heatmap(args.out / "edit_correlation_phi.pdf", phi)
    plot_correlation_bars(
        args.out / "edit_correlation_score_delta.pdf",
        score_corr_rows,
    )
    plot_helpfulness_bars(
        args.out / "edit_helpfulness_positive_rate_uplift.pdf",
        helpful_rows,
        metric="positive_rate_uplift",
    )
    plot_helpfulness_bars(
        args.out / "edit_helpfulness_odds_ratio.pdf",
        helpful_rows,
        metric="odds_ratio",
    )
    plot_group_metric_heatmap(
        args.out / "edit_helpfulness_odds_ratio_by_backend.pdf",
        backend_helpfulness_rows,
        group_values=backend_groups,
        group_field="backend",
        metric="odds_ratio",
        xlabel="framework",
    )
    plot_enrichment_bars(
        args.out / "edit_enrichment_best_so_far.pdf",
        bsf_enrichment_rows,
    )
    plot_enrichment_bars(
        args.out / "edit_enrichment_final_best_lineage.pdf",
        lineage_enrichment_rows,
    )
    plot_prevalence_bars(
        args.out / "edit_label_prevalence.pdf",
        effect_rows,
        n_total=len(rows),
    )
    plot_label_count_distribution(
        args.out / "edit_label_count_distribution.pdf",
        label_count_rows,
    )
    plot_multilabel_rate_bars(
        args.out / "edit_multilabel_rate_by_label.pdf",
        multilabel_rows,
    )
    plot_top_combinations(
        args.out / "top_edit_label_combinations.pdf",
        combo_rows,
    )
    plot_all_edit_composition(
        args.out / "all_edit_composition_by_stage.pdf",
        all_stage_share_rows,
        group_values=stage_names(args.n_stages),
        group_field="stage",
        xlabel="time in run",
    )
    plot_all_edit_composition(
        args.out / "all_edit_composition_by_backend.pdf",
        all_backend_share_rows,
        group_values=backend_groups,
        group_field="backend",
        xlabel="framework",
    )
    plot_time_distribution(
        args.out / "all_edit_distribution_over_time.pdf",
        time_dist_all_rows,
        subset="all",
        ylabel="share of label occurrences",
    )
    plot_time_heatmap(
        args.out / "all_edit_heatmap_over_time.pdf",
        time_dist_all_rows,
        subset="all",
        colorbar_label="share of label occurrences",
    )
    for domain in ("ale", "math"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        if not domain_rows:
            continue
        domain_effect_rows = label_effect_rows(domain_rows)
        domain_helpful_rows = helpfulness_rows(domain_rows)
        domain_bsf_rows = enrichment_rows(domain_rows, subset="best_so_far")
        domain_lineage_rows = enrichment_rows(domain_rows, subset="final_best_lineage")
        plot_prevalence_bars(
            args.out / f"edit_label_prevalence_{domain}.pdf",
            domain_effect_rows,
            n_total=len(domain_rows),
        )
        plot_label_count_distribution(
            args.out / f"edit_label_count_distribution_{domain}.pdf",
            label_count_distribution_rows(domain_rows),
        )
        plot_helpfulness_bars(
            args.out / f"edit_helpfulness_odds_ratio_{domain}.pdf",
            domain_helpful_rows,
            metric="odds_ratio",
        )
        plot_enrichment_bars(
            args.out / f"edit_enrichment_best_so_far_{domain}.pdf",
            domain_bsf_rows,
        )
        plot_enrichment_bars(
            args.out / f"edit_enrichment_final_best_lineage_{domain}.pdf",
            domain_lineage_rows,
        )
        domain_stage_all = label_share_rows(
            domain_rows,
            group_field="stage",
            group_values=stage_names(args.n_stages),
            subset="all",
        )
        domain_backend_groups = group_order(domain_rows, "backend")
        domain_backend_all = label_share_rows(
            domain_rows,
            group_field="backend",
            group_values=domain_backend_groups,
            subset="all",
        )
        plot_all_edit_composition(
            args.out / f"all_edit_composition_by_stage_{domain}.pdf",
            domain_stage_all,
            group_values=stage_names(args.n_stages),
            group_field="stage",
            xlabel="time in run",
        )
        plot_all_edit_composition(
            args.out / f"all_edit_composition_by_backend_{domain}.pdf",
            domain_backend_all,
            group_values=domain_backend_groups,
            group_field="backend",
            xlabel="framework",
        )
        domain_time_all = time_distribution_rows(
            domain_rows,
            subset="all",
            n_bins=args.n_time_bins,
        )
        plot_time_distribution(
            args.out / f"all_edit_distribution_over_time_{domain}.pdf",
            domain_time_all,
            subset="all",
            ylabel="share of label occurrences",
        )
        plot_time_heatmap(
            args.out / f"all_edit_heatmap_over_time_{domain}.pdf",
            domain_time_all,
            subset="all",
            colorbar_label="share of label occurrences",
        )
        domain_stage_subset: list[dict[str, Any]] = []
        domain_backend_subset: list[dict[str, Any]] = []
        domain_time_subset: list[dict[str, Any]] = []
        for subset, _ in SUCCESS_SUBSETS:
            domain_stage_subset.extend(
                label_share_rows(
                    domain_rows,
                    group_field="stage",
                    group_values=stage_names(args.n_stages),
                    subset=subset,
                )
            )
            domain_backend_subset.extend(
                label_share_rows(
                    domain_rows,
                    group_field="backend",
                    group_values=domain_backend_groups,
                    subset=subset,
                )
            )
            domain_time_subset.extend(
                time_distribution_rows(
                    domain_rows,
                    subset=subset,
                    n_bins=args.n_time_bins,
                )
            )
            plot_success_composition(
                args.out / f"successful_edit_composition_by_stage_{subset}_{domain}.pdf",
                domain_stage_subset,
                group_field="stage",
                group_values=stage_names(args.n_stages),
                xlabel="time in run",
                subset=subset,
            )
            plot_success_composition(
                args.out / f"successful_edit_composition_by_backend_{subset}_{domain}.pdf",
                domain_backend_subset,
                group_field="backend",
                group_values=domain_backend_groups,
                xlabel="framework",
                subset=subset,
            )
            plot_time_distribution(
                args.out / f"successful_edit_distribution_over_time_{subset}_{domain}.pdf",
                domain_time_subset,
                subset=subset,
                ylabel="share of label occurrences",
            )
            plot_time_heatmap(
                args.out / f"successful_edit_heatmap_over_time_{subset}_{domain}.pdf",
                domain_time_subset,
                subset=subset,
                colorbar_label="share of label occurrences",
            )
    for subset, _ in SUCCESS_SUBSETS:
        plot_success_composition(
            args.out / f"successful_edit_composition_by_stage_{subset}.pdf",
            stage_share_rows,
            group_field="stage",
            group_values=stage_names(args.n_stages),
            xlabel="time in run",
            subset=subset,
        )
        plot_success_composition(
            args.out / f"successful_edit_composition_by_backend_{subset}.pdf",
            backend_share_rows,
            group_field="backend",
            group_values=backend_groups,
            xlabel="framework",
            subset=subset,
        )
        plot_time_distribution(
            args.out / f"successful_edit_distribution_over_time_{subset}.pdf",
            time_dist_subset_rows,
            subset=subset,
            ylabel="share of label occurrences",
        )
        plot_time_heatmap(
            args.out / f"successful_edit_heatmap_over_time_{subset}.pdf",
            time_dist_subset_rows,
            subset=subset,
            colorbar_label="share of label occurrences",
        )

    print(
        f"completed runs: {summary['n_completed_runs']}  "
        f"edits: {summary['n_edits']}  "
        f"outputs: {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
