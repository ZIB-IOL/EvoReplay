"""Human-vs-judge agreement: sample, blind-review, score.

Workflow:
    # 1. Sample N edits stratified across judge labels.
    uv run python -m evo_replay.edit_taxonomy.agreement sample \\
        <classified_run_dir> [more...] --n 200 --out session.jsonl

    # 2. Annotate them in a single-keystroke terminal TUI.
    uv run python -m evo_replay.edit_taxonomy.agreement review session.jsonl

    # 3. Score agreement once you're done (or whenever).
    uv run python -m evo_replay.edit_taxonomy.agreement score session.jsonl

    # 4. Check progress at any time.
    uv run python -m evo_replay.edit_taxonomy.agreement show session.jsonl

`sample` writes TWO files: <out> (review file, judge labels HIDDEN to avoid
anchoring) and <out>.judge.jsonl (sealed sidecar, joined back at score time).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evo_replay.core.checkpoints import load_programs
from evo_replay.edit_taxonomy.judge import make_unified_diff
from evo_replay.edit_taxonomy.rubric import CATEGORIES, CATEGORY_NAMES, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Single-key bindings — one letter per category, no collisions.
# ---------------------------------------------------------------------------

LABEL_KEYS: Dict[str, str] = {
    "b": "bug_fix",
    "x": "external_dependency",
    "a": "architectural_change",
    "c": "composition",
    "l": "local_refinement",
    "p": "pruning",
    "r": "refactor",
    "e": "efficiency",
    "h": "hyperparameter_tuning",
}
KEY_FOR_LABEL = {v: k for k, v in LABEL_KEYS.items()}
assert set(LABEL_KEYS.values()) == set(CATEGORY_NAMES), \
    "agreement key bindings must cover every rubric category"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _judge_path_for(review_path: Path) -> Path:
    return review_path.with_suffix(review_path.suffix + ".judge.jsonl")


def _report_path_for(review_path: Path) -> Path:
    return review_path.with_suffix(review_path.suffix + ".report.md")


def _load_classified_run(run_dir: Path) -> Iterable[Dict[str, Any]]:
    """Yield rows from <run_dir>/analysis/llm_edit_taxonomy.jsonl, augmented
    with the materialised parent->child diff and run/relative-path metadata."""
    jsonl = run_dir / "analysis" / "llm_edit_taxonomy.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"No classification jsonl at {jsonl}")
    programs = load_programs(run_dir)
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cid, pid = rec["program_id"], rec["parent_id"]
        if cid not in programs or pid not in programs:
            continue
        parent = programs[pid]
        child = programs[cid]
        if not (parent.get("solution") and child.get("solution")):
            continue
        diff = make_unified_diff(parent["solution"], child["solution"], n=3)
        if not diff.strip():
            continue
        rec["__diff__"] = diff
        rec["__run_dir__"] = str(run_dir)
        rec["__run_short__"] = f"{run_dir.parent.name}/{run_dir.name}"
        yield rec


def _stratify(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    by_key: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        labels = tuple(sorted(r.get("labels") or []))
        by_key[labels or ("<none>",)].append(r)
    return by_key


def stratified_sample(
    rows: List[Dict[str, Any]], n: int, *, seed: int = 42
) -> List[Dict[str, Any]]:
    """Round-robin sample across strata so rare label-tuples are over-sampled.

    Stratum = sorted tuple of judge labels (so {refactor, efficiency} and
    {efficiency, refactor} share a stratum). Empty-label edits live in
    `('<none>',)`.
    """
    if n <= 0:
        return []
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    strata = _stratify(rows)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata.keys())
    out: List[Dict[str, Any]] = []
    while len(out) < n:
        progressed = False
        for k in keys:
            if not strata[k]:
                continue
            out.append(strata[k].pop())
            progressed = True
            if len(out) >= n:
                break
        if not progressed:
            break
    return out


def _review_id(run_short: str, child_id: str) -> str:
    # Compact key the user can recognise across files.
    bench = run_short.split("/", 1)[1].split("_")[0] if "/" in run_short else "?"
    return f"{bench}:{child_id[:8]}"


def cmd_sample(args: argparse.Namespace) -> None:
    rows: List[Dict[str, Any]] = []
    for rd in args.run_dirs:
        rd = rd.resolve()
        if not rd.exists():
            raise SystemExit(f"Missing run dir: {rd}")
        rows.extend(_load_classified_run(rd))
    if not rows:
        raise SystemExit("No edits found in any run.")
    print(f"pool: {len(rows)} edits across {len(args.run_dirs)} runs", file=sys.stderr)
    sample = stratified_sample(rows, args.n, seed=args.seed)
    print(
        f"sampled: {len(sample)} ({len(set(tuple(sorted(r.get('labels') or [])) for r in sample))} distinct strata)",
        file=sys.stderr,
    )

    review_path = args.out
    judge_path = _judge_path_for(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: Dict[str, int] = {}
    review_records, judge_records = [], []
    for r in sample:
        rid = _review_id(r["__run_short__"], r["program_id"])
        # Disambiguate collisions by appending a numeric suffix.
        if rid in seen_ids:
            seen_ids[rid] += 1
            rid = f"{rid}#{seen_ids[rid]}"
        else:
            seen_ids[rid] = 0
        review_records.append(
            {
                "review_id": rid,
                "run": r["__run_short__"],
                "child_id": r["program_id"],
                "parent_id": r["parent_id"],
                "iteration": r.get("iteration"),
                "language": r.get("language"),
                "score": r.get("score"),
                "parent_score": r.get("parent_score"),
                "score_delta": r.get("score_delta"),
                "diff": r["__diff__"],
                "judge_labels_visible": r.get("labels") if args.show_judge else None,
                "judge_rationale_visible": r.get("rationale") if args.show_judge else None,
                "human_labels": None,
                "human_confidence": None,
                "human_notes": "",
            }
        )
        judge_records.append(
            {
                "review_id": rid,
                "judge_labels": list(r.get("labels") or []),
                "judge_rationale": r.get("rationale", ""),
                "score_delta": r.get("score_delta"),
            }
        )

    _write_jsonl(review_path, review_records)
    _write_jsonl(judge_path, judge_records)
    print(f"wrote {review_path}  ({len(review_records)} rows)")
    print(f"wrote {judge_path}   (judge sidecar — keep sealed until scoring)")


def _write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(p)


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Review TUI
# ---------------------------------------------------------------------------


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def _colorize_diff(diff: str, *, color: bool) -> List[str]:
    out = []
    for line in diff.splitlines():
        if not color:
            out.append(line)
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.append(f"\033[32m{line}\033[0m")  # green
        elif line.startswith("-") and not line.startswith("---"):
            out.append(f"\033[31m{line}\033[0m")  # red
        elif line.startswith("@@"):
            out.append(f"\033[36m{line}\033[0m")  # cyan
        else:
            out.append(f"\033[2m{line}\033[0m")   # dim
    return out


def _getch() -> str:
    """Read one keystroke from stdin without echo. Unix only."""
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # If escape, peek for arrow-key sequences.
        if ch == "\x1b":
            tty.setcbreak(fd)
            try:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return f"\x1b[{ch3}"
                return ch + ch2
            except Exception:
                return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _clear() -> None:
    sys.stdout.write("\033[H\033[2J\033[3J")
    sys.stdout.flush()


@dataclass
class Reviewer:
    review_path: Path
    records: List[Dict[str, Any]] = field(default_factory=list)
    idx: int = 0
    scroll: int = 0
    color: bool = True
    status: str = ""
    show_judge: bool = False
    judge_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def load(self) -> None:
        self.records = _read_jsonl(self.review_path)
        judge_path = _judge_path_for(self.review_path)
        if judge_path.exists():
            self.judge_by_id = {
                r["review_id"]: r for r in _read_jsonl(judge_path)
            }
        # Resume at first un-labelled edit.
        for i, r in enumerate(self.records):
            if r.get("human_labels") is None:
                self.idx = i
                break
        else:
            self.idx = len(self.records)

    def save(self) -> None:
        _write_jsonl(self.review_path, self.records)

    @property
    def cur(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.idx < len(self.records):
            return self.records[self.idx]
        return None

    # ---- rendering ---------------------------------------------------------

    def render(self) -> None:
        _clear()
        cur = self.cur
        n = len(self.records)
        done = sum(1 for r in self.records if r.get("human_labels") is not None)
        if cur is None:
            self._render_done(done, n)
            return

        cols, rows_avail = shutil.get_terminal_size((120, 40))
        # Reserve lines: header (3) + selected (2) + keymap (4) + status (1) + footer (2) = 12
        diff_height = max(8, rows_avail - 13)

        labels = cur.get("human_labels") or []
        sd = cur.get("score_delta")
        sd_str = f"Δscore={sd:+.4g}" if isinstance(sd, (int, float)) else "Δscore=—"
        header = (
            f"[{self.idx + 1}/{n}  done={done}]  "
            f"{cur['review_id']}  child={cur['child_id'][:8]}  "
            f"{sd_str}  iter={cur.get('iteration')}  lang={cur.get('language')}"
        )
        sep = "─" * min(cols, 120)
        print(self._bold(header))
        print(sep)

        diff_lines = _colorize_diff(cur["diff"], color=self.color)
        total = len(diff_lines)
        self.scroll = max(0, min(self.scroll, max(0, total - diff_height)))
        view = diff_lines[self.scroll : self.scroll + diff_height]
        for line in view:
            # Truncate to terminal width; ANSI escapes don't count toward width but we just hard-cut.
            print(line[: cols + 16])
        # Pad if diff is shorter than view.
        for _ in range(diff_height - len(view)):
            print()
        if total > diff_height:
            print(self._dim(f"  -- diff lines {self.scroll + 1}-{self.scroll + len(view)} of {total}  (j/k scroll, J/K page)"))
        else:
            print()
        print(sep)

        sel_str = ", ".join(labels) if labels else self._dim("(none — press a letter to add)")
        print(f"selected: {sel_str}")
        note = cur.get("human_notes") or ""
        if note:
            print(self._dim(f"note: {note[:cols - 8]}"))
        if self.show_judge:
            jr = self.judge_by_id.get(cur["review_id"], {})
            jl = jr.get("judge_labels") or []
            jrt = jr.get("judge_rationale") or ""
            jl_str = ", ".join(jl) if jl else "(none)"
            print(self._dim(f"judge:    {jl_str}"))
            if jrt:
                wrapped = textwrap.shorten(jrt, width=cols - 11, placeholder="…")
                print(self._dim(f"  why: {wrapped}"))

        print()
        print(self._format_keymap(labels))
        print()
        judge_hint = "M hide-judge" if self.show_judge else "M show-judge"
        print(self._dim(
            f"  ENTER save & next   s skip   u back   N note   {judge_hint}   R rubric   ? help   q save & quit"
        ))
        if self.status:
            print(self._bold(self.status))

    def _format_keymap(self, selected: List[str]) -> str:
        cells = []
        for k, lab in LABEL_KEYS.items():
            mark = "●" if lab in selected else " "
            cells.append(f"{self._bold('['+k+']')} {mark} {lab}")
        # Three columns
        per = math.ceil(len(cells) / 3)
        cols = [cells[i*per:(i+1)*per] for i in range(3)]
        width = max(len(s) for c in cols for s in c) + 2
        # ANSI escapes inflate len; pad based on visible-length estimate.
        def vis_len(s: str) -> int:
            import re
            return len(re.sub(r"\x1b\[[0-9;]*m", "", s))
        out = []
        for i in range(per):
            row = []
            for col in cols:
                if i < len(col):
                    s = col[i]
                    pad = " " * (width - vis_len(s))
                    row.append(s + pad)
            out.append("  ".join(row))
        return "\n".join(out)

    def _render_done(self, done: int, n: int) -> None:
        print(self._bold(f"All {n} edits labelled.  ✓"))
        print()
        print("Next:")
        print(f"  uv run python -m evo_replay.edit_taxonomy.agreement score {self.review_path}")

    def _bold(self, s: str) -> str:
        return f"\033[1m{s}\033[0m" if self.color else s

    def _dim(self, s: str) -> str:
        return f"\033[2m{s}\033[0m" if self.color else s

    # ---- actions -----------------------------------------------------------

    def toggle(self, key: str) -> None:
        cur = self.cur
        if cur is None:
            return
        label = LABEL_KEYS[key]
        labels = list(cur.get("human_labels") or [])
        if label in labels:
            labels.remove(label)
            self.status = f"  − {label}"
        else:
            labels.append(label)
            self.status = f"  + {label}"
        cur["human_labels"] = labels
        self.save()

    def commit_and_advance(self) -> None:
        cur = self.cur
        if cur is None:
            return
        if cur.get("human_labels") is None:
            cur["human_labels"] = []
        self.save()
        self.idx += 1
        self.scroll = 0
        self.status = ""

    def skip(self) -> None:
        self.idx += 1
        self.scroll = 0
        self.status = "  (skipped)"

    def back(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.scroll = 0
            self.status = ""

    def enter_note(self) -> None:
        cur = self.cur
        if cur is None:
            return
        # Drop out of raw mode to use input().
        sys.stdout.write("\n note (enter to keep current; '-' to clear): ")
        sys.stdout.flush()
        try:
            line = sys.stdin.readline().rstrip("\n")
        except (KeyboardInterrupt, EOFError):
            line = ""
        if line == "-":
            cur["human_notes"] = ""
        elif line:
            cur["human_notes"] = line
        self.save()

    def help(self) -> None:
        _clear()
        print(self._bold("agreement review — help"))
        print()
        print("Toggle labels (any order, any number — multi-label is the point):")
        for k, lab in LABEL_KEYS.items():
            print(f"   {k}  {lab}")
        print()
        print("Navigation:")
        print("   ENTER     save & advance (an empty label set is valid — saves [])")
        print("   s         skip without saving (leave human_labels = null)")
        print("   u         back one")
        print("   N         add/edit a free-text note")
        print("   M         toggle showing the LLM judge's labels + rationale")
        print("   R         show the rubric — exactly what the LLM judge sees")
        print("   j / k     scroll diff one line  (also ↑ / ↓)")
        print("   J / K     scroll diff one page  (space also pages down)")
        print("   ?         this help")
        print("   q         save & quit (resumable next time)")
        print()
        print("Press any key to continue.")
        _getch()

    def _page(self, lines: List[str], header: str) -> None:
        cols, rows = shutil.get_terminal_size((120, 40))
        height = max(8, rows - 4)
        scroll = 0
        while True:
            _clear()
            print(self._bold(header))
            print("─" * min(cols, 120))
            view = lines[scroll : scroll + height]
            for ln in view:
                print(ln[: cols + 16])
            for _ in range(height - len(view)):
                print()
            print("─" * min(cols, 120))
            print(self._dim(
                f"  lines {scroll + 1}-{scroll + len(view)} of {len(lines)}  "
                f"(j/k line  J/K or space page  g/G top/bottom  q close)"
            ))
            ch = _getch()
            max_scroll = max(0, len(lines) - height)
            if ch in ("q", "\r", "\n", "\x1b", "R", "?"):
                return
            elif ch in ("j", "\x1b[B"):
                scroll = min(scroll + 1, max_scroll)
            elif ch in ("k", "\x1b[A"):
                scroll = max(0, scroll - 1)
            elif ch in ("J", " "):
                scroll = min(scroll + height - 2, max_scroll)
            elif ch == "K":
                scroll = max(0, scroll - (height - 2))
            elif ch == "g":
                scroll = 0
            elif ch == "G":
                scroll = max_scroll

    def show_rubric(self) -> None:
        """Display the system prompt verbatim — exactly what the LLM judge sees."""
        self._page(
            SYSTEM_PROMPT.splitlines(),
            header="What the LLM judge sees (system prompt)",
        )

    # ---- main loop ---------------------------------------------------------

    def run(self) -> None:
        while True:
            self.render()
            cur = self.cur
            if cur is None:
                # All done — exit cleanly.
                return
            ch = _getch()
            self.status = ""
            if ch in LABEL_KEYS:
                self.toggle(ch)
            elif ch in ("\r", "\n"):
                self.commit_and_advance()
            elif ch == "s":
                self.skip()
            elif ch == "u":
                self.back()
            elif ch == "N":
                self.enter_note()
            elif ch in ("j", "\x1b[B"):
                self.scroll += 1
            elif ch in ("k", "\x1b[A"):
                self.scroll = max(0, self.scroll - 1)
            elif ch in ("J", " "):
                self.scroll += 10
            elif ch == "K":
                self.scroll = max(0, self.scroll - 10)
            elif ch == "R":
                self.show_rubric()
            elif ch == "M":
                if not self.judge_by_id:
                    self.status = "  (no judge sidecar found — was the session sampled with --show-judge?)"
                else:
                    self.show_judge = not self.show_judge
                    self.status = "  judge labels SHOWN" if self.show_judge else "  judge labels hidden"
            elif ch == "?":
                self.help()
            elif ch == "q" or ch == "\x03":  # Ctrl-C
                self.save()
                _clear()
                done = sum(1 for r in self.records if r.get("human_labels") is not None)
                print(f"saved.  progress: {done}/{len(self.records)}")
                return


def cmd_review(args: argparse.Namespace) -> None:
    review_path = args.review.resolve()
    if not review_path.exists():
        raise SystemExit(f"Not found: {review_path}")
    rev = Reviewer(review_path=review_path, color=_supports_color())
    rev.load()
    if not rev.records:
        raise SystemExit("Empty review file.")
    # First-time-only orientation: show the rubric before any labelling.
    # Skipped on resume, since the marker file means you've already seen it.
    marker = review_path.with_suffix(review_path.suffix + ".seen-rubric")
    done_already = sum(1 for r in rev.records if r.get("human_labels") is not None)
    if not marker.exists() and done_already == 0 and not args.skip_rubric:
        _clear()
        print(rev._bold("First time on this session — opening the rubric (what the LLM judge sees)."))
        print(rev._dim("Press any key to view it. Press R any time during labelling to revisit."))
        _getch()
        rev.show_rubric()
        marker.touch()
    rev.run()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Two-rater binary Cohen's kappa from a 2x2 contingency.

    tp = both yes, fp = judge yes / human no, fn = judge no / human yes,
    tn = both no.
    """
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    p_obs = (tp + tn) / n
    judge_yes = (tp + fp) / n
    human_yes = (tp + fn) / n
    p_e = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
    if p_e >= 1.0:
        return 1.0 if p_obs >= 1.0 else 0.0
    return (p_obs - p_e) / (1 - p_e)


def score_pairs(
    judge_per_id: Dict[str, List[str]],
    human_per_id: Dict[str, List[str]],
) -> Dict[str, Any]:
    ids = sorted(set(judge_per_id) & set(human_per_id))
    n = len(ids)
    if n == 0:
        raise SystemExit("No labelled rows in common between human and judge.")
    per_cat: Dict[str, Dict[str, int]] = {
        k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in CATEGORY_NAMES
    }
    micro_tp = micro_fp = micro_fn = 0
    jaccards: List[float] = []
    hammings: List[float] = []
    exact = 0
    for rid in ids:
        gold = set(human_per_id[rid])  # human is the "gold" we score the judge against
        pred = set(judge_per_id[rid])
        for k in CATEGORY_NAMES:
            in_h = k in gold
            in_j = k in pred
            cell = per_cat[k]
            if in_h and in_j:
                cell["tp"] += 1; micro_tp += 1
            elif in_j and not in_h:
                cell["fp"] += 1; micro_fp += 1
            elif in_h and not in_j:
                cell["fn"] += 1; micro_fn += 1
            else:
                cell["tn"] += 1
        union = gold | pred
        inter = gold & pred
        jaccards.append(1.0 if not union else len(inter) / len(union))
        hammings.append(1 - len(gold ^ pred) / len(CATEGORY_NAMES))
        if gold == pred:
            exact += 1

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0

    cat_metrics: Dict[str, Dict[str, float]] = {}
    macro_f1_sum = 0.0
    macro_kappa_sum = 0.0
    macro_kappa_count = 0
    for k in CATEGORY_NAMES:
        c = per_cat[k]
        support_h = c["tp"] + c["fn"]   # number of human-positive
        support_j = c["tp"] + c["fp"]   # number of judge-positive
        prec = c["tp"] / support_j if support_j else 0.0
        rec  = c["tp"] / support_h if support_h else 0.0
        f1   = _f1(prec, rec)
        kappa = cohens_kappa(c["tp"], c["fp"], c["fn"], c["tn"])
        cat_metrics[k] = {
            "support_human": support_h,
            "support_judge": support_j,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
            "precision": prec, "recall": rec, "f1": f1,
            "kappa": kappa,
        }
        macro_f1_sum += f1
        if support_h > 0 or support_j > 0:
            macro_kappa_sum += kappa
            macro_kappa_count += 1

    return {
        "n": n,
        "exact_match_accuracy": exact / n,
        "mean_jaccard": sum(jaccards) / n,
        "mean_hamming_similarity": sum(hammings) / n,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": _f1(micro_p, micro_r),
        "macro_f1": macro_f1_sum / len(CATEGORY_NAMES),
        "macro_kappa": macro_kappa_sum / macro_kappa_count if macro_kappa_count else 0.0,
        "per_category": cat_metrics,
    }


def _kappa_label(k: float) -> str:
    if k < 0.0:    return "worse than chance"
    if k < 0.20:   return "slight"
    if k < 0.40:   return "fair"
    if k < 0.60:   return "moderate"
    if k < 0.80:   return "substantial"
    return "almost perfect"


def cmd_score(args: argparse.Namespace) -> None:
    review_path = args.review.resolve()
    judge_path = _judge_path_for(review_path)
    if not judge_path.exists():
        raise SystemExit(f"Missing judge sidecar: {judge_path}")

    review_rows = _read_jsonl(review_path)
    judge_rows = _read_jsonl(judge_path)

    human_per_id = {
        r["review_id"]: list(r.get("human_labels") or [])
        for r in review_rows
        if r.get("human_labels") is not None
    }
    judge_per_id = {r["review_id"]: list(r.get("judge_labels") or []) for r in judge_rows}
    judge_meta = {r["review_id"]: r for r in judge_rows}

    metrics = score_pairs(judge_per_id, human_per_id)

    # --- console summary --------------------------------------------------
    n = metrics["n"]
    print(f"labelled rows: {n}")
    print(f"  exact match           {metrics['exact_match_accuracy']:.2%}")
    print(f"  mean jaccard          {metrics['mean_jaccard']:.3f}")
    print(f"  mean hamming sim      {metrics['mean_hamming_similarity']:.3f}")
    print(f"  micro F1              {metrics['micro_f1']:.3f}  (P={metrics['micro_precision']:.3f}  R={metrics['micro_recall']:.3f})")
    print(f"  macro F1              {metrics['macro_f1']:.3f}")
    print(f"  macro Cohen's kappa   {metrics['macro_kappa']:.3f}  ({_kappa_label(metrics['macro_kappa'])})")
    print()
    print("per-category (sup_h / sup_j / P / R / F1 / kappa / interpretation):")
    for k in CATEGORY_NAMES:
        c = metrics["per_category"][k]
        print(
            f"  {k:<24} "
            f"sup_h={c['support_human']:>3}  sup_j={c['support_judge']:>3}  "
            f"P={c['precision']:.2f}  R={c['recall']:.2f}  F1={c['f1']:.2f}  "
            f"κ={c['kappa']:+.2f}  ({_kappa_label(c['kappa'])})"
        )

    # --- markdown disagreement report ------------------------------------
    report = _render_report(review_rows, judge_meta, human_per_id, metrics)
    out_md = _report_path_for(review_path)
    out_md.write_text(report)
    print()
    print(f"wrote {out_md}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.json_out}")


def _render_report(
    review_rows: List[Dict[str, Any]],
    judge_meta: Dict[str, Dict[str, Any]],
    human_per_id: Dict[str, List[str]],
    metrics: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# agreement report\n")
    lines.append(f"_n labelled = {metrics['n']}_\n")
    lines.append("## summary\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| exact match | {metrics['exact_match_accuracy']:.2%} |")
    lines.append(f"| mean jaccard | {metrics['mean_jaccard']:.3f} |")
    lines.append(f"| mean hamming sim | {metrics['mean_hamming_similarity']:.3f} |")
    lines.append(f"| micro F1 | {metrics['micro_f1']:.3f} |")
    lines.append(f"| macro F1 | {metrics['macro_f1']:.3f} |")
    lines.append(f"| macro Cohen's κ | {metrics['macro_kappa']:.3f} ({_kappa_label(metrics['macro_kappa'])}) |")
    lines.append("")
    lines.append("## per-category\n")
    lines.append("| category | sup_h | sup_j | P | R | F1 | κ | interpretation |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in CATEGORY_NAMES:
        c = metrics["per_category"][k]
        lines.append(
            f"| `{k}` | {c['support_human']} | {c['support_judge']} | "
            f"{c['precision']:.2f} | {c['recall']:.2f} | {c['f1']:.2f} | "
            f"{c['kappa']:+.2f} | {_kappa_label(c['kappa'])} |"
        )
    lines.append("")

    # Disagreements section.
    lines.append("## disagreements\n")
    lines.append("Edits where human and judge labels differ. The diff is included so you "
                 "can identify systematic patterns and feed them back into the rubric.\n")
    n_disagree = 0
    for r in review_rows:
        if r.get("human_labels") is None:
            continue
        rid = r["review_id"]
        h = sorted(human_per_id[rid])
        j = sorted(judge_meta.get(rid, {}).get("judge_labels") or [])
        if h == j:
            continue
        n_disagree += 1
        only_h = sorted(set(h) - set(j))
        only_j = sorted(set(j) - set(h))
        lines.append(f"### `{rid}`  ({r['run']}, child={r['child_id'][:8]})\n")
        sd = r.get("score_delta")
        if isinstance(sd, (int, float)):
            lines.append(f"- score Δ: **{sd:+.4g}**")
        lines.append(f"- human:    `{h}`")
        lines.append(f"- judge:    `{j}`")
        if only_h:
            lines.append(f"- human added:    `{only_h}`")
        if only_j:
            lines.append(f"- judge added:    `{only_j}`")
        rationale = judge_meta.get(rid, {}).get("judge_rationale", "")
        if rationale:
            lines.append(f"- judge rationale: _{rationale}_")
        note = r.get("human_notes") or ""
        if note:
            lines.append(f"- human note: _{note}_")
        lines.append("")
        lines.append("```diff")
        diff = r.get("diff", "")
        # Cap each disagreement diff so the report stays readable.
        if len(diff) > 4000:
            diff = diff[:4000] + "\n... [truncated]"
        lines.append(diff)
        lines.append("```")
        lines.append("")
    if n_disagree == 0:
        lines.append("_No disagreements._\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Show / progress
# ---------------------------------------------------------------------------


def cmd_show(args: argparse.Namespace) -> None:
    review_path = args.review.resolve()
    if not review_path.exists():
        raise SystemExit(f"Not found: {review_path}")
    rows = _read_jsonl(review_path)
    n = len(rows)
    done = sum(1 for r in rows if r.get("human_labels") is not None)
    skipped = sum(
        1 for r in rows
        if r.get("human_labels") is None and r.get("human_notes")
    )
    label_counts = Counter()
    for r in rows:
        if r.get("human_labels") is None:
            continue
        if not r["human_labels"]:
            label_counts["<none>"] += 1
        for l in r["human_labels"]:
            label_counts[l] += 1
    print(f"{review_path}")
    print(f"  total rows:    {n}")
    print(f"  labelled:      {done}  ({done / n:.0%})")
    print(f"  remaining:     {n - done}")
    if label_counts:
        print(f"  human-label distribution so far:")
        for k in CATEGORY_NAMES + ["<none>"]:
            if k in label_counts:
                print(f"    {k:<24} {label_counts[k]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample", help="sample N edits from classified runs (stratified)")
    sp.add_argument("run_dirs", nargs="+", type=Path)
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--show-judge", action="store_true",
                    help="include judge labels in the review file (non-blind)")
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser("review", help="annotate edits in a TUI")
    sp.add_argument("review", type=Path)
    sp.add_argument("--skip-rubric", action="store_true",
                    help="don't auto-open the rubric on first launch")
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("score", help="compute agreement metrics + report")
    sp.add_argument("review", type=Path)
    sp.add_argument("--json-out", type=Path, default=None)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("show", help="show progress on a review file")
    sp.add_argument("review", type=Path)
    sp.set_defaults(func=cmd_show)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
