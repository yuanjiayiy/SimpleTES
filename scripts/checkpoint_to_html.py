#!/usr/bin/env python3
"""Build self-contained HTML reports for one SimpleTES checkpoint (db_state_*).

Writes a run dashboard (index.html: score trend, failure trend, one row per chain)
plus one report per chain (chainNN.html: every candidate the chain produced, with
code, parent code, reflection, metrics and -- for rejected candidates -- the LLM
prompt/response), all derived from the files the engine already checkpoints:
nodes.json, policy.json, metadata.json, config.json and failure.json.

The page style mirrors nanodiscover's scripts/run_to_html.py (dashboard) and
scripts/eval_to_html.py (per-chain report); chain reports embed their rows as a
gzip+base64 payload decoded in the browser with DecompressionStream.

Usage:
    python scripts/checkpoint_to_html.py checkpoints/<date>/instance-<id>/db_state_<ts>
    python scripts/checkpoint_to_html.py <db_state_dir> --out <html_dir> --title "AC3 C=32 K=16 L=100"

Reference lines: every score chart draws the published results listed for the run's
task in TASK_REFERENCES (the task is the evaluator's directory name in config.json).
Add more lines for one run with --reference LABEL=SCORE, or skip the built-in ones
with --no-task-references. Scores are in combined_score units (what the evaluator
returns), so convert a paper's raw metric with the task's own scoring formula first.

    python scripts/checkpoint_to_html.py <db_state_dir> --reference "my baseline=1.00120"
"""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# Published results drawn as reference lines, keyed by task (the directory name of the
# run's evaluator). Each score is in combined_score units, converted from the paper's raw
# metric with that task's evaluator formula. Sources: SimpleTES paper (arXiv 2604.19341),
# Supplementary Table 17; "previous best" is the strongest prior result that table lists.
_AC3_BENCHMARK = 1.4556427953745406  # autocorrelation_third/evaluator.py: score = BENCHMARK / C3
TASK_REFERENCES: dict[str, list[tuple[str, float]]] = {
    # autocorrelation_first/evaluator.py: score = 1 / (1e-8 + C1); lower C1 is better.
    "autocorrelation_first": [
        ("SimpleTES paper (C1 1.503871)", 1.0 / (1e-8 + 1.503871)),
        ("previous best, Together AI (C1 1.502862)", 1.0 / (1e-8 + 1.502862)),
    ],
    # autocorrelation_second/evaluator.py: score = C2; higher is better.
    "autocorrelation_second": [
        ("SimpleTES paper (C2 0.962694)", 0.962694),
        ("previous best, Together AI (C2 0.961206)", 0.961206),
    ],
    "autocorrelation_third": [
        ("SimpleTES paper (C3 1.453675)", _AC3_BENCHMARK / 1.453675),
        ("previous best, Together AI (C3 1.454555)", _AC3_BENCHMARK / 1.454555),
    ],
}


def task_name(config: dict) -> str | None:
    """Task key for TASK_REFERENCES: the directory holding the run's evaluator."""

    evaluator = config.get("evaluator_path")
    return Path(evaluator).parent.name if evaluator else None


def esc(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def fmt(value: float | None, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def as_float(value) -> float | None:
    """Checkpoint JSON stores numbers as strings ('1.0013', '-inf'); return a finite float or None."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_dict(value) -> dict:
    """metrics/parent_ids are stored as Python reprs (e.g. "{'c3': 1.45, 'combined_score': -inf}")."""

    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return ast.literal_eval(str(value).replace("-inf", "None").replace("inf", "None").replace("nan", "None"))
    except (ValueError, SyntaxError):
        return {"raw": str(value)}


def as_list(value) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        return list(ast.literal_eval(str(value)))
    except (ValueError, SyntaxError):
        return []


def parse_time(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


@dataclass
class ChainSummary:
    chain: int
    steps: int = 0
    budget: int = 0
    candidates: int = 0
    valid: int = 0
    errors: int = 0
    rejects: int = 0
    gen_failures: int = 0
    improved: int = 0
    labeled: int = 0
    best: float | None = None
    best_c3: float | None = None
    last_gain_step: int | None = None
    rows: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_checkpoint(db_state: Path) -> dict:
    def read(name: str, default):
        path = db_state / name
        if path.exists():
            return json.loads(path.read_text())
        # Runs launched with --gzip write nodes.json.gz instead of nodes.json.
        gz_path = db_state / f"{name}.gz"
        if gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        return default

    policy = read("policy.json", {}).get("state", {})
    return {
        "nodes": read("nodes.json", []),
        "failures": read("failure.json", []),
        "policy": policy,
        "metadata": read("metadata.json", {}),
        "config": read("config.json", {}),
    }


def build_chains(ckpt: dict) -> tuple[dict[int, ChainSummary], dict]:
    nodes = ckpt["nodes"]
    policy = ckpt["policy"]
    by_id = {n["id"]: n for n in nodes}
    root_id = policy.get("root_id") or next((n["id"] for n in nodes if not as_list(n.get("parent_ids"))), None)
    root = by_id.get(root_id, {})
    seed_score = as_float(root.get("score"))

    history = {int(c): set(ids) for c, ids in policy.get("chain_history", {}).items()}
    budgets = {int(c): int(v) for c, v in policy.get("prompt_budget", {}).items()}
    prompt_counts = {int(c): int(v) for c, v in policy.get("chain_prompt_count", {}).items()}

    # Batch (gen_id) -> step number within its chain, 1-based, in dispatch order.
    gen_ids_by_chain: dict[int, set[int]] = defaultdict(set)
    for record in [*nodes, *ckpt["failures"]]:
        chain = record.get("chain_idx")
        gen = record.get("gen_id")
        if chain not in (None, "") and gen not in (None, ""):
            gen_ids_by_chain[int(chain)].add(int(gen))
    step_of = {
        (chain, gen): step
        for chain, gens in gen_ids_by_chain.items()
        for step, gen in enumerate(sorted(gens), start=1)
    }

    chains: dict[int, ChainSummary] = {}
    num_chains = int(policy.get("num_chains") or (max(gen_ids_by_chain) + 1 if gen_ids_by_chain else 0))
    for c in range(num_chains):
        chains[c] = ChainSummary(chain=c, steps=prompt_counts.get(c, 0), budget=budgets.get(c, 0))

    def base_row(record: dict, chain: int) -> dict:
        gen = int(record["gen_id"]) if record.get("gen_id") not in (None, "") else None
        parents = as_list(record.get("parent_ids"))
        return {
            "id": record.get("id"),
            "gen_id": gen,
            "step": step_of.get((chain, gen)),
            "parent_id": parents[0] if parents else None,
            "construction_id": record.get("shared_construction_id"),
            "created_at": record.get("created_at"),
        }

    for n in nodes:
        if n.get("chain_idx") in (None, ""):
            continue
        chain = int(n["chain_idx"])
        metrics = as_dict(n.get("metrics"))
        score = as_float(n.get("score"))
        row = base_row(n, chain)
        error = metrics.get("error")
        row.update(
            kind="node",
            score=score,
            c3=as_float(metrics.get("c3")),
            eval_time=as_float(metrics.get("eval_time")),
            status="committed" if n["id"] in history.get(chain, set()) else ("valid" if score is not None else "error"),
            msg=str(error) if error else "",
            metrics=json.dumps(metrics, indent=2, default=str),
            code=n.get("code", ""),
            reflection=n.get("reflection") or "",
        )
        chains[chain].rows.append(row)

    for f in ckpt["failures"]:
        if f.get("chain_idx") in (None, ""):
            continue
        chain = int(f["chain_idx"])
        row = base_row(f, chain)
        is_eval = f.get("type") == "evaluation"
        metrics = as_dict(f.get("metrics"))
        row.update(
            kind="rejected" if is_eval else "gen_failure",
            score=None,
            c3=as_float(metrics.get("c3")),
            eval_time=as_float(metrics.get("eval_time")),
            status="rejected" if is_eval else "gen_failure",
            msg=str(metrics.get("error") or f.get("reason") or ""),
            metrics=json.dumps(metrics, indent=2, default=str) if metrics else "",
            code=f.get("code", ""),
            reflection="",
            llm_input=f.get("llm_input", ""),
            llm_output=f.get("llm_output", ""),
        )
        chains[chain].rows.append(row)

    parent_score = {n["id"]: as_float(n.get("score")) for n in nodes}
    for summary in chains.values():
        summary.rows.sort(key=lambda r: (r["step"] or 0, r["created_at"] or ""))
        running_best = seed_score
        for i, row in enumerate(summary.rows, start=1):
            row["i"] = i
            summary.candidates += 1
            if row["kind"] == "rejected":
                summary.rejects += 1
            elif row["kind"] == "gen_failure":
                summary.gen_failures += 1
            elif row["score"] is None:
                summary.errors += 1
            else:
                summary.valid += 1
            parent = parent_score.get(row["parent_id"])
            if row["score"] is not None and parent is not None:
                summary.labeled += 1
                delta = row["score"] - parent
                row["improvement_label"] = "improved" if delta > 1e-12 else ("regressed" if delta < -1e-12 else "same")
                summary.improved += row["improvement_label"] == "improved"
            else:
                row["improvement_label"] = None
            if row["score"] is not None and (summary.best is None or row["score"] > summary.best):
                summary.best, summary.best_c3 = row["score"], row["c3"]
            if row["score"] is not None and running_best is not None and row["score"] > running_best + 1e-12:
                running_best = row["score"]
                summary.last_gain_step = row["step"]

    run_info = {
        "root_id": root_id,
        "root_code": root.get("code", ""),
        "seed_score": seed_score,
        "seed_c3": as_float(as_dict(root.get("metrics")).get("c3")),
        "by_id_code": {n["id"]: n.get("code", "") for n in nodes if n["id"] in set().union(*history.values())} if history else {},
    }
    return chains, run_info


# --------------------------------------------------------------------------------------
# Shared CSS (palette and components from nanodiscover's run_to_html.py / eval_to_html.py)
# --------------------------------------------------------------------------------------

BASE_CSS = r"""
:root {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --muted:          #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --series-blue:    #2a78d6;
  --series-green:   #1c8a4b;
  --series-purple:  #7c3aed;
  --series-orange:  #c2670c;
  --series-teal:    #1baf7a;
  --seq-550:        #1c5cab;
  --good:           #0ca30c;
  --critical:       #d03b3b;
  --warn:           #b8860b;
  --row-hover:      rgba(11,11,11,0.035);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --series-blue:    #3987e5;
    --series-green:   #35c07a;
    --series-purple:  #a78bfa;
    --series-orange:  #e69138;
    --series-teal:    #199e70;
    --seq-550:        #7fb2ef;
    --good:           #0ca30c;
    --critical:       #e66767;
    --warn:           #d9a441;
    --row-hover:      rgba(255,255,255,0.04);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --page:           #0d0d0d;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --muted:          #898781;
  --gridline:       #2c2c2a;
  --baseline:       #383835;
  --border:         rgba(255,255,255,0.10);
  --series-blue:    #3987e5;
  --series-green:   #35c07a;
  --series-purple:  #a78bfa;
  --series-orange:  #e69138;
  --series-teal:    #199e70;
  --seq-550:        #7fb2ef;
  --good:           #0ca30c;
  --critical:       #e66767;
  --warn:           #d9a441;
  --row-hover:      rgba(255,255,255,0.04);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px;
}
.wrap { max-width: 1400px; margin: 0 auto; padding: 20px 20px 60px; }
h1 { font-size: 18px; margin: 0 0 4px; }
.subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }
.subtitle a { color: var(--series-blue); text-decoration: none; font-weight: 600; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.tile {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  min-width: 130px;
}
.tile .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
.tile .value { font-size: 22px; font-variant-numeric: tabular-nums; margin-top: 2px; }
.tile .value.good { color: var(--good); }
.chart-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
.chart-card .title { font-size: 13px; color: var(--text-secondary); margin-bottom: 10px; }
.chart-svg { display: block; width: 100%; height: 150px; }
.chart-line-reward { stroke: var(--series-blue); stroke-width: 2; }
.chart-line-archive { stroke: var(--series-green); stroke-width: 2; }
.chart-line-score { stroke: var(--series-purple); stroke-width: 2; }
.chart-line-teal { stroke: var(--series-teal); stroke-width: 2; }
.chart-dot { fill: var(--surface-1); stroke: currentColor; stroke-width: 1.5; }
.chart-baseline { stroke: var(--baseline); stroke-width: 1; }
.chart-axis-text { fill: var(--muted); font-size: 10px; }
.chart-reference { stroke-width: 1.5; stroke-dasharray: 6 4; }
.chart-reference-text { font-size: 10px; font-weight: 600; }
.ref-0 { stroke: var(--series-orange); fill: var(--series-orange); }
.ref-1 { stroke: var(--text-secondary); fill: var(--text-secondary); }
.ref-2 { stroke: var(--series-teal); fill: var(--series-teal); }
.ref-3 { stroke: var(--warn); fill: var(--warn); }
.chart-marker { fill: var(--good); }
.chart-empty { color: var(--muted); font-size: 13px; padding: 50px 0; text-align: center; }
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
thead th {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid var(--gridline);
  color: var(--text-secondary);
  font-weight: 600;
  font-size: 12px;
  white-space: nowrap;
}
tbody tr { border-bottom: 1px solid var(--gridline); }
tbody tr:hover { background: var(--row-hover); }
td { padding: 7px 10px; vertical-align: middle; font-variant-numeric: tabular-nums; }
td.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
.status { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 600; }
.status.good { color: var(--good); }
.status.warn { color: var(--warn); }
.status.critical { color: var(--critical); }
.status.muted { color: var(--muted); }
.status.info { color: var(--series-blue); }
.status .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
a.report-link { color: var(--series-blue); text-decoration: none; font-weight: 600; }
a.report-link:hover { text-decoration: underline; }
footer { color: var(--muted); font-size: 12px; margin-top: 16px; }
"""


def render_line_chart_svg(
    points: list[tuple[float, float]],
    *,
    width: int = 1000,
    height: int = 150,
    pad: int = 26,
    css_class: str = "chart-line-reward",
    x_label: str = "step",
    digits: int = 4,
    dots: bool = True,
    references: list[tuple[float, str]] | None = None,
    markers: list[tuple[float, float]] | None = None,
    marker_note: str = "",
) -> str:
    """Render a minimal inline-SVG line chart (same shape as nanodiscover's run dashboard).

    ``references`` are optional (y, label) pairs, each drawn as a dashed horizontal
    line; the y-range is widened to include them. ``markers`` are extra highlighted points drawn
    on the same scale as the line.
    """

    if not points:
        return '<div class="chart-empty">No data yet</div>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    references = references or []
    # A reference within one data-span of the curve widens the y-range; one farther
    # away would flatten the curve, so it is pinned to the chart edge instead.
    span = (max(ys) - min(ys)) or abs(max(ys)) * 1e-6 or 1e-6
    near = [y for y, _ in references if min(ys) - span <= y <= max(ys) + span]
    ys.extend(near)
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    if x_hi == x_lo:
        x_hi = x_lo + 1
    if y_hi == y_lo:
        y_hi += max(abs(y_hi), 1e-6) * 1e-6

    def sx(x: float) -> float:
        return pad + (width - 2 * pad) * (x - x_lo) / (x_hi - x_lo)

    def sy(y: float) -> float:
        return height - pad - (height - 2 * pad) * (y - y_lo) / (y_hi - y_lo)

    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
    dot_svg = (
        "".join(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" class="chart-dot">'
            f"<title>{x_label} {x:g}: {y:.{digits}f}</title></circle>"
            for x, y in points
        )
        if dots
        else ""
    )
    marker_svg = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" class="chart-marker">'
        f"<title>{x_label} {x:g}: {y:.{digits}f}{marker_note}</title></circle>"
        for x, y in (markers or [])
    )
    ref_svg = ""
    for idx, (ref_y, ref_label) in enumerate(references):
        cls = f"ref-{idx % 4}"
        # Alternate label sides so nearby lines don't print over each other.
        anchor, label_x = ("end", width - pad) if idx % 2 == 0 else ("start", pad + 90)
        if ref_y < y_lo:
            line_y, label_y, note = height - pad, height - pad - 4, "↓ below chart: "
        elif ref_y > y_hi:
            line_y, label_y, note = pad, pad + 12, "↑ above chart: "
        else:
            line_y, label_y, note = sy(ref_y), sy(ref_y) - 4, ""
        ref_svg += (
            f'<line x1="{pad}" y1="{line_y:.1f}" x2="{width - pad}" y2="{line_y:.1f}" class="chart-reference {cls}">'
            f"<title>{esc(ref_label)}: {ref_y:.{digits}f}</title></line>"
            f'<text class="chart-reference-text {cls}" x="{label_x}" y="{label_y:.1f}" text-anchor="{anchor}">'
            f"{note}{esc(ref_label)} {ref_y:.{digits}f}</text>"
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" class="chart-svg">'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="chart-baseline"/>'
        f"{ref_svg}"
        f'<path d="{path}" class="{css_class}" fill="none"/>'
        f"{dot_svg}{marker_svg}"
        f'<text class="chart-axis-text" x="{pad}" y="14">{y_hi:.{digits}f}</text>'
        f'<text class="chart-axis-text" x="{pad}" y="{height - 6}">{y_lo:.{digits}f}</text>'
        f'<text class="chart-axis-text" x="{width - pad - 70}" y="{height - 6}">{x_label} {x_hi:g}</text>'
        f"</svg>"
    )


# --------------------------------------------------------------------------------------
# Run dashboard (index.html)
# --------------------------------------------------------------------------------------

DASHBOARD_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__CSS__
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; margin-bottom: 20px; }
.config { color: var(--text-secondary); font-size: 12px; margin: -12px 0 20px; display: flex; flex-wrap: wrap; gap: 6px 18px; }
.config b { color: var(--text-primary); font-weight: 600; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="subtitle">__SUBTITLE__</p>
  <div class="config">__CONFIG__</div>

  <div class="tiles">__TILES__</div>

  <div class="charts">
    <div class="chart-card">
      <div class="title">Best score so far, by evaluation</div>
      __BEST_CHART__
    </div>
    <div class="chart-card">
      <div class="title">Mean of per-chain best score, by step</div>
      __CHAIN_BEST_CHART__
    </div>
    <div class="chart-card">
      <div class="title">Valid candidates per step (% across all chains)</div>
      __VALID_CHART__
    </div>
    <div class="chart-card">
      <div class="title">Chains that set a new chain best at this step</div>
      __IMPROVING_CHART__
    </div>
  </div>

  <div class="table-scroll">
  <table>
    <thead>
      <tr>
        <th>Chain</th>
        <th>Status</th>
        <th>Steps</th>
        <th>Candidates</th>
        <th title="Candidates that ran and returned a finite combined_score.">Valid %</th>
        <th title="Valid candidates whose score beat their parent's.">Improved %</th>
        <th>Best score</th>
        <th>Best C3</th>
        <th title="Best score minus the seed's score.">Gain over seed</th>
        <th title="Step of the last batch that raised this chain's best score.">Last gain</th>
        <th title="Program crashed or timed out (score -inf).">Errors</th>
        <th title="Evaluation results the engine rejected, plus LLM outputs with no EVOLVE-BLOCK.">Rejected</th>
        <th>Report</th>
      </tr>
    </thead>
    <tbody>
__ROWS__
    </tbody>
  </table>
  </div>
  <footer>Generated by scripts/checkpoint_to_html.py from __DB_STATE__</footer>
</div>
</body>
</html>
"""


def render_dashboard(ckpt: dict, chains: dict[int, ChainSummary], run_info: dict, title: str, source: str) -> str:
    nodes = ckpt["nodes"]
    meta = ckpt["metadata"]
    config = ckpt["config"]
    policy = ckpt["policy"]
    seed = run_info["seed_score"]

    ordered = sorted(nodes, key=lambda n: n.get("created_at") or "")
    best_points: list[tuple[float, float]] = []
    running = None
    for idx, n in enumerate(ordered, start=1):
        s = as_float(n.get("score"))
        if s is not None and (running is None or s > running):
            running = s
            best_points.append((idx, s))
    if best_points and best_points[-1][0] != len(ordered):
        best_points.append((len(ordered), best_points[-1][1]))

    max_step = max((r["step"] or 0 for c in chains.values() for r in c.rows), default=0)
    valid_by_step = defaultdict(lambda: [0, 0])
    chain_best_at = {c: [None] * (max_step + 1) for c in chains}
    for c, summary in chains.items():
        best = seed
        by_step: dict[int, float] = {}
        for r in summary.rows:
            if r["step"] is None:
                continue
            valid_by_step[r["step"]][1] += 1
            if r["score"] is not None:
                valid_by_step[r["step"]][0] += 1
                by_step[r["step"]] = max(by_step.get(r["step"], -math.inf), r["score"])
        for step in range(1, max_step + 1):
            if step in by_step and (best is None or by_step[step] > best):
                best = by_step[step]
            chain_best_at[c][step] = best
    mean_best = [
        (step, sum(chain_best_at[c][step] for c in chains) / len(chains))
        for step in range(1, max_step + 1)
        if all(chain_best_at[c][step] is not None for c in chains)
    ]
    valid_pct = [(step, 100.0 * v / t) for step, (v, t) in sorted(valid_by_step.items()) if t]
    new_best_counts = [
        (
            step,
            float(
                sum(
                    1
                    for c in chains
                    if chain_best_at[c][step] is not None
                    and (chain_best_at[c][step - 1] if step > 1 else seed) is not None
                    and chain_best_at[c][step] > (chain_best_at[c][step - 1] if step > 1 else seed) + 1e-12
                )
            ),
        )
        for step in range(1, max_step + 1)
    ]

    total = sum(c.candidates for c in chains.values())
    valid = sum(c.valid for c in chains.values())
    rejected = sum(c.rejects + c.gen_failures for c in chains.values())
    best = as_float(meta.get("best_score"))
    best_node = next((n for n in nodes if n["id"] == meta.get("best_node_id")), None)
    best_c3 = as_float(as_dict(best_node.get("metrics")).get("c3")) if best_node else None
    times = [t for t in (parse_time(n.get("created_at")) for n in nodes) if t]
    elapsed = (max(times) - min(times)).total_seconds() / 3600 if times else None

    tiles = [
        ("Candidates", f"{total:,}", False),
        ("Valid rate", f"{100 * valid / total:.1f}%" if total else "–", True),
        ("Best score", fmt(best, 7), False),
        ("Best C3", fmt(best_c3, 7), False),
        ("Gain over seed", f"{best - seed:+.2e}" if best is not None and seed is not None else "–", True),
        ("Seed score", fmt(seed, 7), False),
        ("Rejected", f"{rejected:,}", False),
        ("Wall time", f"{elapsed:.1f} h" if elapsed is not None else "–", False),
    ]
    if best is not None:
        for offset, (ref_score, ref_label) in enumerate(run_info.get("references", [])):
            tiles.insert(5 + offset, (f"Best − {ref_label}", f"{best - ref_score:+.2e}", best >= ref_score))
    tiles_html = "".join(
        f'<div class="tile"><div class="label">{esc(label)}</div>'
        f'<div class="value{" good" if good else ""}">{esc(value)}</div></div>'
        for label, value, good in tiles
    )

    config_items = [
        ("model", config.get("model")),
        ("selector", config.get("selector")),
        ("C", policy.get("num_chains")),
        ("K", policy.get("k")),
        ("L", max(policy.get("prompt_budget", {"0": 0}).values(), default=None)),
        ("budget", config.get("max_generations")),
        ("gen / eval concurrency", f"{config.get('gen_concurrency')} / {config.get('eval_concurrency')}"),
        ("eval timeout", f"{config.get('eval_timeout')}s"),
        ("instance", meta.get("instance_id")),
    ]
    config_html = "".join(f"<span>{esc(k)} <b>{esc(v)}</b></span>" for k, v in config_items if v is not None)

    rows = []
    for c, s in sorted(chains.items()):
        if s.budget and s.steps >= s.budget:
            status = '<span class="status good"><span class="dot"></span>complete</span>'
        elif s.steps:
            status = f'<span class="status warn"><span class="dot"></span>stopped at {s.steps}/{s.budget}</span>'
        else:
            status = '<span class="status muted"><span class="dot"></span>pending</span>'
        gain = s.best - seed if s.best is not None and seed is not None else None
        valid_cell = f"{100 * s.valid / s.candidates:.1f}%" if s.candidates else "–"
        improved_cell = f"{100 * s.improved / s.labeled:.1f}%" if s.labeled else "–"
        rows.append(
            "      <tr>"
            f'<td class="mono">{c}</td>'
            f"<td>{status}</td>"
            f"<td>{s.steps} / {s.budget}</td>"
            f"<td>{s.candidates:,}</td>"
            f"<td>{valid_cell}</td>"
            f"<td>{improved_cell}</td>"
            f"<td>{fmt(s.best, 7)}</td>"
            f"<td>{fmt(s.best_c3, 7)}</td>"
            f"<td>{'–' if gain is None else f'{gain:+.2e}'}</td>"
            f"<td>{s.last_gain_step if s.last_gain_step is not None else '–'}</td>"
            f"<td>{s.errors:,}</td>"
            f"<td>{s.rejects + s.gen_failures:,}</td>"
            f'<td><a class="report-link" href="chain{c:02d}.html">View →</a></td>'
            "</tr>"
        )

    subtitle = (
        f"{len(chains)} chains &middot; {total:,} candidates &middot; seed {fmt(seed, 7)} → best {fmt(best, 7)}"
    )
    return (
        DASHBOARD_TEMPLATE.replace("__CSS__", BASE_CSS)
        .replace("__TITLE__", esc(title))
        .replace("__SUBTITLE__", subtitle)
        .replace("__CONFIG__", config_html)
        .replace("__TILES__", tiles_html)
        .replace(
            "__BEST_CHART__",
            render_line_chart_svg(
                best_points, css_class="chart-line-reward", x_label="eval", digits=7,
                dots=len(best_points) <= 60, references=run_info.get("references"),
            ),
        )
        .replace(
            "__CHAIN_BEST_CHART__",
            render_line_chart_svg(mean_best, css_class="chart-line-score", digits=7, dots=False, references=run_info.get("references")),
        )
        .replace(
            "__VALID_CHART__",
            render_line_chart_svg(valid_pct, css_class="chart-line-archive", digits=1, dots=False),
        )
        .replace(
            "__IMPROVING_CHART__",
            render_line_chart_svg(new_best_counts, css_class="chart-line-teal", digits=0, dots=False),
        )
        .replace("__ROWS__", "\n".join(rows))
        .replace("__DB_STATE__", esc(source))
    )


# --------------------------------------------------------------------------------------
# Per-chain report (chainNN.html)
# --------------------------------------------------------------------------------------

CHAIN_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__CSS__
#hist { display: block; width: 100%; height: 120px; }
.hist-bar { fill: var(--series-blue); }
.hist-bar:hover { fill: var(--seq-550); }
.hist-axis-text { fill: var(--muted); font-size: 10px; }
.chart-card { margin-bottom: 20px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; margin-bottom: 20px; }
.charts .chart-card { margin-bottom: 0; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
.controls input, .controls select {
  background: var(--surface-1);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 13px;
  font-family: inherit;
}
.controls label { color: var(--text-secondary); font-size: 12px; }
.controls .group { display: flex; align-items: center; gap: 6px; }
#count-label { color: var(--muted); font-size: 12px; margin-left: auto; }
thead th { cursor: pointer; user-select: none; }
thead th:hover { color: var(--text-primary); }
thead th.sorted::after { content: " " attr(data-dir); color: var(--muted); }
tbody tr.row { cursor: pointer; }
tbody tr.row.expanded { background: var(--row-hover); }
tbody tr.row.committed td:first-child { box-shadow: inset 3px 0 0 var(--good); }
td { vertical-align: top; }
td.msg { color: var(--text-secondary); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
td.mono { color: var(--muted); }
td.score { white-space: nowrap; }
.reward-bar-bg { background: var(--gridline); border-radius: 3px; height: 6px; width: 80px; display: inline-block; vertical-align: middle; margin-right: 6px; overflow: hidden; }
.reward-bar-fill { background: var(--series-blue); height: 100%; border-radius: 3px; }
tr.detail-row td { background: var(--page); padding: 0; }
.detail { padding: 16px 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.detail > * { min-width: 0; }
.detail-full { grid-column: 1 / -1; }
.detail h4 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); margin: 0 0 6px; }
.detail-block { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px; max-height: 420px; overflow: auto; }
.detail-block pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; line-height: 1.5; }
.detail-block.prose pre { font-family: inherit; font-size: 13px; }
.meta-row { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 12px; font-size: 12px; color: var(--text-secondary); }
.meta-row b { color: var(--text-primary); }
.loading { color: var(--muted); padding: 40px; text-align: center; }
@media (max-width: 760px) { .detail { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="subtitle"><a href="index.html">← Run dashboard</a> &middot; __SUBTITLE__</p>

  <div class="tiles" id="tiles"></div>

  <div class="charts">
    <div class="chart-card">
      <div class="title">Committed winner score by step (green = new chain best)</div>
      __TRAJECTORY_CHART__
    </div>
    <div class="chart-card">
      <div class="title">Score distribution (valid candidates)</div>
      <svg id="hist" viewBox="0 0 1000 140" preserveAspectRatio="none"></svg>
    </div>
  </div>

  <div class="controls">
    <div class="group">
      <label for="f-score">Min score</label>
      <input type="number" id="f-score" step="any" placeholder="any" style="width:130px">
    </div>
    <div class="group">
      <label for="f-status">Status</label>
      <select id="f-status">
        <option value="">all</option>
        <option value="committed">committed</option>
        <option value="valid">valid</option>
        <option value="error">error</option>
        <option value="rejected">rejected</option>
        <option value="gen_failure">generation failure</option>
      </select>
    </div>
    <div class="group">
      <label for="f-step">Step</label>
      <input type="number" id="f-step" min="1" placeholder="any" style="width:80px">
    </div>
    <div class="group">
      <label for="f-search">Search msg/id/code</label>
      <input type="text" id="f-search" placeholder="filter…" style="width:180px">
    </div>
    <span id="count-label"></span>
  </div>

  <div class="table-scroll">
  <table id="tbl">
    <thead id="thead"></thead>
    <tbody id="tbody">
      <tr><td class="loading">Decompressing…</td></tr>
    </tbody>
  </table>
  </div>
  <footer>Generated by scripts/checkpoint_to_html.py from __DB_STATE__ &middot; rows with a green edge are committed winners (one per step)</footer>
</div>

<script>
const DATA_B64 = "__DATA_B64__";
const META = __META_JSON__;

async function decodePayload(b64) {
  const binaryStr = atob(b64);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
  const ds = new DecompressionStream("gzip");
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  const buf = await new Response(stream).arrayBuffer();
  return JSON.parse(new TextDecoder().decode(buf));
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
function fmt(n, d = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return Number(n).toFixed(d);
}
function shortId(id) { return id ? String(id).slice(0, 8) : "–"; }

let ROWS = [];
let CODE_BY_ID = {};
let sortKey = "i";
let sortDir = 1;

const COLUMNS = [
  { key: "i", label: "#" },
  { key: "step", label: "Step" },
  { key: "id", label: "Node" },
  { key: "score", label: "Score" },
  { key: "status", label: "Status" },
  { key: "improvement_label", label: "vs parent" },
  { key: "c3", label: "C3" },
  { key: "eval_time", label: "Eval (s)" },
  { key: "parent_id", label: "Parent" },
  { key: "msg", label: "Msg" },
];

function renderHead() {
  document.getElementById("thead").innerHTML =
    "<tr>" + COLUMNS.map((c) => `<th data-key="${c.key}">${esc(c.label)}</th>`).join("") + "</tr>";
}

function renderTiles() {
  const scores = ROWS.map((r) => r.score).filter((v) => typeof v === "number");
  const best = scores.length ? Math.max(...scores) : null;
  const labeled = ROWS.filter((r) => r.improvement_label != null).length;
  const improved = ROWS.filter((r) => r.improvement_label === "improved").length;
  const items = [
    ["Candidates", ROWS.length.toLocaleString(), false],
    ["Valid rate", ROWS.length ? ((100 * scores.length) / ROWS.length).toFixed(1) + "%" : "–", true],
    ["Improved %", labeled ? ((100 * improved) / labeled).toFixed(1) + "%" : "–", true],
    ["Best score", fmt(best, 7), false],
    ["Gain over seed", best != null && META.seed_score != null ? (best - META.seed_score).toExponential(2) : "–", true],
    ["Steps", `${META.steps} / ${META.budget}`, false],
    ["Errors", ROWS.filter((r) => r.status === "error").length, false],
    ["Rejected", ROWS.filter((r) => r.status === "rejected" || r.status === "gen_failure").length, false],
  ];
  document.getElementById("tiles").innerHTML = items
    .map(([label, value, good]) =>
      `<div class="tile"><div class="label">${esc(label)}</div><div class="value${good ? " good" : ""}">${esc(value)}</div></div>`)
    .join("");
}

function renderHistogram() {
  const svg = document.getElementById("hist");
  const values = ROWS.map((r) => r.score).filter((v) => typeof v === "number");
  if (!values.length) { svg.innerHTML = ""; return; }
  // Clip to the top 95% of values so a few far-off outliers don't flatten the bins.
  const sorted = [...values].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.05)], hi = sorted[sorted.length - 1];
  const nb = 24, span = hi - lo || Math.abs(hi) * 1e-9 || 1e-9;
  const bins = Array.from({ length: nb }, (_, i) => ({ lo: lo + (span * i) / nb, hi: lo + (span * (i + 1)) / nb, count: 0 }));
  for (const v of values) {
    if (v < lo) continue;
    bins[Math.min(nb - 1, Math.floor(((v - lo) / span) * nb))].count++;
  }
  const W = 1000, H = 140, padB = 20, padT = 4;
  const maxCount = Math.max(...bins.map((b) => b.count), 1);
  const bw = W / bins.length;
  let out = "";
  bins.forEach((b, i) => {
    const h = ((H - padT - padB) * b.count) / maxCount;
    out += `<rect class="hist-bar" x="${(i * bw + 1).toFixed(1)}" y="${(H - padB - h).toFixed(1)}" width="${Math.max(bw - 2, 1).toFixed(1)}" height="${h.toFixed(1)}" rx="2"><title>${b.lo.toFixed(8)} – ${b.hi.toFixed(8)}: ${b.count}</title></rect>`;
  });
  out += `<line x1="0" y1="${H - padB}" x2="${W}" y2="${H - padB}" stroke="var(--baseline)" stroke-width="1"/>`;
  out += `<text class="hist-axis-text" x="2" y="${H - 4}">${lo.toFixed(7)} (p5)</text>`;
  out += `<text class="hist-axis-text" x="${W - 70}" y="${H - 4}">${hi.toFixed(7)}</text>`;
  svg.innerHTML = out;
}

function filteredSorted() {
  const minScore = document.getElementById("f-score").value;
  const status = document.getElementById("f-status").value;
  const step = document.getElementById("f-step").value;
  const search = document.getElementById("f-search").value.trim().toLowerCase();
  let out = ROWS.filter((r) => {
    if (minScore !== "" && !(r.score >= Number(minScore))) return false;
    if (status && r.status !== status) return false;
    if (step !== "" && r.step !== Number(step)) return false;
    if (search) {
      const hay = `${r.msg || ""} ${r.id || ""} ${r.parent_id || ""} ${r.code || ""}`.toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });
  out.sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey];
    if (av === null || av === undefined) av = -Infinity;
    if (bv === null || bv === undefined) bv = -Infinity;
    if (typeof av === "string" || typeof bv === "string") return sortDir * String(av).localeCompare(String(bv));
    return sortDir * (av - bv);
  });
  return out;
}

const STATUS_CLASS = { committed: "good", valid: "info", error: "critical", rejected: "warn", gen_failure: "warn" };
const STATUS_LABEL = { committed: "committed", valid: "valid", error: "error", rejected: "rejected", gen_failure: "no code" };

function statusBadge(r) {
  return `<span class="status ${STATUS_CLASS[r.status] || "muted"}"><span class="dot"></span>${esc(STATUS_LABEL[r.status] || r.status)}</span>`;
}
function improvementBadge(r) {
  const label = r.improvement_label;
  if (label == null) return "–";
  const cls = label === "improved" ? "good" : label === "regressed" ? "critical" : "muted";
  return `<span class="status ${cls}"><span class="dot"></span>${esc(label)}</span>`;
}
function scoreCell(r, lo, hi) {
  const v = r.score;
  if (typeof v !== "number") return "–";
  const pct = hi > lo ? Math.max(0, Math.min(100, (100 * (v - lo)) / (hi - lo))) : 100;
  return `<span class="reward-bar-bg"><span class="reward-bar-fill" style="width:${pct.toFixed(1)}%"></span></span>${fmt(v, 9)}`;
}

function detailHTML(r) {
  const blocks = [];
  blocks.push(`
    <div class="detail-full meta-row">
      <span>Node <b>${esc(r.id || "–")}</b></span>
      <span>Batch <b>${esc(r.gen_id ?? "–")}</b></span>
      <span>Step <b>${esc(r.step ?? "–")}</b></span>
      <span>Parent <b>${esc(shortId(r.parent_id))}</b>${r.parent_score != null ? ` (${fmt(r.parent_score, 9)})` : ""}</span>
      <span>Warm-start construction <b>${esc(shortId(r.construction_id))}</b></span>
      <span>Created <b>${esc(r.created_at || "–")}</b></span>
    </div>`);
  if (r.msg) {
    blocks.push(`
    <div class="detail-full">
      <h4>Error</h4>
      <div class="detail-block"><pre>${esc(r.msg)}</pre></div>
    </div>`);
  }
  if (r.reflection) {
    blocks.push(`
    <div class="detail-full">
      <h4>Reflection on this winner (fed into later prompts of this chain)</h4>
      <div class="detail-block prose"><pre>${esc(r.reflection)}</pre></div>
    </div>`);
  }
  if (r.llm_input || r.llm_output) {
    blocks.push(`
    <div>
      <h4>Generation — input (prompt)</h4>
      <div class="detail-block prose"><pre>${esc(r.llm_input || "(not saved)")}</pre></div>
    </div>
    <div>
      <h4>Generation — response</h4>
      <div class="detail-block prose"><pre>${esc(r.llm_output || "(empty)")}</pre></div>
    </div>`);
  }
  const parentCode = CODE_BY_ID[r.parent_id];
  if (parentCode) {
    blocks.push(`
    <div>
      <h4>Parent code (${esc(shortId(r.parent_id))})</h4>
      <div class="detail-block"><pre>${esc(parentCode)}</pre></div>
    </div>`);
  }
  blocks.push(`
    <div class="${parentCode ? "" : "detail-full"}">
      <h4>Candidate code</h4>
      <div class="detail-block"><pre>${esc(r.code) || "(no code extracted)"}</pre></div>
    </div>`);
  if (r.metrics) {
    blocks.push(`
    <div class="detail-full">
      <h4>Metrics</h4>
      <div class="detail-block"><pre>${esc(r.metrics)}</pre></div>
    </div>`);
  }
  return `<div class="detail">${blocks.join("")}</div>`;
}

let expanded = new Set();

function renderTable() {
  const rows = filteredSorted();
  document.getElementById("count-label").textContent = `${rows.length.toLocaleString()} / ${ROWS.length.toLocaleString()} candidates`;
  const scores = ROWS.map((r) => r.score).filter((v) => typeof v === "number").sort((a, b) => a - b);
  const lo = scores.length ? scores[Math.floor(scores.length * 0.05)] : 0;
  const hi = scores.length ? scores[scores.length - 1] : 1;
  const tbody = document.getElementById("tbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="${COLUMNS.length}" class="loading">No candidates match the current filters.</td></tr>`;
    return;
  }
  // Rendering ~1.6k rows with inline detail is fast enough; detail HTML is only built for expanded rows.
  let html = "";
  for (const r of rows) {
    const isOpen = expanded.has(r.i);
    html += `<tr class="row${isOpen ? " expanded" : ""}${r.status === "committed" ? " committed" : ""}" data-i="${r.i}">
      <td>${r.i}</td>
      <td>${r.step ?? "–"}</td>
      <td class="mono">${esc(shortId(r.id))}</td>
      <td class="score">${scoreCell(r, lo, hi)}</td>
      <td>${statusBadge(r)}</td>
      <td>${improvementBadge(r)}</td>
      <td>${fmt(r.c3, 9)}</td>
      <td>${fmt(r.eval_time, 1)}</td>
      <td class="mono">${esc(shortId(r.parent_id))}</td>
      <td class="msg" title="${esc(r.msg || "")}">${esc(r.msg || "")}</td>
    </tr>`;
    if (isOpen) html += `<tr class="detail-row" data-i="${r.i}"><td colspan="${COLUMNS.length}">${detailHTML(r)}</td></tr>`;
  }
  tbody.innerHTML = html;
  tbody.querySelectorAll("tr.row").forEach((tr) => {
    tr.addEventListener("click", () => {
      const i = Number(tr.dataset.i);
      if (expanded.has(i)) expanded.delete(i); else expanded.add(i);
      renderTable();
    });
  });
}

function wireControls() {
  ["f-score", "f-status", "f-step", "f-search"].forEach((id) => {
    document.getElementById(id).addEventListener("input", renderTable);
  });
  document.querySelectorAll("#thead th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (sortKey === key) sortDir *= -1;
      else { sortKey = key; sortDir = key === "i" || key === "step" ? 1 : -1; }
      document.querySelectorAll("#thead th").forEach((h) => h.classList.remove("sorted"));
      th.classList.add("sorted");
      th.dataset.dir = sortDir === 1 ? "▲" : "▼";
      renderTable();
    });
  });
}

(async function init() {
  renderHead();
  try {
    const payload = await decodePayload(DATA_B64);
    ROWS = payload.rows;
    CODE_BY_ID = payload.code_by_id;
  } catch (err) {
    document.getElementById("tbody").innerHTML =
      `<tr><td colspan="${COLUMNS.length}" class="loading">Failed to decode payload: ${esc(err)}. This page needs a browser with native gzip DecompressionStream support (recent Chrome/Edge/Firefox/Safari).</td></tr>`;
    return;
  }
  renderTiles();
  renderHistogram();
  wireControls();
  renderTable();
})();
</script>
</body>
</html>
"""


def render_chain(summary: ChainSummary, ckpt: dict, run_info: dict, title: str, source: str) -> str:
    history = ckpt["policy"].get("chain_history", {}).get(str(summary.chain), [])
    score_by_id = {n["id"]: as_float(n.get("score")) for n in ckpt["nodes"]}
    code_by_id = {nid: run_info["by_id_code"].get(nid, "") for nid in history}
    if run_info["root_id"]:
        code_by_id[run_info["root_id"]] = run_info["root_code"]

    for row in summary.rows:
        row["parent_score"] = score_by_id.get(row["parent_id"])

    # Committed winner score per step, marking steps that set a new chain best.
    winner_points: list[tuple[float, float]] = []
    new_best_points: list[tuple[float, float]] = []
    best = run_info["seed_score"]
    for row in summary.rows:
        if row["status"] == "committed" and row["step"] is not None and row["score"] is not None:
            winner_points.append((row["step"], row["score"]))
            if best is None or row["score"] > best + 1e-12:
                best = row["score"]
                new_best_points.append((row["step"], row["score"]))
    chart = render_line_chart_svg(
        winner_points,
        css_class="chart-line-score",
        digits=7,
        dots=False,
        references=run_info.get("references"),
        markers=new_best_points,
        marker_note=" (new chain best)",
    )

    payload = json.dumps({"rows": summary.rows, "code_by_id": code_by_id}, separators=(",", ":"), default=str)
    data_b64 = base64.b64encode(gzip.compress(payload.encode("utf-8"), compresslevel=9, mtime=0)).decode("ascii")
    meta = {"seed_score": run_info["seed_score"], "steps": summary.steps, "budget": summary.budget}
    gain = summary.best - run_info["seed_score"] if summary.best is not None and run_info["seed_score"] is not None else None
    subtitle = (
        f"{summary.candidates:,} candidates over {summary.steps} steps &middot; best {fmt(summary.best, 9)}"
        + (f" ({gain:+.2e} over seed)" if gain is not None else "")
    )
    return (
        CHAIN_TEMPLATE.replace("__CSS__", BASE_CSS)
        .replace("__TITLE__", esc(f"{title} · chain {summary.chain}"))
        .replace("__SUBTITLE__", subtitle)
        .replace("__TRAJECTORY_CHART__", chart)
        .replace("__DB_STATE__", esc(source))
        .replace("__META_JSON__", json.dumps(meta))
        .replace("__DATA_B64__", data_b64)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db_state", type=Path, help="Checkpoint directory (db_state_*) containing nodes.json")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: <db_state>/html)")
    parser.add_argument("--title", default=None, help="Page title (default: SimpleTES run <instance>)")
    parser.add_argument("--chains", type=int, nargs="*", default=None, help="Only write reports for these chains")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="LABEL=SCORE",
        help="Extra reference result in combined_score units, drawn as a dashed line on score charts (repeatable)",
    )
    parser.add_argument(
        "--no-task-references",
        action="store_true",
        help="Do not draw the built-in TASK_REFERENCES for this run's task",
    )
    parser.add_argument(
        "--reference-score",
        type=float,
        default=None,
        help="Single reference result (same as --reference LABEL=SCORE with --reference-label)",
    )
    parser.add_argument("--reference-label", default="reference", help="Label for --reference-score")
    parser.add_argument("--source", default=None, help="Checkpoint location shown in page footers (default: db_state path)")
    args = parser.parse_args()

    db_state = args.db_state.resolve()
    out = (args.out or db_state / "html").resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {db_state} …")
    ckpt = load_checkpoint(db_state)
    chains, run_info = build_chains(ckpt)
    task = task_name(ckpt["config"])
    references = [] if args.no_task_references else [(score, label) for label, score in TASK_REFERENCES.get(task, [])]
    if references:
        print(f"  task {task}: drawing {len(references)} built-in reference line(s)")
    if args.reference_score is not None:
        references.append((args.reference_score, args.reference_label))
    for item in args.reference:
        label, sep, value = item.rpartition("=")
        if not sep or not label:
            parser.error(f"--reference expects LABEL=SCORE, got {item!r}")
        references.append((float(value), label))
    run_info["references"] = references
    source = args.source or str(db_state)
    title = args.title or f"SimpleTES run {ckpt['metadata'].get('instance_id', db_state.parent.name)}"

    (out / "index.html").write_text(render_dashboard(ckpt, chains, run_info, title, source), encoding="utf-8")
    print(f"  wrote {out / 'index.html'}")
    for c, summary in sorted(chains.items()):
        if args.chains is not None and c not in args.chains:
            continue
        path = out / f"chain{c:02d}.html"
        path.write_text(render_chain(summary, ckpt, run_info, title, source), encoding="utf-8")
        print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
