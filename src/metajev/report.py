"""Render a decision store as a self-contained interactive page.

A report is the argument this package makes, drawn. It shows the recorded
probabilities, how well they match the recorded outcomes, and a slider that
moves the accept boundary across the whole history. Moving the slider re-resolves
every decision in the browser, from numbers already on the page.

There is no server, no build step, and no network access. The page is one file
with the decisions embedded as JSON, which is also why it is safe to open: the
only data in it is the probabilities and the labels they were checked against.
State text is never stored in a decision, so it cannot leak into a report.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from .calibrate import Observation, reliability, summarise
from .policy import Policy
from .store import DecisionStore
from .types import Action

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --ink: #16181d;
    --muted: #5c6370;
    --line: #e3e6ea;
    --bg: #ffffff;
    --panel: #f7f8fa;
    --accept: #1f7a4d;
    --review: #b26a00;
    --divert: #b3261e;
    --accent: #2b4acb;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 48px 24px 96px;
    background: var(--bg); color: var(--ink);
    font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  main { max-width: 880px; margin: 0 auto; }
  h1 { font-size: 26px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -0.01em; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.07em;
       color: var(--muted); font-weight: 600; margin: 52px 0 14px; }
  p.lede { color: var(--muted); margin: 0 0 32px; max-width: 62ch; }
  .stats { display: flex; flex-wrap: wrap; gap: 32px; padding: 20px 0;
           border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  .stat b { display: block; font-size: 22px; font-weight: 600; letter-spacing: -0.01em; }
  .stat span { color: var(--muted); font-size: 13px; }
  .controls { background: var(--panel); border: 1px solid var(--line);
              border-radius: 10px; padding: 22px 24px; margin-bottom: 28px; }
  .slider-row { display: flex; align-items: center; gap: 18px; }
  .slider-row label { font-size: 13px; color: var(--muted); white-space: nowrap; }
  input[type=range] { flex: 1; accent-color: var(--accent); height: 22px; }
  output { font-variant-numeric: tabular-nums; font-weight: 600;
           font-size: 17px; min-width: 4.2ch; text-align: right; }
  .outs { display: flex; flex-wrap: wrap; gap: 30px; margin-top: 20px; }
  .out b { display: block; font-size: 21px; font-variant-numeric: tabular-nums;
           font-weight: 600; letter-spacing: -0.01em; }
  .out span { font-size: 12px; color: var(--muted); }
  .out.good b { color: var(--accept); }
  .out.bad b { color: var(--divert); }
  .note { font-size: 13px; color: var(--muted); margin-top: 18px;
          padding-top: 14px; border-top: 1px dashed var(--line); }
  figure { margin: 0; }
  svg { width: 100%; height: auto; display: block; }
  figcaption { font-size: 13px; color: var(--muted); margin-top: 10px; }
  .bar { display: flex; height: 26px; border-radius: 5px; overflow: hidden;
         border: 1px solid var(--line); }
  .bar i { display: block; height: 100%; }
  .legend { display: flex; gap: 18px; font-size: 13px; color: var(--muted); margin-top: 10px; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
              margin-right: 6px; vertical-align: -1px; }
  footer { margin-top: 64px; padding-top: 20px; border-top: 1px solid var(--line);
           font-size: 13px; color: var(--muted); }
  code { background: var(--panel); padding: 1px 5px; border-radius: 4px;
         font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>
</head>
<body>
<main>
  <h1>Change the policy, not the model</h1>
  <p class="lede">Every decision below was made once and recorded with the full
  distribution the model returned. Everything on this page is computed from those
  recorded numbers. Moving the boundary costs nothing and calls nothing.</p>

  <div class="stats">
    <div class="stat"><b id="s-decisions">0</b><span>decisions recorded</span></div>
    <div class="stat"><b id="s-outcomes">0</b><span>outcomes known</span></div>
    <div class="stat"><b id="s-cost">0</b><span>USD spent</span></div>
    <div class="stat"><b id="s-model">0</b><span>model calls per boundary change</span></div>
  </div>

  <h2>Move the accept boundary</h2>
  <div class="controls">
    <div class="slider-row">
      <label for="accept">accept at or above</label>
      <input id="accept" type="range" min="0" max="1" step="0.01" value="__ACCEPT__">
      <output id="accept-out">__ACCEPT_TEXT__</output>
    </div>
    <div class="outs">
      <div class="out"><b id="o-accepted">0</b><span>accepted</span></div>
      <div class="out"><b id="o-precision">-</b><span>precision among accepted</span></div>
      <div class="out good"><b id="o-errors">0</b><span>wrong answers kept out</span></div>
      <div class="out bad"><b id="o-lost">0</b><span>right answers sent to review</span></div>
    </div>
    <div class="note">Compared against the boundary of __BASELINE__, which is where
    this history was originally resolved.</div>
  </div>

  <div class="bar" id="bar"></div>
  <div class="legend">
    <span><i style="background:var(--accept)"></i>accepted</span>
    <span><i style="background:var(--review)"></i>review</span>
    <span><i style="background:var(--divert)"></i>low confidence</span>
  </div>

  <h2>Whether the probabilities mean anything</h2>
  <figure>
    <svg id="reliability" viewBox="0 0 640 320" role="img"
         aria-label="Reliability curve"></svg>
    <figcaption>__RELIABILITY_CAPTION__</figcaption>
  </figure>

  <h2>What each boundary would have cost</h2>
  <figure>
    <svg id="sweep" viewBox="0 0 640 300" role="img"
         aria-label="Precision and coverage against boundary"></svg>
    <figcaption>The upper line is the share of decisions accepted. The lower line is
    the precision among them. Where the two cross is where a further increase stops
    buying accuracy and starts costing coverage.</figcaption>
  </figure>

  <footer>
    Generated by <code>metajev report</code>. This page is self-contained and makes
    no network requests.
  </footer>
</main>
<script>
const DATA = __DATA__;

const el = id => document.getElementById(id);

function recompute(accept) {
  const base = DATA.baseline;
  let acc = 0, right = 0, wrong = 0;
  let diverted = 0, lost = 0, review = 0, low = 0;

  for (const row of DATA.rows) {
    const p = row[0];
    const known = row.length > 1;
    const correct = known ? row[1] === 1 : null;

    if (p >= accept) {
      acc += 1;
      if (known) { if (correct) right += 1; else wrong += 1; }
    } else if (p >= DATA.review) {
      review += 1;
    } else {
      low += 1;
    }

    if (known) {
      const wasAccepted = p >= base;
      const nowAccepted = p >= accept;
      if (wasAccepted && !nowAccepted && !correct) diverted += 1;
      if (wasAccepted && !nowAccepted && correct) lost += 1;
    }
  }
  return { acc, right, wrong, diverted, lost, review, low,
           precision: (right + wrong) ? right / (right + wrong) : null };
}

function drawBar(r) {
  const total = r.acc + r.review + r.low || 1;
  el("bar").innerHTML =
    '<i style="background:var(--accept);width:' + (r.acc / total * 100) + '%"></i>' +
    '<i style="background:var(--review);width:' + (r.review / total * 100) + '%"></i>' +
    '<i style="background:var(--divert);width:' + (r.low / total * 100) + '%"></i>';
}

function drawReliability() {
  const svg = el("reliability");
  const W = 640, H = 320, pad = 48;
  const bins = DATA.reliability;
  const x = p => pad + p * (W - pad * 2);
  const y = p => H - pad - p * (H - pad * 2);
  let s = "";

  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    s += '<line x1="' + x(t) + '" y1="' + y(0) + '" x2="' + x(t) + '" y2="' + y(1) +
         '" stroke="var(--line)" stroke-width="1"/>';
    s += '<line x1="' + x(0) + '" y1="' + y(t) + '" x2="' + x(1) + '" y2="' + y(t) +
         '" stroke="var(--line)" stroke-width="1"/>';
    s += '<text x="' + x(t) + '" y="' + (y(0) + 18) + '" font-size="11" fill="var(--muted)"' +
         ' text-anchor="middle">' + t.toFixed(1) + '</text>';
    s += '<text x="' + (x(0) - 8) + '" y="' + (y(t) + 4) + '" font-size="11" fill="var(--muted)"' +
         ' text-anchor="end">' + t.toFixed(1) + '</text>';
  }
  s += '<line x1="' + x(0) + '" y1="' + y(0) + '" x2="' + x(1) + '" y2="' + y(1) +
       '" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4"/>';
  s += '<text x="' + x(0.72) + '" y="' + (y(0.72) - 8) + '" font-size="11" fill="var(--muted)">' +
       'a perfectly calibrated model</text>';

  let barW = (W - pad * 2) / Math.max(bins.length, 1) * 0.5;
  for (const b of bins) {
    const cx = x((b.lo + b.hi) / 2);
    s += '<rect x="' + (cx - barW / 2) + '" y="' + y(b.actual) + '" width="' + barW +
         '" height="' + Math.max(1, y(0) - y(b.actual)) + '" fill="var(--accent)" opacity="0.75"/>';
    s += '<circle cx="' + cx + '" cy="' + y(b.predicted) + '" r="3.5" fill="var(--ink)"/>';
  }
  s += '<text x="' + (W / 2) + '" y="' + (H - 8) + '" font-size="12" fill="var(--muted)"' +
       ' text-anchor="middle">confidence the model reported</text>';
  svg.innerHTML = s;
}

function drawSweep(accept) {
  const svg = el("sweep");
  const W = 640, H = 300, pad = 48;
  const curve = DATA.sweep;
  const x = p => pad + p * (W - pad * 2);
  const y = p => H - pad - p * (H - pad * 2);
  let s = "";

  for (let i = 0; i <= 4; i++) {
    const t = i / 4;
    s += '<line x1="' + x(0) + '" y1="' + y(t) + '" x2="' + x(1) + '" y2="' + y(t) +
         '" stroke="var(--line)" stroke-width="1"/>';
    s += '<text x="' + (x(0) - 8) + '" y="' + (y(t) + 4) + '" font-size="11" fill="var(--muted)"' +
         ' text-anchor="end">' + t.toFixed(2) + '</text>';
  }
  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    s += '<text x="' + x(t) + '" y="' + (y(0) + 18) + '" font-size="11" fill="var(--muted)"' +
         ' text-anchor="middle">' + t.toFixed(1) + '</text>';
  }

  const line = (key, color) => {
    const pts = curve.map(c => x(c[0]) + "," + y(c[key])).join(" ");
    return '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="2"/>';
  };
  s += line(1, "var(--accent)");
  s += line(2, "var(--accept)");

  s += '<line x1="' + x(accept) + '" y1="' + y(0) + '" x2="' + x(accept) + '" y2="' + y(1) +
       '" stroke="var(--divert)" stroke-width="1.5" stroke-dasharray="3 3"/>';
  s += '<text x="' + (x(accept) + 6) + '" y="' + (y(1) + 4) + '" font-size="11" fill="var(--divert)">' +
       'you are here</text>';
  s += '<text x="' + (W / 2) + '" y="' + (H - 8) + '" font-size="12" fill="var(--muted)"' +
       ' text-anchor="middle">accept boundary</text>';
  svg.innerHTML = s;
}

function update() {
  const accept = parseFloat(el("accept").value);
  el("accept-out").textContent = accept.toFixed(2);
  const r = recompute(accept);
  el("o-accepted").textContent = r.acc;
  el("o-precision").textContent = r.precision === null ? "-" : r.precision.toFixed(3);
  el("o-errors").textContent = r.diverted;
  el("o-lost").textContent = r.lost;
  drawBar(r);
  drawSweep(accept);
  location.hash = "a=" + accept.toFixed(2);
}

el("s-decisions").textContent = DATA.rows.length;
el("s-outcomes").textContent = DATA.rows.filter(r => r.length > 1).length;
el("s-cost").textContent = DATA.cost.toFixed(4);
el("s-model").textContent = "0";
drawReliability();
el("accept").addEventListener("input", update);
const fromHash = /a=([0-9.]+)/.exec(location.hash);
if (fromHash) el("accept").value = fromHash[1];
update();
</script>
</body>
</html>
"""


@dataclass(frozen=True)
class ReportInput:
    """Everything the page needs, gathered once."""

    rows: tuple[tuple[float, ...], ...]
    reliability: tuple[dict, ...]
    sweep: tuple[tuple[float, float, float], ...]
    baseline: float
    review: float
    cost: float
    decisions: int
    outcomes: int


def collect(
    store: DecisionStore,
    *,
    on: str = "confidence",
    baseline: Policy | None = None,
    bins: int = 10,
    sweep_steps: int = 40,
) -> ReportInput:
    """Read a store into the compact form the page embeds."""
    from .calibrate import observations_from_store

    observations, _ = observations_from_store(store, on=on)
    by_key = {o.key: o for o in observations}

    rows: list[tuple[float, ...]] = []
    for decision in store.decisions():
        if on == "confidence":
            predicted = decision.confidence
        elif on.startswith("label:"):
            predicted = decision.probability_of(on[len("label:") :])
        else:
            predicted = decision.confidence
        observation = by_key.get(decision.key)
        if observation is None:
            rows.append((round(predicted, 4),))
        else:
            rows.append((round(predicted, 4), 1 if observation.correct else 0))

    buckets = reliability(observations, bins=bins)
    reliability_payload = tuple(
        {"lo": b.lo, "hi": b.hi, "count": b.count,
         "predicted": b.mean_predicted, "actual": b.accuracy}
        for b in buckets
    )

    baseline_accept = _accept_boundary(baseline)
    review = _review_boundary(baseline)

    curve: list[tuple[float, float, float]] = []
    for index in range(sweep_steps + 1):
        threshold = index / sweep_steps
        accepted = [r for r in rows if r[0] >= threshold]
        judged = [r for r in accepted if len(r) > 1]
        coverage = len(accepted) / len(rows) if rows else 0.0
        precision = (
            sum(1 for r in judged if r[1] == 1) / len(judged) if judged else 0.0
        )
        curve.append((round(threshold, 4), round(coverage, 4), round(precision, 4)))

    stats = store.stats()
    return ReportInput(
        rows=tuple(rows),
        reliability=reliability_payload,
        sweep=tuple(curve),
        baseline=baseline_accept,
        review=review,
        cost=stats.total_cost_usd,
        decisions=stats.decisions,
        outcomes=stats.outcomes,
    )


def _accept_boundary(policy: Policy | None) -> float:
    """Read the accept boundary out of a policy, falling back to a common value."""
    if policy is None:
        return 0.85
    for rule in policy.rules:
        if rule.on != "confidence":
            continue
        for band in rule.bands:
            if band.action is Action.ACCEPT:
                return band.lo
    return 0.85


def _review_boundary(policy: Policy | None) -> float:
    """Read the review floor out of a policy."""
    if policy is None:
        return 0.55
    for rule in policy.rules:
        if rule.on != "confidence":
            continue
        for band in rule.bands:
            if band.action is Action.REVIEW:
                return band.lo
    return 0.55


def render_html(
    store: DecisionStore,
    *,
    title: str = "metajev report",
    on: str = "confidence",
    baseline: Policy | None = None,
    bins: int = 10,
) -> str:
    """Render a store as one self-contained interactive page."""
    payload = collect(store, on=on, baseline=baseline, bins=bins)

    observations, unjudged = _observations(store, on)
    summary = summarise(observations, name=on, bins=bins)
    caption = (
        f"{summary.n} decisions have a recorded outcome. "
        f"The Brier score is {summary.brier:.4f} and the expected calibration error "
        f"is {summary.ece:.4f}."
    )
    if summary.overconfidence > 0.02:
        caption += (
            f" The model reads {summary.overconfidence:.3f} more confident than it "
            f"has earned, which is what the gap below the diagonal shows."
        )
    elif summary.overconfidence < -0.02:
        caption += (
            f" The model is {abs(summary.overconfidence):.3f} less confident than it "
            f"has earned."
        )
    else:
        caption += " The probabilities track the outcomes closely."
    if unjudged:
        caption += f" {unjudged} decisions have no outcome yet and are excluded from this curve."

    data = {
        "rows": [list(row) for row in payload.rows],
        "reliability": list(payload.reliability),
        "sweep": [list(point) for point in payload.sweep],
        "baseline": payload.baseline,
        "review": payload.review,
        "cost": payload.cost,
    }

    return (
        _PAGE.replace("__TITLE__", html.escape(title))
        .replace("__ACCEPT_TEXT__", f"{payload.baseline:.2f}")
        .replace("__ACCEPT__", f"{payload.baseline:.2f}")
        .replace("__BASELINE__", f"{payload.baseline:.2f}")
        .replace("__RELIABILITY_CAPTION__", html.escape(caption))
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
    )


def _observations(store: DecisionStore, on: str) -> tuple[list[Observation], int]:
    from .calibrate import observations_from_store

    return observations_from_store(store, on=on)
