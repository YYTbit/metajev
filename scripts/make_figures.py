"""Generate the figures used in the README.

The figures are computed from a store built the same way the example builds one,
so the numbers in the picture are the numbers the code produces rather than
numbers chosen to look good. Re-run this after changing anything that alters
scores, and commit the regenerated files.

    python scripts/make_figures.py

Writes docs/reliability.svg and docs/trade.svg.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import metajev
from metajev.calibrate import observations_from_store, reliability, summarise
from metajev.report import collect
from metajev.store import DecisionStore

DOCS = ROOT / "docs"

INK = "#16181d"
MUTED = "#5c6370"
LINE = "#e3e6ea"
PANEL = "#ffffff"
ACCENT = "#2b4acb"
ACCEPT = "#1f7a4d"
DIVERT = "#b3261e"

FONT = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)

OVERCONFIDENCE = 0.15
TICKETS = 500


def unit_hash(*parts: str) -> float:
    """A reproducible number in [0, 1) derived from the given strings.

    Kept identical to the one in examples/triage.py so that a figure and the
    console output printed beside it describe the same run.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:6], "big") / float(1 << 48)


def build_store(path: str) -> DecisionStore:
    """Build the demo history the figures are drawn from."""
    question = metajev.Question.noul(
        "Does this ticket need a person before anything happens?",
        yes="a human should read this before any automated reply goes out",
    )
    store = DecisionStore(path)
    with metajev.Client(
        provider=metajev.MockProvider(sharpness=2.5),
        store=store,
        policy=metajev.three_band_policy("loose", accept=0.70, review=0.45),
    ) as client:
        for index in range(TICKETS):
            ticket = f"ticket-{index:04d}"
            answer = client.decide(ticket, question)
            skill = max(0.0, min(1.0, answer.decision.confidence - OVERCONFIDENCE))
            if unit_hash(ticket, "resolution") < skill:
                truth = answer.decision.answer
            else:
                truth = "no" if answer.decision.answer == "yes" else "yes"
            client.record_outcome(answer, truth)
    return store


def svg_open(width: int, height: int, label: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{label}">',
        f'<rect width="{width}" height="{height}" fill="{PANEL}"/>',
        f'<g font-family="{FONT}">',
    ]


def reliability_figure(store: DecisionStore) -> str:
    """Predicted probability against observed frequency."""
    observations, unjudged = observations_from_store(store, on="confidence")
    summary = summarise(observations, name="confidence", bins=10)
    buckets = reliability(observations, bins=10)

    W, H, pad = 460, 380, 62
    plot = W - pad - 26
    plot_h = H - pad - 58

    def x(p: float) -> float:
        return pad + p * plot

    def y(p: float) -> float:
        return H - pad - p * plot_h

    parts = svg_open(W, H, "Reliability curve")
    parts.append(
        f'<text x="{pad}" y="30" font-size="15" font-weight="600" fill="{INK}">'
        f"Whether the numbers mean anything</text>"
    )
    parts.append(
        f'<text x="{pad}" y="49" font-size="12" fill="{MUTED}">'
        f"{summary.n} decisions with a recorded outcome</text>"
    )

    for index in range(6):
        t = index / 5
        parts.append(
            f'<line x1="{x(t):.1f}" y1="{y(0):.1f}" x2="{x(t):.1f}" y2="{y(1):.1f}" '
            f'stroke="{LINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{x(0):.1f}" y1="{y(t):.1f}" x2="{x(1):.1f}" y2="{y(t):.1f}" '
            f'stroke="{LINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x(t):.1f}" y="{y(0) + 17:.1f}" font-size="10" fill="{MUTED}" '
            f'text-anchor="middle">{t:.1f}</text>'
        )
        parts.append(
            f'<text x="{x(0) - 9:.1f}" y="{y(t) + 3.5:.1f}" font-size="10" fill="{MUTED}" '
            f'text-anchor="end">{t:.1f}</text>'
        )

    parts.append(
        f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}" '
        f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>'
    )
    parts.append(
        f'<text x="{x(0.04):.1f}" y="{y(0.94):.1f}" font-size="10" fill="{MUTED}">'
        f"perfect calibration</text>"
    )

    bar = plot / 11
    for bucket in buckets:
        cx = x((bucket.lo + bucket.hi) / 2)
        top = y(bucket.accuracy)
        parts.append(
            f'<rect x="{cx - bar / 2:.1f}" y="{top:.1f}" width="{bar:.1f}" '
            f'height="{max(1.0, y(0) - top):.1f}" fill="{ACCENT}" opacity="0.72"/>'
        )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{y(bucket.mean_predicted):.1f}" r="3.6" fill="{INK}"/>'
        )

    parts.append(
        f'<text x="{x(0):.1f}" y="{H - 20:.1f}" font-size="11" fill="{MUTED}">'
        f"confidence the model reported</text>"
    )
    parts.append(
        f'<text x="{pad}" y="{H - 4:.1f}" font-size="12" fill="{INK}">'
        f"Brier {summary.brier:.3f}   gap {summary.overconfidence:+.3f}</text>"
    )
    parts.append("</g></svg>")
    return "\n".join(parts)


def trade_figure(store: DecisionStore) -> str:
    """Coverage and precision against the accept boundary, with the crossing marked."""
    payload = collect(store, baseline=metajev.three_band_policy(accept=0.85, review=0.55))
    # A boundary above every recorded confidence accepts nothing, so both lines fall
    # to zero together. That is true and it reads as a crash, so the curve stops at
    # the last boundary that still accepts something.
    last = max((i for i, point in enumerate(payload.sweep) if point[1] > 0), default=0)
    curve = payload.sweep[: last + 1]

    W, H, pad = 780, 380, 62
    plot = W - pad - 214
    plot_h = H - pad - 58

    def x(p: float) -> float:
        return pad + p * plot

    def y(p: float) -> float:
        return H - pad - p * plot_h

    parts = svg_open(W, H, "Coverage and precision against the accept boundary")
    parts.append(
        f'<text x="{pad}" y="30" font-size="15" font-weight="600" fill="{INK}">'
        f"What each boundary would have cost</text>"
    )
    parts.append(
        f'<text x="{pad}" y="49" font-size="12" fill="{MUTED}">'
        f"{payload.decisions} decisions, one model call each, never called again</text>"
    )

    for index in range(5):
        t = index / 4
        parts.append(
            f'<line x1="{x(0):.1f}" y1="{y(t):.1f}" x2="{x(1):.1f}" y2="{y(t):.1f}" '
            f'stroke="{LINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x(0) - 9:.1f}" y="{y(t) + 3.5:.1f}" font-size="10" fill="{MUTED}" '
            f'text-anchor="end">{t:.2f}</text>'
        )
    for index in range(6):
        t = index / 5
        parts.append(
            f'<text x="{x(t):.1f}" y="{y(0) + 17:.1f}" font-size="10" fill="{MUTED}" '
            f'text-anchor="middle">{t:.1f}</text>'
        )

    coverage = " ".join(f"{x(p[0]):.1f},{y(p[1]):.1f}" for p in curve)
    precision = " ".join(f"{x(p[0]):.1f},{y(p[2]):.1f}" for p in curve)
    parts.append(f'<polyline points="{coverage}" fill="none" stroke="{ACCENT}" stroke-width="2.2"/>')
    parts.append(f'<polyline points="{precision}" fill="none" stroke="{ACCEPT}" stroke-width="2.2"/>')

    # Mark where precision overtakes coverage.
    crossing = None
    for point in curve:
        if point[2] > point[1]:
            crossing = point
            break
    if crossing:
        parts.append(
            f'<line x1="{x(crossing[0]):.1f}" y1="{y(0):.1f}" x2="{x(crossing[0]):.1f}" '
            f'y2="{y(crossing[2]):.1f}" stroke="{DIVERT}" stroke-width="1.4" '
            f'stroke-dasharray="3 3"/>'
        )
        parts.append(
            f'<circle cx="{x(crossing[0]):.1f}" cy="{y(crossing[2]):.1f}" r="4" '
            f'fill="{DIVERT}"/>'
        )

    parts.append(
        f'<line x1="{x(payload.baseline):.1f}" y1="{y(0):.1f}" x2="{x(payload.baseline):.1f}" '
        f'y2="{y(1):.1f}" stroke="{INK}" stroke-width="1.2" opacity="0.35"/>'
    )

    legend_x = pad + plot + 22
    parts.append(
        f'<line x1="{legend_x}" y1="{y(0.86):.1f}" x2="{legend_x + 18}" y2="{y(0.86):.1f}" '
        f'stroke="{ACCENT}" stroke-width="2.2"/>'
    )
    parts.append(
        f'<text x="{legend_x + 24}" y="{y(0.86) + 4:.1f}" font-size="11" fill="{MUTED}">'
        f"share accepted</text>"
    )
    parts.append(
        f'<line x1="{legend_x}" y1="{y(0.74):.1f}" x2="{legend_x + 18}" y2="{y(0.74):.1f}" '
        f'stroke="{ACCEPT}" stroke-width="2.2"/>'
    )
    parts.append(
        f'<text x="{legend_x + 24}" y="{y(0.74) + 4:.1f}" font-size="11" fill="{MUTED}">'
        f"precision among them</text>"
    )
    if crossing:
        parts.append(
            f'<text x="{legend_x + 24}" y="{y(0.62) + 4:.1f}" font-size="11" fill="{DIVERT}">'
            f"they cross at {crossing[0]:.2f}</text>"
        )
    parts.append(
        f'<text x="{x(0):.1f}" y="{H - 20:.1f}" font-size="11" fill="{MUTED}">'
        f"accept at or above</text>"
    )
    accepted = payload.baseline
    here = min(curve, key=lambda p: abs(p[0] - accepted))
    parts.append(
        f'<text x="{pad}" y="{H - 4:.1f}" font-size="12" fill="{INK}">'
        f"at {accepted:.2f}, {here[1] * 100:.1f}% accepted at {here[2]:.3f} precision</text>"
    )
    parts.append("</g></svg>")
    return "\n".join(parts)


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    store_path = "/tmp/metajev-figures.db"
    for stale in (store_path,):
        Path(stale).unlink(missing_ok=True)

    store = build_store(store_path)
    try:
        (DOCS / "reliability.svg").write_text(reliability_figure(store), encoding="utf-8")
        (DOCS / "trade.svg").write_text(trade_figure(store), encoding="utf-8")
        print(f"wrote {DOCS / 'reliability.svg'}")
        print(f"wrote {DOCS / 'trade.svg'}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
