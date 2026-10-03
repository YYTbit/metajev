"""Move a threshold without re-running the model.

Run this with nothing installed but the package itself:

    python examples/triage.py

The example routes support tickets to a person or to automation. Every ticket is
judged once by an offline provider, and the truth for each ticket is recorded
afterwards. A second policy is then applied to that same recorded history, and the
difference between the two policies is reported in tickets rather than in calls,
because no calls are made.

The labels are synthetic so the example runs without a network or an API key. They
simulate a model whose confidence ranks tickets correctly but runs 0.15 hot: a
ticket it scores 0.90 is right about three times in four. That is the case a
calibration report exists for, and it is why the sweep below finds a boundary that
buys real error reduction rather than shuffling tickets at random.

Both the counts and the calibration curve are computed from whatever is in the
store, so the arithmetic is real even though the tickets are not.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import metajev


QUESTION = metajev.Question.noul(
    "Does this ticket need a person before anything happens?",
    yes="a human should read this before any automated reply goes out",
    tags=("triage",),
)

# How far above its true hit rate the simulated model's confidence sits.
OVERCONFIDENCE = 0.15


def unit_hash(*parts: str) -> float:
    """A reproducible number in [0, 1) derived from the given strings."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:6], "big") / float(1 << 48)


def truth_for(ticket: str, answer, confidence: float) -> str:
    """The truth for a ticket, from a simulated model with a known hit rate.

    A real deployment reads this from how the ticket actually resolved. Here the
    model is right with probability ``confidence - OVERCONFIDENCE``, so confidence
    orders tickets by how likely the answer is to hold while still sitting too high.
    """
    skill = max(0.0, min(1.0, confidence - OVERCONFIDENCE))
    stated = answer.decision.answer
    if unit_hash(ticket, "resolution") < skill:
        return stated
    return "no" if stated == "yes" else "yes"


def main() -> int:
    # A real deployment passes TypesafeProvider(), OpenAICompatProvider(), or
    # SGLangProvider(). The mock keeps this example offline.
    provider = metajev.MockProvider(sharpness=2.5)

    with metajev.Client(
        provider=provider,
        store=metajev.DecisionStore(":memory:"),
        policy=metajev.three_band_policy("loose", accept=0.70, review=0.45),
        budget=metajev.Budget(usd_limit=1.0),
    ) as client:
        tickets = [f"ticket-{index:04d}" for index in range(500)]
        print(f"judging {len(tickets)} tickets, one model call each")
        for ticket in tickets:
            answer = client.decide(ticket, QUESTION)
            client.record_outcome(answer, truth_for(ticket, answer, answer.decision.confidence))

        stats = client.stats()
        print(f"  {stats.decisions} decisions, {stats.outcomes} outcomes recorded")
        print(f"  {stats.mean_confidence:.3f} mean confidence")
        print(f"  {stats.total_cost_usd:.4f} USD spent, {client.budget.calls} calls")
        print()

        print(client.calibration().render())
        print()

        print("=" * 74)
        print("applying a stricter policy to the history already on disk")
        print("=" * 74)
        strict = metajev.three_band_policy("strict", accept=0.90, review=0.60)
        print(client.replay(strict, baseline=client.policy).render())
        print()

        print("=" * 74)
        print("sweeping the accept boundary")
        print("=" * 74)
        header = (
            f"{'accept':>8}{'changed':>10}{'accepted':>10}{'precision':>11}"
            f"{'errors caught':>15}{'right lost':>12}"
        )
        print(header)
        print("-" * len(header))
        for value in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
            policy = metajev.three_band_policy(f"accept-{value}", accept=value, review=0.40)
            swept = client.replay(policy, baseline=client.policy)
            accepted = swept.bucket(metajev.Action.ACCEPT)
            precision = swept.accept_precision if swept.accept_precision is not None else 0.0
            print(
                f"{value:>8.2f}{swept.flips:>10}{(accepted.count if accepted else 0):>10}"
                f"{precision:>11.3f}{swept.errors_diverted:>15}{swept.correct_diverted:>12}"
            )
        print()

        print(f"model calls made in total: {client.budget.calls}")
        print("every number above came from the recorded distributions")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
