"""Re-resolve recorded decisions under a different policy, with no model calls.

This is the payoff of keeping decisions and policies apart. A store holds the
distributions a model produced. A policy reads them. Applying a second policy to
the same store answers the question people actually ask when they want to move a
threshold: what would have happened if we had drawn the line somewhere else?

The answer arrives in milliseconds and costs nothing, because every quantity a
policy reads is already recorded. The same replay also reports what the change
would have cost, since a boundary that diverts errors also diverts some correct
answers, and a report that only counted the errors would be misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from .policy import Policy
from .store import DecisionStore
from .types import Action, Decision, Outcome


@dataclass(frozen=True)
class Bucket:
    """One action, and how the decisions routed to it turned out."""

    action: Action
    count: int
    share: float
    judged: int
    correct: int
    accuracy: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "count": self.count,
            "share": self.share,
            "judged": self.judged,
            "correct": self.correct,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class Flip:
    """A decision whose action differs between two policies."""

    key: str
    before: Action
    after: Action
    value: float
    outcome_label: str | None
    correct: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "before": self.before.value,
            "after": self.after.value,
            "value": self.value,
            "outcome": self.outcome_label,
            "correct": self.correct,
        }


@dataclass(frozen=True)
class ReplayReport:
    """What a policy does to a recorded history."""

    policy: str
    fingerprint: str
    total: int
    judged: int
    buckets: tuple[Bucket, ...]
    accept_precision: float | None
    baseline: str | None = None
    flips: int | None = None
    errors_diverted: int | None = None
    errors_admitted: int | None = None
    correct_diverted: int | None = None
    examples: tuple[Flip, ...] = ()

    def bucket(self, action: Action) -> Bucket | None:
        """Return the bucket for ``action``, or None when nothing landed there."""
        for bucket in self.buckets:
            if bucket.action is action:
                return bucket
        return None

    def render(self) -> str:
        """Render the report as a plain-text block."""
        lines = [
            f"policy {self.policy} (fingerprint {self.fingerprint})",
            f"replayed {self.total} decisions, {self.judged} with a recorded outcome",
            "",
            f"{'action':<10}{'count':>8}{'share':>9}{'judged':>8}{'correct':>9}{'accuracy':>10}",
        ]
        for bucket in self.buckets:
            accuracy = f"{bucket.accuracy:.3f}" if bucket.accuracy is not None else "-"
            lines.append(
                f"{bucket.action.value:<10}{bucket.count:>8}{bucket.share:>9.1%}"
                f"{bucket.judged:>8}{bucket.correct:>9}{accuracy:>10}"
            )
        if self.accept_precision is not None:
            lines.append("")
            lines.append(f"precision among accepted decisions: {self.accept_precision:.3f}")
        if self.baseline is not None:
            lines.append("")
            lines.append(f"against baseline policy {self.baseline}")
            lines.append(f"  actions that change:        {self.flips}")
            lines.append(f"  errors moved off accept:    {self.errors_diverted}")
            lines.append(f"  errors newly admitted:      {self.errors_admitted}")
            lines.append(f"  correct answers diverted:   {self.correct_diverted}")
        if self.examples:
            lines.append("")
            lines.append("largest changes")
            for flip in self.examples:
                verdict = "-" if flip.correct is None else ("right" if flip.correct else "wrong")
                lines.append(
                    f"  {flip.key[:12]}  {flip.before.value} -> {flip.after.value}"
                    f"  value={flip.value:.3f}  {verdict}"
                )
        return "\n".join(lines)


def _buckets(
    resolutions: Sequence[tuple[Decision, Action]],
    outcomes: dict[str, Outcome],
) -> tuple[Bucket, ...]:
    """Group decisions by action and count how they turned out."""
    total = len(resolutions) or 1
    order = list(Action)
    counts: dict[Action, list[int]] = {action: [0, 0, 0] for action in order}
    for decision, action in resolutions:
        counts[action][0] += 1
        outcome = outcomes.get(decision.key)
        if outcome is not None:
            counts[action][1] += 1
            if decision.answer == outcome.label:
                counts[action][2] += 1
    buckets = []
    for action in order:
        count, judged, correct = counts[action]
        if count == 0:
            continue
        buckets.append(
            Bucket(
                action=action,
                count=count,
                share=count / total,
                judged=judged,
                correct=correct,
                accuracy=(correct / judged) if judged else None,
            )
        )
    return tuple(buckets)


def replay(
    store: DecisionStore,
    policy: Policy,
    *,
    baseline: Policy | None = None,
    examples: int = 5,
    decisions: Iterable[Decision] | None = None,
) -> ReplayReport:
    """Apply ``policy`` to stored decisions and report what it does.

    Passing ``baseline`` adds a comparison: how many actions change, how many wrong
    answers move off accept, how many wrong answers stay accepted anyway, and how
    many right answers the change diverts. The last number is the price of the
    change, and leaving it out would make the report read as a free lunch.
    """
    population = list(decisions) if decisions is not None else list(store.decisions())
    outcomes = {outcome.key: outcome for outcome in store.outcomes()}
    judged = sum(1 for decision in population if decision.key in outcomes)

    resolved = [(decision, policy.resolve(decision).action) for decision in population]
    buckets = _buckets(resolved, outcomes)

    accepted = [decision for decision, action in resolved if action is Action.ACCEPT]
    judged_accepted = [d for d in accepted if d.key in outcomes]
    accept_precision = None
    if judged_accepted:
        hits = sum(1 for d in judged_accepted if d.answer == outcomes[d.key].label)
        accept_precision = hits / len(judged_accepted)

    if baseline is None:
        return ReplayReport(
            policy=policy.name,
            fingerprint=policy.fingerprint,
            total=len(population),
            judged=judged,
            buckets=buckets,
            accept_precision=accept_precision,
        )

    flips: list[Flip] = []
    errors_diverted = 0
    errors_admitted = 0
    correct_diverted = 0
    for decision, action in resolved:
        before = baseline.resolve(decision).action
        if before is action:
            continue
        outcome = outcomes.get(decision.key)
        correct = None if outcome is None else (decision.answer == outcome.label)
        flips.append(
            Flip(
                key=decision.key,
                before=before,
                after=action,
                value=decision.confidence,
                outcome_label=None if outcome is None else outcome.label,
                correct=correct,
            )
        )
        if correct is False:
            if before is Action.ACCEPT and action is not Action.ACCEPT:
                errors_diverted += 1
            elif before is not Action.ACCEPT and action is Action.ACCEPT:
                errors_admitted += 1
        elif correct is True and before is Action.ACCEPT and action is not Action.ACCEPT:
            correct_diverted += 1

    flips.sort(key=lambda flip: (flip.before.value, -flip.value))
    return ReplayReport(
        policy=policy.name,
        fingerprint=policy.fingerprint,
        total=len(population),
        judged=judged,
        buckets=buckets,
        accept_precision=accept_precision,
        baseline=baseline.name,
        flips=len(flips),
        errors_diverted=errors_diverted,
        errors_admitted=errors_admitted,
        correct_diverted=correct_diverted,
        examples=tuple(flips[:examples]),
    )


def compare(
    store: DecisionStore,
    policies: Sequence[Policy],
    *,
    baseline: Policy | None = None,
) -> list[ReplayReport]:
    """Replay several policies over one store.

    Comparing candidates against a shared history is the reason to record decisions
    at all. Each policy costs a pass over the store and nothing else.
    """
    return [replay(store, policy, baseline=baseline) for policy in policies]


def sweep(
    store: DecisionStore,
    make_policy,
    values: Sequence[float],
    *,
    baseline: Policy | None = None,
    decisions: Iterable[Decision] | None = None,
) -> list[tuple[float, ReplayReport]]:
    """Replay one policy family across a range of a parameter.

    ``make_policy`` takes a value and returns a policy. Use this to see where a
    boundary stops buying error reduction and starts costing correct answers.
    """
    population = None if decisions is None else list(decisions)
    return [
        (value, replay(store, make_policy(value), baseline=baseline, decisions=population))
        for value in values
    ]
