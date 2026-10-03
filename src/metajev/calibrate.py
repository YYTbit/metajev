"""Calibration statistics over recorded decisions and outcomes.

A probability is useful only if it means something. This module measures whether
the numbers a model returns can be read as frequencies: when it says 0.8, is it
right about four times in five? Two summaries answer that, the Brier score for
overall accuracy of the probabilities and the expected calibration error for the
shape of the reliability curve, plus a bucketed table you can eyeball.

Everything here operates on decisions that carry a recorded outcome. Decisions
without one are counted and skipped rather than silently treated as correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from .store import DecisionStore
from .types import Decision, Outcome, QuestionType


@dataclass(frozen=True)
class Observation:
    """One decision paired with the truth, reduced to what calibration needs."""

    key: str
    predicted: float
    correct: bool
    weight: float = 1.0


@dataclass(frozen=True)
class ReliabilityBin:
    """One bucket of the reliability curve."""

    lo: float
    hi: float
    count: int
    mean_predicted: float
    accuracy: float
    gap: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "count": self.count,
            "mean_predicted": self.mean_predicted,
            "accuracy": self.accuracy,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class Calibration:
    """A calibration summary over a set of observations."""

    name: str
    n: int
    weighted_n: float
    brier: float
    ece: float
    mean_predicted: float
    accuracy: float
    bins: tuple[ReliabilityBin, ...]

    @property
    def overconfidence(self) -> float:
        """How far above the observed frequency the model's numbers sit, on average."""
        return self.mean_predicted - self.accuracy

    def render(self) -> str:
        """Render the summary and its reliability table as plain text."""
        lines = [
            f"calibration for {self.name}",
            f"  observations:      {self.n}",
            f"  brier score:       {self.brier:.4f}   (lower is better, 0.25 is a coin at p=0.5)",
            f"  expected cal err:  {self.ece:.4f}",
            f"  mean predicted:    {self.mean_predicted:.4f}",
            f"  observed accuracy: {self.accuracy:.4f}",
            f"  gap:               {self.overconfidence:+.4f}"
            + ("  (overconfident)" if self.overconfidence > 0.02 else "")
            + ("  (underconfident)" if self.overconfidence < -0.02 else ""),
            "",
            f"  {'bin':<16}{'count':>7}{'predicted':>11}{'actual':>9}{'gap':>9}",
        ]
        for bucket in self.bins:
            lines.append(
                f"  {f'[{bucket.lo:.1f},{bucket.hi:.1f})':<16}{bucket.count:>7}"
                f"{bucket.mean_predicted:>11.3f}{bucket.accuracy:>9.3f}{bucket.gap:>+9.3f}"
            )
        return "\n".join(lines)


def observations_from_store(
    store: DecisionStore,
    *,
    on: str = "confidence",
    question_id: str | None = None,
    provider: str | None = None,
    label: str | None = None,
) -> tuple[list[Observation], int]:
    """Read judged decisions out of a store as calibration observations.

    ``on`` selects which number to treat as the prediction. ``confidence`` uses the
    probability of the answer the model led with, which measures whether the model
    knows when it knows. ``label:<name>`` uses the probability of one named label,
    which measures whether the model's probability for that label can be read as a
    frequency. ``expected`` uses the normalised position on the answer scale for a
    score question.

    Returns the observations and the number of stored decisions that had no outcome.
    """
    observations: list[Observation] = []
    unjudged = 0
    for decision, outcome in store.judged():
        if question_id is not None and decision.question_id != question_id:
            continue
        if provider is not None and decision.provider != provider:
            continue
        if label is not None:
            predicted = decision.probability_of(label)
            correct = outcome.label == label
        elif on == "confidence":
            predicted = decision.confidence
            correct = decision.answer == outcome.label
        elif on == "expected":
            if decision.question_type is QuestionType.SCORE:
                span = len(decision.distribution) - 1
                predicted = decision.expected_level_index() / span if span > 0 else decision.confidence
            else:
                predicted = decision.confidence
            correct = decision.answer == outcome.label
        elif on.startswith("label:"):
            target = on[len("label:") :]
            predicted = decision.probability_of(target)
            correct = outcome.label == target
        else:
            raise ValueError(f"unknown calibration target {on!r}")
        observations.append(
            Observation(
                key=decision.key,
                predicted=min(1.0, max(0.0, predicted)),
                correct=correct,
                weight=outcome.weight,
            )
        )
    judged_keys = {decision.key for decision, _ in store.judged()}
    unjudged = max(0, store.count() - len(judged_keys))
    return observations, unjudged


def brier(observations: Sequence[Observation]) -> float:
    """Return the weighted Brier score, the mean squared error of the probabilities."""
    if not observations:
        return 0.0
    total_weight = sum(o.weight for o in observations)
    if total_weight <= 0:
        return 0.0
    return sum(o.weight * (o.predicted - (1.0 if o.correct else 0.0)) ** 2 for o in observations) / total_weight


def reliability(
    observations: Sequence[Observation],
    *,
    bins: int = 10,
) -> tuple[ReliabilityBin, ...]:
    """Bucket observations by predicted probability and compare to observed frequency."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    edges = [index / bins for index in range(bins + 1)]
    out: list[ReliabilityBin] = []
    for index in range(bins):
        lo, hi = edges[index], edges[index + 1]
        members = [
            o
            for o in observations
            if (lo <= o.predicted < hi) or (index == bins - 1 and o.predicted == 1.0)
        ]
        if not members:
            continue
        weight = sum(o.weight for o in members)
        mean_predicted = sum(o.weight * o.predicted for o in members) / weight
        accuracy = sum(o.weight * (1.0 if o.correct else 0.0) for o in members) / weight
        out.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                count=len(members),
                mean_predicted=mean_predicted,
                accuracy=accuracy,
                gap=mean_predicted - accuracy,
            )
        )
    return tuple(out)


def expected_calibration_error(bins: Sequence[ReliabilityBin]) -> float:
    """Return the count-weighted mean gap between predicted and observed frequency."""
    total = sum(bucket.count for bucket in bins)
    if total == 0:
        return 0.0
    return sum(bucket.count * abs(bucket.gap) for bucket in bins) / total


def summarise(observations: Sequence[Observation], name: str = "all", bins: int = 10) -> Calibration:
    """Compute the full calibration summary for a set of observations."""
    if not observations:
        return Calibration(
            name=name, n=0, weighted_n=0.0, brier=0.0, ece=0.0,
            mean_predicted=0.0, accuracy=0.0, bins=(),
        )
    total_weight = sum(o.weight for o in observations)
    mean_predicted = sum(o.weight * o.predicted for o in observations) / total_weight
    accuracy = sum(o.weight * (1.0 if o.correct else 0.0) for o in observations) / total_weight
    buckets = reliability(observations, bins=bins)
    return Calibration(
        name=name,
        n=len(observations),
        weighted_n=total_weight,
        brier=brier(observations),
        ece=expected_calibration_error(buckets),
        mean_predicted=mean_predicted,
        accuracy=accuracy,
        bins=buckets,
    )


def threshold_table(
    observations: Sequence[Observation],
    thresholds: Iterable[float],
) -> list[dict[str, float]]:
    """For each threshold, report what accepting at or above it would have caught.

    This is the calibration view of a policy boundary. It answers how many wrong
    answers a boundary admits and how many right answers it turns away, so the
    choice of boundary can be made against numbers rather than taste.
    """
    total = len(observations)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        admitted = [o for o in observations if o.predicted >= threshold]
        turned_away = total - len(admitted)
        right = sum(1 for o in admitted if o.correct)
        wrong = len(admitted) - right
        rows.append(
            {
                "threshold": threshold,
                "admitted": len(admitted),
                "coverage": len(admitted) / total if total else 0.0,
                "precision": (right / len(admitted)) if admitted else 0.0,
                "errors_admitted": wrong,
                "right_turned_away": sum(1 for o in observations if o.predicted < threshold and o.correct),
                "turned_away": turned_away,
            }
        )
    return rows
