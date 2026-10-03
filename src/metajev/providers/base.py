"""The provider interface every decision backend implements.

A provider answers one typed question about one state and returns a
:class:`~metajev.types.Decision`. Everything a provider does beyond that, retries,
routing, budgeting, is another module's business. Keeping the interface this narrow
is what lets a store hold decisions from several backends side by side and lets a
policy treat them alike.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..errors import ProviderResponseError
from ..types import Decision, Question, QuestionType, decision_key, now, state_hash


@dataclass(frozen=True)
class ProviderAnswer:
    """What a provider returns before it is assembled into a decision."""

    answer: str
    distribution: tuple[tuple[str, float], ...]
    cost_usd: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)


def normalise_distribution(
    distribution: Mapping[str, float] | tuple[tuple[str, float], ...],
    labels: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    """Coerce a raw distribution into ordered non-negative pairs that sum to one.

    Providers disagree about ordering and occasionally emit values that do not sum
    to exactly one. Normalising here means every downstream reader can assume the
    same thing. Labels the question declares but the provider omitted are filled in
    at zero, so a distribution always covers the whole answer space.
    """
    items = list(distribution.items()) if isinstance(distribution, Mapping) else list(distribution)
    values: dict[str, float] = {}
    for label, probability in items:
        try:
            values[str(label)] = max(0.0, float(probability))
        except (TypeError, ValueError) as exc:
            raise ProviderResponseError(
                f"distribution value for {label!r} is not a number: {probability!r}"
            ) from exc
    if not values:
        raise ProviderResponseError("provider returned an empty distribution")
    total = sum(values.values())
    if total <= 0:
        raise ProviderResponseError("provider returned a distribution with no mass")
    # Emit every declared label in the order the question declares them, so that a
    # score question's levels keep their positions, then any undeclared label the
    # provider volunteered. A level the provider omitted takes zero mass rather than
    # vanishing, which would shift the indices of every level after it and silently
    # corrupt the expected position.
    ordered = list(labels)
    ordered += [label for label in values if label not in ordered]
    return tuple((label, values.get(label, 0.0) / total) for label in ordered)


class Provider(ABC):
    """Answers typed questions about states.

    Subclasses implement :meth:`_answer`. The surrounding :meth:`decide` handles
    timing, key derivation, and distribution validation so that no subclass has to.
    """

    name: str = "provider"
    model: str = "unknown"

    def __init__(self, *, name: str | None = None, model: str | None = None) -> None:
        if name is not None:
            self.name = name
        if model is not None:
            self.model = model

    @abstractmethod
    def _answer(self, state: str, question: Question) -> ProviderAnswer:
        """Produce a raw answer. Subclasses implement this."""

    def decide(self, state: str, question: Question) -> Decision:
        """Answer a question about a state and record the result as a decision."""
        started = time.perf_counter()
        raw = self._answer(state, question)
        latency_ms = (time.perf_counter() - started) * 1000.0

        distribution = normalise_distribution(raw.distribution, question.labels)
        answer = raw.answer
        if answer not in {label for label, _ in distribution}:
            answer = max(distribution, key=lambda pair: pair[1])[0]
        confidence = dict(distribution).get(answer, 0.0)

        digest = state_hash(state)
        return Decision(
            key=decision_key(digest, question, self.name, self.model),
            question_id=question.id,
            question_type=question.type,
            state_hash=digest,
            provider=self.name,
            model=self.model,
            answer=answer,
            confidence=confidence,
            distribution=distribution,
            tags=question.tags,
            latency_ms=latency_ms,
            cost_usd=raw.cost_usd,
            created_at=now(),
            meta=dict(raw.meta),
        )

    def estimate_cost(self, state: str, question: Question) -> float:
        """Estimate what :meth:`decide` would cost, before calling it.

        The default is zero, which is correct for a local model and is what the
        budget needs to know about one.
        """
        return 0.0

    def health(self) -> tuple[bool, str]:
        """Check whether this provider is usable. Return a flag and an explanation."""
        return True, "no health check implemented"

    def describe(self) -> dict[str, Any]:
        """Return a short description of the provider, for the CLI."""
        return {"name": self.name, "model": self.model, "kind": type(self).__name__}


__all__ = ["Provider", "ProviderAnswer", "normalise_distribution", "QuestionType"]
