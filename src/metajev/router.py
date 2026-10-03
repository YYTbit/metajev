"""Choose which provider answers a question, using what past answers cost and caught.

The rule is short. Among the providers that can answer this question and are within
budget, take the cheapest whose observed accuracy on this question clears a floor.
Observed accuracy comes from the store, so the router improves as outcomes are
recorded and it needs no configuration to start, only a store that accumulates.

When nothing has been observed yet the router has no evidence and falls back to a
declared order, which is the honest behaviour rather than pretending a guess is a
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .budget import Budget
from .errors import MetajevError
from .providers.base import Provider
from .store import DecisionStore
from .types import Decision, Question


@dataclass(frozen=True)
class RouteDecision:
    """Which provider was chosen, and why."""

    provider: Provider
    reason: str
    estimated_cost: float
    observed_accuracy: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider.name,
            "model": self.provider.model,
            "reason": self.reason,
            "estimated_cost": self.estimated_cost,
            "observed_accuracy": self.observed_accuracy,
        }


@dataclass(frozen=True)
class ProviderScore:
    """How one provider has performed on one question."""

    provider: str
    judged: int
    correct: int
    cost_usd: float

    @property
    def accuracy(self) -> float | None:
        return None if self.judged == 0 else self.correct / self.judged

    @property
    def mean_cost(self) -> float:
        return 0.0 if self.judged == 0 else self.cost_usd / self.judged


def score_providers(
    store: DecisionStore,
    question_id: str,
) -> dict[str, ProviderScore]:
    """Summarise how each provider has done on one question, where outcomes exist."""
    outcomes = {outcome.key: outcome for outcome in store.outcomes()}
    tally: dict[str, list[float]] = {}
    for decision in store.decisions(question_id=question_id):
        entry = tally.setdefault(decision.provider, [0.0, 0.0, 0.0])
        entry[2] += decision.cost_usd
        outcome = outcomes.get(decision.key)
        if outcome is not None:
            entry[0] += 1
            if decision.answer == outcome.label:
                entry[1] += 1
    return {
        name: ProviderScore(provider=name, judged=int(v[0]), correct=int(v[1]), cost_usd=v[2])
        for name, v in tally.items()
    }


class Router:
    """Picks a provider for each question.

    ``min_accuracy`` is the floor a provider must clear on a question before it is
    preferred over a pricier alternative. A provider with no recorded outcomes is
    not penalised, since absence of evidence is not evidence of a problem; it
    competes on price until the store says otherwise.
    """

    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        store: DecisionStore | None = None,
        budget: Budget | None = None,
        min_accuracy: float = 0.5,
        preferences: Mapping[str, str] | None = None,
        fallback_order: Sequence[str] | None = None,
    ) -> None:
        if not providers:
            raise MetajevError("a router needs at least one provider")
        self.providers = list(providers)
        self.store = store
        self.budget = budget
        self.min_accuracy = min_accuracy
        self.preferences = dict(preferences or {})
        self.fallback_order = list(fallback_order or [p.name for p in self.providers])
        self._by_name = {provider.name: provider for provider in self.providers}
        unknown = set(self.preferences.values()) - set(self._by_name)
        if unknown:
            raise MetajevError(f"preferences name unknown providers: {sorted(unknown)}")

    def route(self, state: str, question: Question) -> RouteDecision:
        """Choose a provider for this question about this state."""
        preferred = self.preferences.get(question.id)
        if preferred is not None:
            provider = self._by_name[preferred]
            estimate = provider.estimate_cost(state, question)
            return RouteDecision(provider, f"pinned for question {question.id[:12]}", estimate, None)

        scores = score_providers(self.store, question.id) if self.store is not None else {}
        candidates: list[tuple[float, int, Provider, str, float | None]] = []
        for index, provider in enumerate(self.providers):
            estimate = provider.estimate_cost(state, question)
            allowed, note = provider.health()
            if not allowed:
                continue
            if self.budget is not None and self.budget.would_exceed(estimate):
                continue
            score = scores.get(provider.name)
            accuracy = score.accuracy if score is not None else None
            if accuracy is not None and score is not None and score.judged >= 5:
                if accuracy < self.min_accuracy:
                    continue
            rank = self.fallback_order.index(provider.name) if provider.name in self.fallback_order else index
            candidates.append((estimate, rank, provider, note, accuracy))

        if not candidates:
            raise MetajevError(
                f"no provider can answer this question within budget; "
                f"considered {[p.name for p in self.providers]}"
            )

        candidates.sort(key=lambda item: (item[0], item[1]))
        estimate, _, provider, note, accuracy = candidates[0]
        if accuracy is None:
            reason = f"cheapest available at {estimate:.6f} USD, no outcomes recorded yet"
        else:
            reason = f"cheapest available clearing {self.min_accuracy:.2f}, observed {accuracy:.3f}"
        return RouteDecision(provider, reason, estimate, accuracy)

    def decide(self, state: str, question: Question) -> tuple[Decision, RouteDecision]:
        """Route, call, charge the budget, and record the decision.

        A store is required because recording the result is the point; routing
        without keeping the answer would throw away the evidence the next routing
        decision depends on.
        """
        if self.store is None:
            raise MetajevError("decide() needs a store to record into")
        route = self.route(state, question)
        if self.budget is not None:
            self.budget.check(route.estimated_cost)
        decision = route.provider.decide(state, question)
        if self.budget is not None:
            self.budget.charge(decision.cost_usd)
        self.store.put(decision)
        return decision, route

    def describe(self) -> list[dict[str, object]]:
        """Return one row per provider, for the CLI."""
        rows: list[dict[str, object]] = []
        for provider in self.providers:
            allowed, note = provider.health()
            described = provider.describe()
            described["healthy"] = allowed
            described["note"] = note
            rows.append(described)
        return rows
