"""The surface most callers use: ask a question, get an action, keep the receipt.

A client ties together a provider, a store, and a policy, and adds the one
behaviour that makes the separation pay off. Before it calls a model it derives the
decision key and looks in the store. A hit means the question has been asked of
this model about this state before, so the answer is already recorded and no call
is made. The stored distribution is then resolved under whatever policy is current,
which may be a different policy than the one in force when the call was made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .budget import Budget
from .calibrate import Calibration, summarise, observations_from_store
from .errors import MetajevError
from .ledger import Ledger
from .policy import Policy, Resolution, three_band_policy
from .providers.base import Provider
from .replay import ReplayReport, replay
from .router import RouteDecision, Router
from .store import DecisionStore, StoreStats
from .types import Decision, Question, decision_key, state_hash


@dataclass(frozen=True)
class Answer:
    """A resolution, the decision behind it, and whether a model was called."""

    resolution: Resolution
    decision: Decision
    cached: bool
    route: RouteDecision | None = None

    @property
    def action(self):
        """Shorthand for the action the policy chose."""
        return self.resolution.action

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.resolution.action.value,
            "answer": self.decision.answer,
            "confidence": self.decision.confidence,
            "decision_key": self.decision.key,
            "provider": self.decision.provider,
            "model": self.decision.model,
            "cached": self.cached,
            "reason": self.resolution.reason,
        }


class Client:
    """Ask typed questions, act on the answers, and never lose the record.

    Pass either a single ``provider`` or a list of ``providers`` to route across.
    Passing a list builds a :class:`~metajev.router.Router`, which consults the
    store for observed accuracy before choosing.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        providers: Sequence[Provider] | None = None,
        store: DecisionStore | str | None = None,
        policy: Policy | None = None,
        budget: Budget | None = None,
        ledger: Ledger | str | None = None,
        min_accuracy: float = 0.5,
    ) -> None:
        if provider is None and not providers:
            raise MetajevError("a client needs a provider or a list of providers")
        if store is None:
            self.store = DecisionStore(":memory:")
            self._owns_store = True
        elif isinstance(store, str):
            self.store = DecisionStore(store)
            self._owns_store = True
        else:
            self.store = store
            self._owns_store = False

        if ledger is None:
            self.ledger: Ledger | None = None
        elif isinstance(ledger, str):
            self.ledger = Ledger(ledger)
        else:
            self.ledger = ledger

        self.budget = budget or Budget()
        self.policy = policy or three_band_policy()
        self.provider = provider
        self.router: Router | None = None
        if providers:
            self.router = Router(
                providers, store=self.store, budget=self.budget, min_accuracy=min_accuracy
            )

    # -- asking ------------------------------------------------------------

    def decide(self, state: str, question: Question) -> Answer:
        """Answer a question about a state, calling a model only on a cache miss."""
        if self.router is not None:
            return self._decide_routed(state, question)
        if self.provider is None:
            raise MetajevError("no provider configured")
        return self._decide_with(self.provider, state, question)

    def _decide_with(
        self, provider: Provider, state: str, question: Question
    ) -> Answer:
        key = decision_key(state_hash(state), question, provider.name, provider.model)
        cached = self.store.get(key)
        if cached is not None:
            resolution = self.policy.resolve(cached)
            return Answer(resolution=resolution, decision=cached, cached=True)

        estimate = provider.estimate_cost(state, question)
        self.budget.check(estimate)
        decision = provider.decide(state, question)
        self.budget.charge(decision.cost_usd)
        self.store.put(decision)
        resolution = self.policy.resolve(decision)
        if self.ledger is not None:
            self.ledger.record_decision(decision, resolution)
        return Answer(resolution=resolution, decision=decision, cached=False)

    def _decide_routed(self, state: str, question: Question) -> Answer:
        decision, route = self.router.decide(state, question)  # type: ignore[union-attr]
        resolution = self.policy.resolve(decision)
        if self.ledger is not None:
            self.ledger.record_decision(decision, resolution)
        return Answer(
            resolution=resolution, decision=decision, cached=False, route=route
        )

    def decide_many(
        self, pairs: Iterable[tuple[str, Question]]
    ) -> list[Answer]:
        """Answer several questions in order."""
        return [self.decide(state, question) for state, question in pairs]

    # -- feedback ----------------------------------------------------------

    def record_outcome(
        self, answer: Answer | Decision | str, label: str, *, note: str = ""
    ) -> None:
        """Record the truth for a decision, so calibration and routing can use it."""
        if isinstance(answer, Answer):
            key = answer.decision.key
        elif isinstance(answer, Decision):
            key = answer.key
        else:
            key = answer
        self.store.record_outcome(key, label, note=note)
        if self.ledger is not None:
            self.ledger.record_outcome(key, label, note=note)

    def record_outcomes(self, pairs: Iterable[tuple[str, str]]) -> int:
        """Record several outcomes given ``(decision_key, label)`` pairs."""
        count = 0
        for key, label in pairs:
            self.record_outcome(key, label)
            count += 1
        return count

    # -- reading history ---------------------------------------------------

    def replay(self, policy: Policy, *, baseline: Policy | None = None) -> ReplayReport:
        """Re-resolve every stored decision under ``policy``, calling no model."""
        return replay(self.store, policy, baseline=baseline or self.policy)

    def compare(self, policies: Sequence[Policy]) -> list[ReplayReport]:
        """Replay several policies over the same history."""
        return [self.replay(policy) for policy in policies]

    def calibration(self, *, on: str = "confidence", bins: int = 10) -> Calibration:
        """Summarise how well the recorded probabilities match the recorded outcomes."""
        observations, _ = observations_from_store(self.store, on=on)
        return summarise(observations, name=on, bins=bins)

    def stats(self) -> StoreStats:
        """Summarise what has been decided and how much it cost."""
        return self.store.stats()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the store when this client opened it."""
        if self._owns_store:
            self.store.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
