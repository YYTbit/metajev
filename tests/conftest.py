"""Shared fixtures and a provider that counts how often it is called.

The counting provider is the instrument behind the central claim of this package.
If changing a policy calls a model, the claim is false, and a test that merely
compares two reports would not notice.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from metajev.providers.base import Provider, ProviderAnswer
from metajev.types import Question


class CountingProvider(Provider):
    """A deterministic provider that records how many times it was asked."""

    name = "counting"
    model = "counting-1"

    def __init__(self, *, cost_usd: float = 0.0, confidence: float | None = None) -> None:
        super().__init__()
        self.calls = 0
        self.cost_usd = cost_usd
        self.confidence = confidence

    def _answer(self, state: str, question: Question) -> ProviderAnswer:
        self.calls += 1
        labels = question.labels
        digest = hashlib.sha256(f"{state}|{question.id}".encode()).digest()
        lead = labels[digest[0] % len(labels)]
        confidence = self.confidence
        if confidence is None:
            confidence = 0.5 + 0.499 * ((digest[1] / 255.0))
        rest = (1.0 - confidence) / max(1, len(labels) - 1)
        distribution = {
            label: (confidence if label == lead else rest) for label in labels
        }
        return ProviderAnswer(
            answer=lead,
            distribution=tuple(distribution.items()),
            cost_usd=self.cost_usd,
        )

    def estimate_cost(self, state: str, question: Question) -> float:
        return self.cost_usd


class FailingProvider(Provider):
    """A provider whose endpoint is down, for exercising fallback paths."""

    name = "failing"
    model = "failing-1"

    def _answer(self, state: str, question: Question) -> ProviderAnswer:
        raise RuntimeError("provider is unreachable")

    def health(self) -> tuple[bool, str]:
        return False, "deliberately unhealthy"


@pytest.fixture()
def noul_question() -> Question:
    return Question.noul(
        "Does this need a person?",
        yes="a human should read this before anything happens",
        tags=("triage",),
    )


@pytest.fixture()
def choice_question() -> Question:
    return Question.choice("Which queue?", ("billing", "support", "sales"), tags=("route",))


@pytest.fixture()
def score_question() -> Question:
    return Question.score("How urgent?", ("low", "normal", "high"), tags=("urgency",))


@pytest.fixture()
def counting() -> CountingProvider:
    return CountingProvider()


@pytest.fixture()
def store():
    from metajev.store import DecisionStore

    with DecisionStore(":memory:") as opened:
        yield opened


def ground_truth(text: str, threshold: int = 30) -> str:
    """A label defined independently of any provider, for calibration fixtures."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    return "yes" if int(digest, 16) % 100 < threshold else "no"


def populate(client: Any, count: int = 120, question: Question | None = None) -> None:
    """Ask a client about synthetic states and record the truth for each."""
    question = question or Question.noul("Does this need a person?", tags=("triage",))
    for index in range(count):
        state = f"ticket-{index}"
        answer = client.decide(state, question)
        client.record_outcome(answer, ground_truth(state))
