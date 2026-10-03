"""A deterministic offline provider, for tests, demos, and dry runs.

The mock reads nothing but its inputs, so the same state and question always
produce the same distribution. That makes it useful in three places: unit tests
that need a provider without a network, examples that must run on a fresh clone,
and dry runs that check a policy before spending anything.

Its numbers are shaped to be plausible rather than meaningful. It decides which
label to lead with from a hash, and draws a confidence from a distribution
skewed toward the middle. It knows nothing about the truth, which is what makes
it safe to ship.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..errors import ProviderResponseError
from ..types import Question
from .base import Provider, ProviderAnswer


def _stream(*parts: str) -> int:
    """Derive a stable integer from the given strings."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class MockProvider(Provider):
    """A provider that answers from a hash rather than a model.

    ``sharpness`` controls how peaked the distributions are. At 0 the distribution
    is nearly uniform and confidence sits near chance. At 6 it is sharply peaked and
    confidence sits near one. ``cost_usd`` lets an example exercise the budget code
    without a paid backend.
    """

    name = "mock"
    model = "mock-1"

    def __init__(
        self,
        *,
        name: str = "mock",
        model: str = "mock-1",
        sharpness: float = 2.5,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__(name=name, model=model)
        self.sharpness = sharpness
        self.cost_usd = cost_usd
        self.latency_ms = latency_ms

    def _answer(self, state: str, question: Question) -> ProviderAnswer:
        labels = question.labels
        if not labels:
            raise ProviderResponseError(f"question {question.id} declares no labels")

        seed = _stream(self.model, state, question.id)
        lead = labels[seed % len(labels)]

        # Draw the leading probability from a bounded uniform, so confidence lands
        # spread across the useful range instead of piling up near certainty.
        unit = (_stream(self.model, state, question.id, "confidence") % 1_000_000) / 1_000_000.0
        floor = 1.0 / len(labels)
        confidence = floor + (0.995 - floor) * unit

        remaining = 1.0 - confidence
        others = [label for label in labels if label != lead]
        distribution: dict[str, float] = {lead: confidence}
        if others:
            weights = {}
            for index, label in enumerate(others):
                raw = _stream(self.model, state, question.id, label, str(index))
                weights[label] = ((raw % 1_000_000) / 1_000_000.0 + 1e-9) ** self.sharpness
            total = sum(weights.values())
            for label, value in weights.items():
                distribution[label] = remaining * value / total

        answer = max(distribution, key=lambda label: distribution[label])

        if self.latency_ms:
            # Busy-wait so the recorded latency is real rather than declared.
            import time

            deadline = time.perf_counter() + self.latency_ms / 1000.0
            while time.perf_counter() < deadline:
                pass

        return ProviderAnswer(
            answer=answer,
            distribution=tuple(distribution.items()),
            cost_usd=self.cost_usd,
            meta={"simulated": True, "sharpness": self.sharpness},
        )

    def health(self) -> tuple[bool, str]:
        return True, "offline provider, always available"

    def describe(self) -> dict[str, Any]:
        described = super().describe()
        described.update({"sharpness": self.sharpness, "cost_usd": self.cost_usd})
        return described
