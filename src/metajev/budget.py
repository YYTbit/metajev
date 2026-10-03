"""Spending limits for a run.

Decision models are cheap per call, which is exactly why the total creeps up. A
budget is checked before a call and charged after it, so a run stops at the limit
rather than discovering it in a bill. Both a dollar limit and a call limit are
supported, and either can be left unset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import BudgetExceeded


@dataclass(frozen=True)
class BudgetSnapshot:
    """What a budget looks like at one moment."""

    spent_usd: float
    calls: int
    usd_limit: float | None
    call_limit: int | None

    @property
    def remaining_usd(self) -> float | None:
        return None if self.usd_limit is None else max(0.0, self.usd_limit - self.spent_usd)

    @property
    def remaining_calls(self) -> int | None:
        return None if self.call_limit is None else max(0, self.call_limit - self.calls)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "spent_usd": self.spent_usd,
            "calls": self.calls,
            "usd_limit": self.usd_limit,
            "call_limit": self.call_limit,
            "remaining_usd": self.remaining_usd,
            "remaining_calls": self.remaining_calls,
        }


@dataclass
class Budget:
    """A running limit on spending and call count.

    ``usd_limit`` and ``call_limit`` are both optional. Leaving both unset makes
    the budget a pure accounting object that records what was spent without ever
    refusing a call.
    """

    usd_limit: float | None = None
    call_limit: int | None = None
    spent_usd: float = 0.0
    calls: int = 0

    def check(self, estimated_cost: float = 0.0) -> None:
        """Raise :class:`~metajev.errors.BudgetExceeded` when a call would pass a limit."""
        if self.usd_limit is not None and self.spent_usd + estimated_cost > self.usd_limit + 1e-12:
            raise BudgetExceeded(
                f"call would spend {self.spent_usd + estimated_cost:.6f} USD, "
                f"limit is {self.usd_limit:.6f}"
            )
        if self.call_limit is not None and self.calls + 1 > self.call_limit:
            raise BudgetExceeded(f"call {self.calls + 1} exceeds the limit of {self.call_limit}")

    def charge(self, cost_usd: float) -> None:
        """Record what a completed call cost."""
        self.spent_usd += max(0.0, cost_usd)
        self.calls += 1

    def would_exceed(self, estimated_cost: float = 0.0) -> bool:
        """Return whether a call at this price would pass a limit."""
        try:
            self.check(estimated_cost)
        except BudgetExceeded:
            return True
        return False

    def snapshot(self) -> BudgetSnapshot:
        """Return the current state as a plain record."""
        return BudgetSnapshot(
            spent_usd=self.spent_usd,
            calls=self.calls,
            usd_limit=self.usd_limit,
            call_limit=self.call_limit,
        )

    def render(self) -> str:
        """Render the budget as one line of text."""
        parts = [f"{self.calls} calls", f"{self.spent_usd:.6f} USD"]
        if self.usd_limit is not None:
            parts.append(f"limit {self.usd_limit:.6f} USD")
        if self.call_limit is not None:
            parts.append(f"limit {self.call_limit} calls")
        return ", ".join(parts)
