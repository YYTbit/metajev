"""An MCP server exposing a decision store to any MCP client.

The tools here are the package's own operations, so an agent that talks to this
server gets the same separation a program does: it can ask a typed question,
record what actually happened, and then try a different threshold over the whole
history without paying for another model call.

Configuration is by environment variable, since an MCP client launches the server
as a subprocess and has nowhere else to put it.

    METAJEV_STORE      path to the decision store, default ~/.metajev/decisions.db
    METAJEV_PROVIDER   provider spec, default typesafe
    METAJEV_POLICY     accept,review boundaries, default 0.85,0.55
    JEV_API_KEY        credential for the endpoint
    JEV_API_URL        endpoint, default https://api.typesafe.ai/v1/systemone

Run it directly to check the wiring before adding it to a client:

    metajev-mcp --check
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .budget import Budget
from .calibrate import observations_from_store, summarise, threshold_table
from .client import Client
from .errors import MetajevError
from .policy import three_band_policy
from .providers import build_provider
from .replay import replay
from .store import DecisionStore
from .types import Action, Question, QuestionType

INSTRUCTIONS = """\
A decision store for typed questions answered by a Jev-class model.

Ask a question with `decide` and you get back an action: accept, review, abstain,
escalate, or deny. The action comes from a policy, and the answer it came from is
recorded with the full probability distribution the model returned.

That recording is the point. `replay` applies a different policy to every question
already asked, and it calls no model, so trying a second threshold costs nothing.
`calibrate` says whether the probabilities can be read as frequencies, which is
what decides whether a threshold means anything at all.

Record what actually happened with `record_outcome` whenever you learn it. Without
outcomes, `calibrate` has nothing to measure and `replay` cannot tell you what a
policy change would have caught.
"""


@dataclass(frozen=True)
class Settings:
    """Everything the server reads from the environment."""

    store_path: str
    provider: str
    accept: float
    review: float

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = env if env is not None else dict(os.environ)
        store_path = env.get("METAJEV_STORE") or str(
            Path.home() / ".metajev" / "decisions.db"
        )
        boundaries = env.get("METAJEV_POLICY", "0.85,0.55")
        try:
            accept_text, review_text = boundaries.split(",", 1)
            accept = float(accept_text)
            review = float(review_text)
        except ValueError as exc:
            raise MetajevError(
                f"METAJEV_POLICY must look like '0.85,0.55', got {boundaries!r}"
            ) from exc
        if not 0.0 <= review < accept <= 1.0:
            raise MetajevError(
                f"METAJEV_POLICY needs 0 <= review < accept <= 1, got {boundaries!r}"
            )
        return cls(
            store_path=store_path,
            provider=env.get("METAJEV_PROVIDER", "typesafe"),
            accept=accept,
            review=review,
        )


class Runtime:
    """The operations the tools call, kept apart from the transport.

    Everything here is a plain method so it can be tested without starting a
    server or speaking the protocol.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = DecisionStore(settings.store_path)
        self.client = Client(
            provider=build_provider(settings.provider),
            store=self.store,
            policy=self.policy(settings.accept, settings.review),
            budget=Budget(),
        )

    @staticmethod
    def policy(accept: float, review: float):
        """Build the three-band policy the boundaries describe."""
        return three_band_policy(
            f"accept-{accept:g}-review-{review:g}", accept=accept, review=review
        )

    def close(self) -> None:
        self.client.close()

    # -- operations --------------------------------------------------------

    def decide(
        self,
        state: str,
        question: str,
        kind: str = "noul",
        options: Sequence[str] | None = None,
        levels: Sequence[str] | None = None,
        yes: str | None = None,
    ) -> dict[str, Any]:
        """Ask one typed question and return the action the policy chose."""
        try:
            question_type = QuestionType(kind)
        except ValueError as exc:
            raise MetajevError(
                f"unknown question type {kind!r}; use noul, choice, or score"
            ) from exc

        if question_type is QuestionType.CHOICE:
            if not options:
                raise MetajevError("a choice question needs options")
            built = Question.choice(question, options)
        elif question_type is QuestionType.SCORE:
            if not levels:
                raise MetajevError("a score question needs levels")
            built = Question.score(question, levels)
        else:
            built = Question.noul(question, yes=yes)

        answer = self.client.decide(state, built)
        payload = answer.to_dict()
        payload["distribution"] = {
            label: round(probability, 6)
            for label, probability in answer.decision.distribution
        }
        payload["available_actions"] = [action.value for action in Action]
        return payload

    def record_outcome(self, decision_key: str, label: str, note: str = "") -> dict[str, Any]:
        """Record the truth for a decision, once it is known."""
        decision = self.store.get(decision_key)
        if decision is None:
            raise MetajevError(f"no decision stored under key {decision_key}")
        if label not in {name for name, _ in decision.distribution}:
            known = [name for name, _ in decision.distribution]
            raise MetajevError(f"label {label!r} is not one of {known}")
        self.store.record_outcome(decision_key, label, note=note)
        judged = self.store.outcome_count()
        return {
            "recorded": label,
            "decision_key": decision_key,
            "outcomes_known": judged,
            "decisions_recorded": self.store.count(),
            "note": note,
        }

    def replay(
        self,
        accept: float | None = None,
        review: float | None = None,
        examples: int = 5,
    ) -> dict[str, Any]:
        """Re-resolve every recorded decision under different boundaries.

        This calls no model. The boundaries only decide which action each already
        recorded distribution maps to.
        """
        accept = self.settings.accept if accept is None else accept
        review = self.settings.review if review is None else review
        policy = self.policy(accept, review)
        report = replay(self.store, policy, baseline=self.client.policy, examples=examples)
        return {
            "policy": report.policy,
            "fingerprint": report.fingerprint,
            "decisions": report.total,
            "with_outcomes": report.judged,
            "model_calls": 0,
            "buckets": [
                {
                    "action": bucket.action.value,
                    "count": bucket.count,
                    "share": round(bucket.share, 4),
                    "judged": bucket.judged,
                    "correct": bucket.correct,
                    "accuracy": None if bucket.accuracy is None else round(bucket.accuracy, 4),
                }
                for bucket in report.buckets
            ],
            "precision_among_accepted": (
                None if report.accept_precision is None else round(report.accept_precision, 4)
            ),
            "against_baseline": (
                None
                if report.baseline is None
                else {
                    "baseline": report.baseline,
                    "actions_changed": report.flips,
                    "errors_moved_off_accept": report.errors_diverted,
                    "errors_newly_admitted": report.errors_admitted,
                    "correct_answers_diverted": report.correct_diverted,
                }
            ),
            "summary": report.render(),
        }

    def calibrate(self, on: str = "confidence", bins: int = 10) -> dict[str, Any]:
        """Report whether the recorded probabilities match the recorded outcomes."""
        observations, unjudged = observations_from_store(self.store, on=on)
        if not observations:
            return {
                "observations": 0,
                "note": (
                    "no decision has a recorded outcome yet; call record_outcome "
                    "when you learn what actually happened, or calibration has "
                    "nothing to measure"
                ),
            }
        summary = summarise(observations, name=on, bins=bins)
        return {
            "target": on,
            "observations": summary.n,
            "decisions_without_outcome": unjudged,
            "brier": round(summary.brier, 4),
            "expected_calibration_error": round(summary.ece, 4),
            "mean_predicted": round(summary.mean_predicted, 4),
            "observed_accuracy": round(summary.accuracy, 4),
            "gap": round(summary.overconfidence, 4),
            "reading": (
                "overconfident"
                if summary.overconfidence > 0.02
                else "underconfident"
                if summary.overconfidence < -0.02
                else "calibrated"
            ),
            "reliability": [
                {
                    "lo": bucket.lo,
                    "hi": bucket.hi,
                    "count": bucket.count,
                    "predicted": round(bucket.mean_predicted, 4),
                    "actual": round(bucket.accuracy, 4),
                }
                for bucket in summary.bins
            ],
            "summary": summary.render(),
        }

    def boundary_table(self, steps: int = 10) -> dict[str, Any]:
        """Show what each candidate boundary would admit, from recorded outcomes."""
        observations, _ = observations_from_store(self.store, on="confidence")
        if not observations:
            return {"observations": 0, "note": "no outcomes recorded yet"}
        thresholds = [index / steps for index in range(steps + 1)]
        return {
            "observations": len(observations),
            "model_calls": 0,
            "rows": threshold_table(observations, thresholds),
        }

    def store_summary(self) -> dict[str, Any]:
        """Describe what the store holds."""
        stats = self.store.stats()
        return {
            "store": self.settings.store_path,
            "provider": self.settings.provider,
            "policy": {"accept": self.settings.accept, "review": self.settings.review},
            "decisions": stats.decisions,
            "outcomes": stats.outcomes,
            "spent_usd": round(stats.total_cost_usd, 6),
            "mean_confidence": round(stats.mean_confidence, 4),
            "by_provider": dict(stats.by_provider),
            "by_question_type": dict(stats.by_question_type),
        }


def build_server(runtime: Runtime | None = None):
    """Build the MCP server. Imports the SDK here so the package works without it."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MetajevError(
            "the MCP server needs the mcp package; install it with "
            "pip install 'metajev[mcp]'"
        ) from exc

    settings = Settings.from_env()
    runtime = runtime or Runtime(settings)
    server = FastMCP("metajev", instructions=INSTRUCTIONS)

    @server.tool(
        description=(
            "Ask a typed question about a state and get an action back. The answer "
            "and the full distribution are recorded, so a later policy change "
            "costs no model call. type is noul (yes/no), choice, or score."
        )
    )
    def decide(
        state: str,
        question: str,
        type: str = "noul",
        options: list[str] | None = None,
        levels: list[str] | None = None,
        yes: str | None = None,
    ) -> dict[str, Any]:
        return runtime.decide(state, question, type, options, levels, yes)

    @server.tool(
        description=(
            "Record what actually happened for a decision, once it is known. "
            "Without outcomes, calibration cannot measure anything and replay "
            "cannot say what a policy change would have caught."
        )
    )
    def record_outcome(decision_key: str, label: str, note: str = "") -> dict[str, Any]:
        return runtime.record_outcome(decision_key, label, note)

    @server.tool(
        description=(
            "Re-resolve every recorded decision under different accept and review "
            "boundaries. Makes no model calls, so a second threshold is free. "
            "Reports what moved, what errors it caught, and what right answers it "
            "diverted."
        )
    )
    def replay(accept: float | None = None, review: float | None = None,
               examples: int = 5) -> dict[str, Any]:
        return runtime.replay(accept, review, examples)

    @server.tool(
        description=(
            "Measure whether the recorded probabilities can be read as frequencies. "
            "A gap above zero means the model reads more confident than it has "
            "earned, which is what makes a threshold mean less than it looks."
        )
    )
    def calibrate(on: str = "confidence", bins: int = 10) -> dict[str, Any]:
        return runtime.calibrate(on, bins)

    @server.tool(
        description=(
            "Show what each candidate boundary would admit and what it would turn "
            "away, computed from recorded outcomes. Makes no model calls."
        )
    )
    def boundary_table(steps: int = 10) -> dict[str, Any]:
        return runtime.boundary_table(steps)

    @server.tool(description="Describe the store: what is recorded, what it cost, and the active policy.")
    def store_summary() -> dict[str, Any]:
        return runtime.store_summary()

    server._metajev_runtime = runtime  # type: ignore[attr-defined]
    return server


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. `--check` verifies the wiring and exits."""
    args = list(argv if argv is not None else sys.argv[1:])
    settings = Settings.from_env()

    if "--check" in args:
        print(f"store     {settings.store_path}")
        print(f"provider  {settings.provider}")
        print(f"policy    accept {settings.accept:g}, review {settings.review:g}")
        runtime = Runtime(settings)
        try:
            summary = runtime.store_summary()
            print(f"decisions {summary['decisions']}")
            print(f"outcomes  {summary['outcomes']}")
            provider_ok, note = runtime.client.provider.health()
            print(f"provider  {'ok' if provider_ok else 'warn'}: {note}")
        finally:
            runtime.close()
        print("ready")
        return 0

    try:
        server = build_server()
    except MetajevError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
