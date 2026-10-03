"""Choosing a provider by price, health, and observed accuracy."""

from __future__ import annotations

import pytest

from metajev.budget import Budget
from metajev.router import Router, score_providers
from metajev.store import DecisionStore
from metajev.types import Decision, Question, QuestionType

from conftest import CountingProvider, FailingProvider


def maker(name: str, cost: float) -> CountingProvider:
    provider = CountingProvider(cost_usd=cost)
    provider.name = name
    provider.model = f"{name}-1"
    return provider


def a_question() -> Question:
    return Question.noul("Is this urgent?")


def store_with(
    entries: list[tuple[str, str, str]],
) -> DecisionStore:
    """Build a store from ``(provider, answer, truth)`` triples for one question."""
    store = DecisionStore(":memory:")
    question = a_question()
    for index, (provider, answer, truth) in enumerate(entries):
        state = f"s{index}"
        key = f"k{index}"
        store.put(
            Decision(
                key=key,
                question_id=question.id,
                question_type=QuestionType.NOUL,
                state_hash=state,
                provider=provider,
                model=f"{provider}-1",
                answer=answer,
                confidence=0.9,
                distribution=(("no", 0.1), ("yes", 0.9)),
            )
        )
        store.record_outcome(key, truth)
    return store


def test_the_cheapest_healthy_provider_wins() -> None:
    cheap = maker("cheap", 0.00001)
    pricey = maker("pricey", 0.001)
    router = Router([pricey, cheap], fallback_order=["pricey", "cheap"])
    route = router.route("state", a_question())
    assert route.provider.name == "cheap"


def test_an_unhealthy_provider_is_skipped() -> None:
    router = Router([FailingProvider(), maker("working", 0.001)])
    route = router.route("state", a_question())
    assert route.provider.name == "working"


def test_a_pinned_question_goes_to_its_provider() -> None:
    cheap = maker("cheap", 0.0)
    pricey = maker("pricey", 1.0)
    question = a_question()
    router = Router([cheap, pricey], preferences={question.id: "pricey"})
    route = router.route("state", question)
    assert route.provider.name == "pricey"
    assert "pinned" in route.reason


def test_a_provider_below_the_accuracy_floor_is_dropped() -> None:
    """A cheap provider that keeps getting it wrong stops being cheap."""
    question = a_question()
    store = store_with([("wrong-cheap", "yes", "no") for _ in range(8)])
    cheap = maker("wrong-cheap", 0.0)
    good = maker("good", 0.01)
    router = Router([cheap, good], store=store, min_accuracy=0.5)
    route = router.route("state", question)
    assert route.provider.name == "good"


def test_a_provider_with_no_outcomes_is_not_penalised() -> None:
    """Absence of evidence is not evidence of a problem."""
    question = a_question()
    store = store_with([("veteran", "yes", "yes") for _ in range(8)])
    new = maker("new", 0.0)
    veteran = maker("veteran", 0.01)
    router = Router([new, veteran], store=store, min_accuracy=0.5)
    route = router.route("state", question)
    assert route.provider.name == "new"
    assert route.observed_accuracy is None


def test_a_budget_that_cannot_afford_anything_is_reported() -> None:
    router = Router([maker("pricey", 1.0)], budget=Budget(usd_limit=0.001))
    with pytest.raises(Exception, match="no provider can answer"):
        router.route("state", a_question())


def test_a_router_without_providers_is_rejected() -> None:
    with pytest.raises(Exception, match="at least one provider"):
        Router([])


def test_unknown_pinned_provider_is_rejected() -> None:
    with pytest.raises(Exception, match="unknown providers"):
        Router([maker("a", 0.0)], preferences={"q": "ghost"})


def test_decide_charges_the_budget_and_records_the_decision() -> None:
    budget = Budget(usd_limit=1.0)
    store = DecisionStore(":memory:")
    router = Router([maker("solo", 0.25)], store=store, budget=budget)
    decision, route = router.decide("state", a_question())
    assert route.provider.name == "solo"
    assert budget.spent_usd == pytest.approx(0.25)
    assert budget.calls == 1
    assert store.get(decision.key) is not None


def test_decide_without_a_store_is_rejected() -> None:
    router = Router([maker("solo", 0.0)])
    with pytest.raises(Exception, match="needs a store"):
        router.decide("state", a_question())


def test_score_providers_counts_accuracy_and_cost() -> None:
    store = store_with(
        [("a", "yes", "yes"), ("a", "yes", "no"), ("b", "yes", "yes"), ("b", "yes", "yes")]
    )
    scores = score_providers(store, a_question().id)
    assert scores["a"].judged == 2
    assert scores["a"].correct == 1
    assert scores["a"].accuracy == pytest.approx(0.5)
    assert scores["b"].accuracy == pytest.approx(1.0)


def test_describe_lists_each_provider_with_its_health() -> None:
    rows = Router([maker("ok", 0.0), FailingProvider()]).describe()
    by_name = {row["name"]: row for row in rows}
    assert by_name["ok"]["healthy"] is True
    assert by_name["failing"]["healthy"] is False
