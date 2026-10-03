"""Replaying a recorded history under a different policy."""

from __future__ import annotations

import pytest

from metajev.client import Client
from metajev.policy import three_band_policy
from metajev.replay import replay, sweep
from metajev.store import DecisionStore
from metajev.types import Action, Decision, Question, QuestionType

from conftest import CountingProvider, ground_truth


def a_decision(key: str, confidence: float, provider: str = "p") -> Decision:
    return Decision(
        key=key,
        question_id="q1",
        question_type=QuestionType.NOUL,
        state_hash="s",
        provider=provider,
        model="m1",
        answer="yes" if confidence >= 0.5 else "no",
        confidence=confidence,
        distribution=(("no", 1.0 - confidence), ("yes", confidence)),
    )


def build_store(entries: list[tuple[str, float, str | None]]) -> DecisionStore:
    """Build a store from ``(key, confidence, true label or None)`` triples."""
    store = DecisionStore(":memory:")
    for key, confidence, truth in entries:
        store.put(a_decision(key, confidence))
        if truth is not None:
            store.record_outcome(key, truth)
    return store


def test_replay_counts_every_action() -> None:
    store = build_store([("a", 0.95, None), ("b", 0.70, None), ("c", 0.20, None)])
    report = replay(store, three_band_policy(accept=0.85, review=0.55))
    assert report.total == 3
    assert report.bucket(Action.ACCEPT).count == 1
    assert report.bucket(Action.REVIEW).count == 1
    assert report.bucket(Action.ESCALATE).count == 1
    store.close()


def test_replay_reports_shares() -> None:
    store = build_store([("a", 0.95, None), ("b", 0.95, None), ("c", 0.20, None)])
    report = replay(store, three_band_policy(accept=0.85, review=0.55))
    assert report.bucket(Action.ACCEPT).share == pytest.approx(2 / 3)
    store.close()


def test_accept_precision_uses_only_judged_decisions() -> None:
    store = build_store([("a", 0.95, "yes"), ("b", 0.95, "no"), ("c", 0.95, None)])
    report = replay(store, three_band_policy(accept=0.85, review=0.55))
    assert report.judged == 2
    assert report.accept_precision == pytest.approx(0.5)
    store.close()


def test_tightening_a_boundary_reports_both_sides() -> None:
    """The report must state what the change caught and what it cost."""
    store = build_store(
        [
            ("wrong-confident", 0.90, "no"),
            ("right-confident", 0.90, "yes"),
            ("wrong-sure", 0.99, "no"),
        ]
    )
    report = replay(
        store,
        three_band_policy("tight", accept=0.95, review=0.55),
        baseline=three_band_policy("loose", accept=0.85, review=0.55),
    )
    assert report.flips == 2
    assert report.errors_diverted == 1
    assert report.correct_diverted == 1
    store.close()


def test_a_boundary_change_calls_no_model() -> None:
    """The claim behind the package, stated as a test."""
    provider = CountingProvider()
    client = Client(provider=provider, store=DecisionStore(":memory:"))
    for index in range(25):
        client.decide(f"state-{index}", Question.noul("Is this urgent?"))
    calls_after_deciding = provider.calls
    assert calls_after_deciding == 25

    for accept in (0.99, 0.95, 0.9, 0.8, 0.7, 0.6):
        client.replay(three_band_policy(accept=accept, review=0.3))

    assert provider.calls == calls_after_deciding
    client.close()


def test_replaying_a_larger_population_also_calls_no_model() -> None:
    provider = CountingProvider()
    client = Client(provider=provider, store=DecisionStore(":memory:"))
    for index in range(200):
        state = f"ticket-{index}"
        answer = client.decide(state, Question.noul("Is this urgent?"))
        client.record_outcome(answer, ground_truth(state))
    calls = provider.calls
    report = client.replay(three_band_policy("strict", accept=0.97, review=0.5))
    assert provider.calls == calls
    assert report.judged == 200
    client.close()


def test_sweep_returns_one_report_per_value() -> None:
    store = build_store([("a", 0.95, "yes"), ("b", 0.70, "no"), ("c", 0.20, "no")])
    rows = sweep(store, lambda v: three_band_policy(f"a{v}", accept=v, review=0.1), [0.6, 0.8, 0.95])
    assert [value for value, _ in rows] == [0.6, 0.8, 0.95]
    assert len({report.policy for _, report in rows}) == 3
    store.close()


def test_render_produces_a_table() -> None:
    store = build_store([("a", 0.95, "yes"), ("b", 0.70, "no")])
    text = replay(store, three_band_policy(accept=0.85, review=0.55)).render()
    assert "action" in text
    assert "accept" in text
    assert "precision among accepted decisions" in text
    store.close()


def test_render_names_the_baseline_when_given_one() -> None:
    store = build_store([("a", 0.95, "yes")])
    text = replay(
        store,
        three_band_policy("tight", accept=0.99, review=0.5),
        baseline=three_band_policy("loose", accept=0.5, review=0.2),
    ).render()
    assert "loose" in text
    assert "errors moved off accept" in text
    store.close()


def test_examples_are_capped_by_the_requested_count() -> None:
    store = build_store([(f"k{i}", 0.9 + i * 0.001, "yes") for i in range(10)])
    report = replay(
        store,
        three_band_policy("tight", accept=0.95, review=0.5),
        baseline=three_band_policy("loose", accept=0.85, review=0.5),
        examples=3,
    )
    assert len(report.examples) == 3
    store.close()
