"""The interactive report: embedded data, and agreement with the library."""

from __future__ import annotations

import json
import re

import pytest

from metajev.policy import three_band_policy
from metajev.report import collect, render_html
from metajev.store import DecisionStore
from metajev.types import Action, Decision, QuestionType

from conftest import ground_truth


def a_decision(key: str, confidence: float, answer: str = "yes") -> Decision:
    return Decision(
        key=key,
        question_id="q1",
        question_type=QuestionType.NOUL,
        state_hash="s",
        provider="p",
        model="m",
        answer=answer,
        confidence=confidence,
        distribution=(("no", 1.0 - confidence), ("yes", confidence)),
    )


def build_store(rows: list[tuple[str, float, str | None]]) -> DecisionStore:
    store = DecisionStore(":memory:")
    for key, confidence, truth in rows:
        store.put(a_decision(key, confidence))
        if truth is not None:
            store.record_outcome(key, truth)
    return store


def embedded(html: str) -> dict:
    """Read the JSON payload back out of a rendered page."""
    match = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert match, "the page carries no data payload"
    return json.loads(match.group(1))


def test_the_page_is_self_contained() -> None:
    store = build_store([("a", 0.9, "yes")])
    html = render_html(store, baseline=three_band_policy(accept=0.85, review=0.55))
    assert "<script>" in html
    assert "<style>" in html
    # No external fetches of any kind.
    assert "src=\"http" not in html
    assert "href=\"http" not in html
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    store.close()


def test_no_placeholder_survives_rendering() -> None:
    store = build_store([("a", 0.9, "yes")])
    html = render_html(store, baseline=three_band_policy(accept=0.85, review=0.55))
    assert not re.search(r"__[A-Z_]+__", html)
    store.close()


def test_every_decision_is_embedded() -> None:
    store = build_store([(f"k{i}", 0.5 + i / 100, None) for i in range(20)])
    payload = embedded(render_html(store))
    assert len(payload["rows"]) == 20
    store.close()


def test_a_judged_decision_carries_its_outcome() -> None:
    store = build_store([("a", 0.9, "yes"), ("b", 0.9, "no")])
    payload = embedded(render_html(store))
    outcomes = sorted(row[1] for row in payload["rows"])
    assert outcomes == [0, 1]
    store.close()


def test_an_unjudged_decision_has_no_outcome_field() -> None:
    store = build_store([("a", 0.9, None)])
    payload = embedded(render_html(store))
    assert len(payload["rows"][0]) == 1
    store.close()


def test_no_state_text_reaches_the_page() -> None:
    """A decision stores a state hash, never the state, so a report cannot leak one."""
    store = DecisionStore(":memory:")
    question = metajev_question()
    from metajev.client import Client
    from metajev.providers import MockProvider

    secret = "PROJECT-HELIOTROPE-CONFIDENTIAL"
    with Client(provider=MockProvider(), store=store) as client:
        answer = client.decide(secret, question)
        client.record_outcome(answer, "yes")
    html = render_html(store, baseline=three_band_policy(accept=0.85, review=0.55))
    assert secret not in html
    store.close()


def metajev_question():
    from metajev.types import Question

    return Question.noul("Is this urgent?")


def test_the_sweep_curve_is_monotone_in_coverage() -> None:
    store = build_store([(f"k{i}", i / 100, None) for i in range(1, 100)])
    payload = embedded(render_html(store))
    coverages = [point[1] for point in payload["sweep"]]
    assert coverages == sorted(coverages, reverse=True)
    store.close()


def test_the_baseline_comes_from_the_policy() -> None:
    store = build_store([("a", 0.9, "yes")])
    payload = embedded(render_html(store, baseline=three_band_policy(accept=0.72, review=0.4)))
    assert payload["baseline"] == pytest.approx(0.72)
    store.close()


def test_the_reliability_bins_match_the_summary() -> None:
    store = build_store([(f"k{i}", 0.5 + (i % 50) / 100, "yes" if i % 3 == 0 else "no")
                         for i in range(100)])
    payload = embedded(render_html(store, bins=10))
    assert len(payload["reliability"]) <= 10
    for bucket in payload["reliability"]:
        assert 0.0 <= bucket["predicted"] <= 1.0
        assert 0.0 <= bucket["actual"] <= 1.0
    store.close()


def test_the_caption_reports_the_calibration_error() -> None:
    store = build_store([(f"k{i}", 0.9, "yes" if i % 10 == 0 else "no") for i in range(100)])
    html = render_html(store)
    assert "expected calibration error" in html
    assert "more confident than it has earned" in html
    store.close()


def test_collect_agrees_with_replay() -> None:
    """The numbers the page shows and the numbers the library reports are one source."""
    from metajev.replay import replay

    store = build_store(
        [(f"k{i}", 0.5 + (i % 50) / 100, "yes" if i % 3 else "no") for i in range(120)]
    )
    payload = collect(store, baseline=three_band_policy(accept=0.85, review=0.55))

    for threshold in (0.6, 0.75, 0.9):
        policy = three_band_policy(f"a{threshold}", accept=threshold, review=0.5)
        report = replay(store, policy, baseline=three_band_policy(accept=0.85, review=0.55))
        bucket = report.bucket(Action.ACCEPT)
        expected_coverage = (bucket.count if bucket else 0) / payload.decisions
        point = min(payload.sweep, key=lambda p: abs(p[0] - threshold))
        assert point[1] == pytest.approx(expected_coverage, abs=0.02)
    store.close()


def test_the_page_renders_for_a_large_store() -> None:
    store = build_store([(f"k{i}", (i % 100) / 100, "yes" if i % 2 else "no") for i in range(2000)])
    html = render_html(store)
    assert len(embedded(html)["rows"]) == 2000
    # Still one file, still no requests.
    assert "fetch(" not in html
    store.close()
