"""Storing decisions, recording outcomes, and moving records between stores."""

from __future__ import annotations

import json

from metajev.store import DecisionStore
from metajev.types import Decision, QuestionType


def a_decision(key: str = "k1", *, confidence: float = 0.9, provider: str = "p") -> Decision:
    return Decision(
        key=key,
        question_id="q1",
        question_type=QuestionType.NOUL,
        state_hash="s1",
        provider=provider,
        model="m1",
        answer="yes",
        confidence=confidence,
        distribution=(("no", 1.0 - confidence), ("yes", confidence)),
        tags=("triage",),
        cost_usd=0.001,
    )


def test_put_then_get(store: DecisionStore) -> None:
    decision = a_decision()
    assert store.put(decision) is True
    assert store.get("k1") == decision


def test_storing_the_same_key_twice_reports_not_new(store: DecisionStore) -> None:
    store.put(a_decision())
    assert store.put(a_decision()) is False
    assert store.count() == 1


def test_get_of_a_missing_key_is_none(store: DecisionStore) -> None:
    assert store.get("absent") is None


def test_require_raises_for_a_missing_key(store: DecisionStore) -> None:
    import pytest

    with pytest.raises(Exception, match="no decision stored"):
        store.require("absent")


def test_filtering_by_provider(store: DecisionStore) -> None:
    store.put(a_decision("a", provider="one"))
    store.put(a_decision("b", provider="two"))
    store.put(a_decision("c", provider="two"))
    assert len(list(store.decisions(provider="two"))) == 2
    assert len(list(store.decisions(provider="one"))) == 1


def test_recording_and_reading_an_outcome(store: DecisionStore) -> None:
    store.put(a_decision())
    store.record_outcome("k1", "no", note="reviewed by hand")
    outcome = store.outcome_for("k1")
    assert outcome is not None
    assert outcome.label == "no"
    assert outcome.note == "reviewed by hand"


def test_a_later_outcome_replaces_an_earlier_one(store: DecisionStore) -> None:
    store.put(a_decision())
    store.record_outcome("k1", "yes")
    store.record_outcome("k1", "no")
    assert store.outcome_for("k1").label == "no"
    assert store.outcome_count() == 1


def test_judged_pairs_decisions_with_their_outcomes(store: DecisionStore) -> None:
    store.put(a_decision("a"))
    store.put(a_decision("b"))
    store.record_outcome("b", "yes")
    pairs = list(store.judged())
    assert len(pairs) == 1
    assert pairs[0][0].key == "b"
    assert pairs[0][1].label == "yes"


def test_stats_summarise_the_store(store: DecisionStore) -> None:
    store.put(a_decision("a", confidence=0.8))
    store.put(a_decision("b", confidence=0.6))
    store.record_outcome("a", "yes")
    stats = store.stats()
    assert stats.decisions == 2
    assert stats.outcomes == 1
    assert stats.mean_confidence == 0.7
    assert stats.total_cost_usd == 0.002
    assert stats.by_provider == {"p": 2}


def test_export_then_import_round_trips(tmp_path) -> None:
    path = str(tmp_path / "store.db")
    exported = str(tmp_path / "dump.jsonl")
    with DecisionStore(path) as first:
        first.put(a_decision("a"))
        first.put(a_decision("b"))
        first.record_outcome("a", "no")
        assert first.export_jsonl(exported) == 3

    with DecisionStore(":memory:") as second:
        decisions, outcomes = second.import_jsonl(exported)
        assert decisions == 2
        assert outcomes == 1
        assert second.count() == 2
        assert second.outcome_for("a").label == "no"


def test_exported_lines_are_labelled_by_kind(store: DecisionStore, tmp_path) -> None:
    store.put(a_decision("a"))
    store.record_outcome("a", "yes")
    path = str(tmp_path / "dump.jsonl")
    store.export_jsonl(path)
    kinds = [json.loads(line)["kind"] for line in open(path, encoding="utf-8") if line.strip()]
    assert kinds == ["decision", "outcome"]


def test_an_unknown_record_kind_is_rejected(store: DecisionStore, tmp_path) -> None:
    import pytest

    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"kind": "mystery", "key": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="unknown record kind"):
        store.import_jsonl(str(path))
