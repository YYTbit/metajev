"""Questions, decisions, and the keys that address them."""

from __future__ import annotations

import pytest

from metajev.types import (
    Decision,
    Question,
    QuestionType,
    decision_key,
    state_hash,
)


def test_question_id_ignores_tags() -> None:
    """Tags organise questions; they do not change what was asked."""
    first = Question.noul("Is this urgent?", tags=("a",))
    second = Question.noul("Is this urgent?", tags=("b",))
    assert first.id == second.id


def test_question_id_tracks_wording() -> None:
    assert Question.noul("Is this urgent?").id != Question.noul("Is this urgent").id


def test_choice_requires_two_options() -> None:
    with pytest.raises(ValueError, match="at least two options"):
        Question.choice("Pick one", ["only"])


def test_score_requires_two_levels() -> None:
    with pytest.raises(ValueError, match="at least two levels"):
        Question.score("Rate it", ["single"])


def test_options_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="distinct"):
        Question.choice("Pick one", ["a", "a"])


def test_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty text"):
        Question.noul("   ")


def test_labels_follow_the_question_type() -> None:
    assert Question.noul("q").labels == ("no", "yes")
    assert Question.choice("q", ("x", "y")).labels == ("x", "y")
    assert Question.score("q", ("low", "high")).labels == ("low", "high")


def test_decision_key_excludes_policy_inputs() -> None:
    """Two policies that disagree must still address the same recorded decision."""
    question = Question.noul("Is this urgent?")
    digest = state_hash("some state")
    first = decision_key(digest, question, "provider", "model")
    second = decision_key(digest, question, "provider", "model")
    assert first == second


def test_decision_key_separates_models() -> None:
    question = Question.noul("Is this urgent?")
    digest = state_hash("some state")
    assert decision_key(digest, question, "a", "m") != decision_key(digest, question, "b", "m")
    assert decision_key(digest, question, "a", "m") != decision_key(digest, question, "a", "n")


def test_decision_key_separates_states() -> None:
    question = Question.noul("Is this urgent?")
    assert decision_key(state_hash("one"), question, "p", "m") != decision_key(
        state_hash("two"), question, "p", "m"
    )


def test_probability_of_reads_the_distribution() -> None:
    decision = Decision(
        key="k",
        question_id="q",
        question_type=QuestionType.CHOICE,
        state_hash="s",
        provider="p",
        model="m",
        answer="b",
        confidence=0.7,
        distribution=(("a", 0.3), ("b", 0.7)),
    )
    assert decision.probability_of("b") == pytest.approx(0.7)
    assert decision.probability_of("missing") == 0.0


def test_expected_level_index_weights_by_probability() -> None:
    decision = Decision(
        key="k",
        question_id="q",
        question_type=QuestionType.SCORE,
        state_hash="s",
        provider="p",
        model="m",
        answer="high",
        confidence=0.6,
        distribution=(("low", 0.1), ("normal", 0.3), ("high", 0.6)),
    )
    assert decision.expected_level_index() == pytest.approx(0.1 * 0 + 0.3 * 1 + 0.6 * 2)


def test_decision_round_trips_through_a_dict() -> None:
    decision = Decision(
        key="k",
        question_id="q",
        question_type=QuestionType.NOUL,
        state_hash="s",
        provider="p",
        model="m",
        answer="yes",
        confidence=0.9,
        distribution=(("no", 0.1), ("yes", 0.9)),
        tags=("triage",),
        cost_usd=0.0001,
    )
    restored = Decision.from_dict(decision.as_dict())
    assert restored == decision
