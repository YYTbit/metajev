"""Bands, rules, selectors, and the actions a policy resolves to."""

from __future__ import annotations

import pytest

from metajev.policy import Band, Policy, Rule, label_policy, three_band_policy
from metajev.types import Action, Decision, QuestionType


def make_decision(
    *,
    confidence: float,
    distribution: tuple[tuple[str, float], ...],
    question_type: QuestionType = QuestionType.NOUL,
    tags: tuple[str, ...] = (),
    question_id: str = "q",
    provider: str = "p",
) -> Decision:
    answer = max(distribution, key=lambda pair: pair[1])[0]
    return Decision(
        key="k",
        question_id=question_id,
        question_type=question_type,
        state_hash="s",
        provider=provider,
        model="m",
        answer=answer,
        confidence=confidence,
        distribution=distribution,
        tags=tags,
    )


def test_band_rejects_an_inverted_interval() -> None:
    with pytest.raises(ValueError, match="below upper bound"):
        Band(0.9, 0.5, Action.ACCEPT)


def test_band_is_half_open() -> None:
    band = Band(0.5, 0.9, Action.REVIEW)
    assert band.contains(0.5)
    assert band.contains(0.8999)
    assert not band.contains(0.9)


def test_policy_accepts_a_confident_answer() -> None:
    policy = three_band_policy(accept=0.85, review=0.55)
    decision = make_decision(confidence=0.95, distribution=(("no", 0.05), ("yes", 0.95)))
    assert policy.resolve(decision).action is Action.ACCEPT


def test_policy_sends_a_middling_answer_to_review() -> None:
    policy = three_band_policy(accept=0.85, review=0.55)
    decision = make_decision(confidence=0.70, distribution=(("no", 0.30), ("yes", 0.70)))
    assert policy.resolve(decision).action is Action.REVIEW


def test_policy_escalates_a_low_confidence_answer() -> None:
    policy = three_band_policy(accept=0.85, review=0.55)
    decision = make_decision(confidence=0.52, distribution=(("no", 0.48), ("yes", 0.52)))
    assert policy.resolve(decision).action is Action.ESCALATE


def test_policy_boundaries_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="review < accept"):
        three_band_policy(accept=0.5, review=0.8)


def test_tag_selector_scopes_a_rule() -> None:
    policy = Policy(
        name="scoped",
        rules=(
            Rule(
                name="triage",
                on="confidence",
                tags=("triage",),
                bands=(Band(0.8, 1.0001, Action.ACCEPT),),
            ),
        ),
        default=Action.ABSTAIN,
    )
    inside = make_decision(
        confidence=0.9, distribution=(("no", 0.1), ("yes", 0.9)), tags=("triage",)
    )
    outside = make_decision(
        confidence=0.9, distribution=(("no", 0.1), ("yes", 0.9)), tags=("other",)
    )
    assert policy.resolve(inside).action is Action.ACCEPT
    assert policy.resolve(outside).action is Action.ABSTAIN


def test_question_type_selector_scopes_a_rule() -> None:
    policy = Policy(
        name="typed",
        rules=(
            Rule(
                name="scores only",
                on="confidence",
                question_types=(QuestionType.SCORE,),
                bands=(Band(0.0, 1.0001, Action.ESCALATE),),
            ),
        ),
        default=Action.ACCEPT,
    )
    score = make_decision(
        confidence=0.9,
        distribution=(("low", 0.1), ("high", 0.9)),
        question_type=QuestionType.SCORE,
    )
    noul = make_decision(confidence=0.9, distribution=(("no", 0.1), ("yes", 0.9)))
    assert policy.resolve(score).action is Action.ESCALATE
    assert policy.resolve(noul).action is Action.ACCEPT


def test_label_rule_reads_a_named_probability() -> None:
    policy = label_policy("yes", accept=0.7, review=0.4)
    decision = make_decision(confidence=0.6, distribution=(("no", 0.4), ("yes", 0.6)))
    resolution = policy.resolve(decision)
    assert resolution.value == pytest.approx(0.6)
    assert resolution.action is Action.REVIEW


def test_unmatched_decision_takes_the_default() -> None:
    policy = Policy(name="empty", rules=(), default=Action.ABSTAIN)
    decision = make_decision(confidence=0.99, distribution=(("no", 0.01), ("yes", 0.99)))
    resolution = policy.resolve(decision)
    assert resolution.action is Action.ABSTAIN
    assert resolution.rule == ""


def test_unknown_quantity_is_reported() -> None:
    policy = Policy(
        name="bad",
        rules=(Rule(name="r", on="nonsense", bands=(Band(0.0, 1.0, Action.ACCEPT),)),),
    )
    decision = make_decision(confidence=0.9, distribution=(("no", 0.1), ("yes", 0.9)))
    with pytest.raises(Exception, match="unknown quantity"):
        policy.resolve(decision)


def test_expected_is_normalised_for_a_score_question() -> None:
    policy = Policy(
        name="expected",
        rules=(
            Rule(
                name="position",
                on="expected",
                bands=(Band(0.5, 1.0001, Action.ESCALATE), Band(0.0, 0.5, Action.ACCEPT)),
            ),
        ),
    )
    high = make_decision(
        confidence=0.6,
        distribution=(("low", 0.1), ("normal", 0.3), ("high", 0.6)),
        question_type=QuestionType.SCORE,
    )
    low = make_decision(
        confidence=0.6,
        distribution=(("low", 0.6), ("normal", 0.3), ("high", 0.1)),
        question_type=QuestionType.SCORE,
    )
    assert policy.resolve(high).action is Action.ESCALATE
    assert policy.resolve(low).action is Action.ACCEPT


def test_policy_round_trips_through_json() -> None:
    policy = three_band_policy("balanced", accept=0.85, review=0.55)
    restored = Policy.from_json(policy.to_json())
    assert restored == policy
    assert restored.fingerprint == policy.fingerprint


def test_fingerprint_moves_when_a_boundary_moves() -> None:
    first = three_band_policy("p", accept=0.85, review=0.55)
    second = three_band_policy("p", accept=0.80, review=0.55)
    assert first.fingerprint != second.fingerprint


def test_fingerprint_ignores_the_name() -> None:
    """Two people naming the same rule set the same way must agree on the fingerprint."""
    first = three_band_policy("alpha", accept=0.85, review=0.55)
    second = three_band_policy("beta", accept=0.85, review=0.55)
    assert first.fingerprint == second.fingerprint


def test_describe_lists_every_band() -> None:
    text = three_band_policy("balanced", accept=0.85, review=0.55).describe()
    assert "balanced" in text
    assert "accept" in text
    assert "review" in text
    assert "escalate" in text
