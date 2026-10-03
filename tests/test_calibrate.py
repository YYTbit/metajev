"""Calibration statistics over recorded decisions."""

from __future__ import annotations

import pytest

from metajev.calibrate import (
    Observation,
    brier,
    expected_calibration_error,
    observations_from_store,
    reliability,
    summarise,
    threshold_table,
)
from metajev.store import DecisionStore
from metajev.types import Decision, QuestionType


def test_brier_matches_a_hand_computed_value() -> None:
    observations = [Observation("a", 0.8, True), Observation("b", 0.2, False)]
    # ((0.8 - 1)^2 + (0.2 - 0)^2) / 2 = (0.04 + 0.04) / 2
    assert brier(observations) == pytest.approx(0.04)


def test_brier_of_an_empty_set_is_zero() -> None:
    assert brier([]) == 0.0


def test_brier_rewards_sharp_correct_answers() -> None:
    sharp = [Observation(str(i), 0.99, True) for i in range(10)]
    vague = [Observation(str(i), 0.55, True) for i in range(10)]
    assert brier(sharp) < brier(vague)


def test_reliability_buckets_by_predicted_probability() -> None:
    observations = [
        Observation("a", 0.15, False),
        Observation("b", 0.15, True),
        Observation("c", 0.95, True),
        Observation("d", 0.95, True),
    ]
    buckets = reliability(observations, bins=10)
    assert len(buckets) == 2
    low, high = buckets
    assert low.count == 2
    assert low.accuracy == pytest.approx(0.5)
    assert high.count == 2
    assert high.accuracy == pytest.approx(1.0)


def test_reliability_places_a_certain_prediction_in_the_last_bucket() -> None:
    buckets = reliability([Observation("a", 1.0, True)], bins=10)
    assert len(buckets) == 1
    assert buckets[0].hi == pytest.approx(1.0)


def test_reliability_rejects_a_zero_bin_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        reliability([Observation("a", 0.5, True)], bins=0)


def test_expected_calibration_error_is_count_weighted() -> None:
    buckets = reliability(
        [Observation("a", 0.1, False), Observation("b", 0.9, True)], bins=10
    )
    ece = expected_calibration_error(buckets)
    # Both buckets are perfectly calibrated, so the error is the small residual.
    assert ece == pytest.approx(0.1)


def test_ece_of_an_empty_bucket_list_is_zero() -> None:
    assert expected_calibration_error([]) == 0.0


def test_summarise_detects_overconfidence() -> None:
    observations = [Observation(str(i), 0.9, i % 10 == 0) for i in range(100)]
    summary = summarise(observations, name="overconfident")
    assert summary.n == 100
    assert summary.mean_predicted == pytest.approx(0.9)
    assert summary.accuracy == pytest.approx(0.1)
    assert summary.overconfidence == pytest.approx(0.8)


def test_summarise_of_an_empty_set_does_not_divide_by_zero() -> None:
    summary = summarise([])
    assert summary.n == 0
    assert summary.brier == 0.0
    assert summary.bins == ()


def test_summarise_renders_a_curve() -> None:
    text = summarise([Observation(str(i), 0.7, i % 2 == 0) for i in range(20)], name="x").render()
    assert "brier score" in text
    assert "observed accuracy" in text
    assert "bin" in text


def test_threshold_table_reports_the_trade() -> None:
    observations = [
        Observation("a", 0.9, True),
        Observation("b", 0.9, False),
        Observation("c", 0.2, True),
    ]
    rows = threshold_table(observations, [0.5])
    row = rows[0]
    assert row["admitted"] == 2
    assert row["precision"] == pytest.approx(0.5)
    assert row["errors_admitted"] == 1
    assert row["right_turned_away"] == 1


def test_observations_read_confidence_by_default() -> None:
    with DecisionStore(":memory:") as store:
        for index in range(3):
            decision = Decision(
                key=f"k{index}",
                question_id="q",
                question_type=QuestionType.NOUL,
                state_hash="s",
                provider="p",
                model="m",
                answer="yes",
                confidence=0.8,
                distribution=(("no", 0.2), ("yes", 0.8)),
            )
            store.put(decision)
            store.record_outcome(f"k{index}", "yes" if index else "no")
        observations, unjudged = observations_from_store(store)
        assert len(observations) == 3
        assert unjudged == 0
        assert observations[0].correct is False
        assert observations[1].correct is True


def test_observations_read_a_named_label() -> None:
    with DecisionStore(":memory:") as store:
        decision = Decision(
            key="k",
            question_id="q",
            question_type=QuestionType.NOUL,
            state_hash="s",
            provider="p",
            model="m",
            answer="yes",
            confidence=0.8,
            distribution=(("no", 0.3), ("yes", 0.7)),
        )
        store.put(decision)
        store.record_outcome("k", "no")
        observations, _ = observations_from_store(store, label="yes")
        assert observations[0].predicted == pytest.approx(0.7)
        assert observations[0].correct is False


def test_observations_count_decisions_without_an_outcome() -> None:
    with DecisionStore(":memory:") as store:
        decision = Decision(
            key="k",
            question_id="q",
            question_type=QuestionType.NOUL,
            state_hash="s",
            provider="p",
            model="m",
            answer="yes",
            confidence=0.8,
            distribution=(("no", 0.2), ("yes", 0.8)),
        )
        store.put(decision)
        observations, unjudged = observations_from_store(store)
        assert observations == []
        assert unjudged == 1
