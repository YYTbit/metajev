"""Provider behaviour: normalisation, label alignment, and response reading."""

from __future__ import annotations

import pytest

from metajev.errors import ProviderResponseError
from metajev.providers import (
    MockProvider,
    OpenAICompatProvider,
    SGLangProvider,
    TypesafeProvider,
    build_provider,
)
from metajev.providers.base import normalise_distribution
from metajev.providers.http_base import align_labels
from metajev.types import Question, QuestionType


def test_a_distribution_is_normalised_to_sum_to_one() -> None:
    result = normalise_distribution({"no": 2.0, "yes": 6.0}, ("no", "yes"))
    assert dict(result)["yes"] == pytest.approx(0.75)
    assert sum(value for _, value in result) == pytest.approx(1.0)


def test_missing_labels_are_filled_in_at_zero() -> None:
    result = normalise_distribution({"yes": 1.0}, ("no", "yes"))
    assert dict(result) == {"no": 0.0, "yes": 1.0}


def test_negative_mass_is_clamped_away() -> None:
    result = normalise_distribution({"no": -3.0, "yes": 4.0}, ("no", "yes"))
    assert dict(result)["yes"] == pytest.approx(1.0)


def test_an_omitted_level_keeps_the_remaining_levels_in_position() -> None:
    """A missing level takes zero mass rather than disappearing and shifting the rest."""
    result = normalise_distribution({"high": 1.0}, ("low", "mid", "high"))
    assert [label for label, _ in result] == ["low", "mid", "high"]
    assert dict(result)["low"] == 0.0
    assert dict(result)["mid"] == 0.0


def test_expected_position_is_correct_when_a_provider_omits_a_level() -> None:
    provider = MockProvider()
    question = Question.score("How urgent?", ("low", "mid", "high"))

    def answer(state: str, q: Question):
        from metajev.providers.base import ProviderAnswer

        return ProviderAnswer(answer="high", distribution=(("high", 1.0),))

    provider._answer = answer  # type: ignore[method-assign]
    decision = provider.decide("state", question)
    assert [label for label, _ in decision.distribution] == ["low", "mid", "high"]
    assert decision.expected_level_index() == pytest.approx(2.0)


def test_an_empty_distribution_is_rejected() -> None:
    with pytest.raises(ProviderResponseError, match="empty distribution"):
        normalise_distribution({}, ("no", "yes"))


def test_a_distribution_with_no_mass_is_rejected() -> None:
    with pytest.raises(ProviderResponseError, match="no mass"):
        normalise_distribution({"no": 0.0, "yes": 0.0}, ("no", "yes"))


def test_a_non_numeric_value_is_reported() -> None:
    with pytest.raises(ProviderResponseError, match="not a number"):
        normalise_distribution({"no": "lots", "yes": 1.0}, ("no", "yes"))


def test_align_labels_matches_case_insensitively() -> None:
    question = Question.choice("q", ("billing", "sales"))
    assert align_labels({"Billing": 0.6, "SALES": 0.4}, question) == {
        "billing": 0.6,
        "sales": 0.4,
    }


def test_align_labels_maps_a_positional_list() -> None:
    question = Question.score("q", ("low", "high"))
    assert align_labels([0.25, 0.75], question) == {"low": 0.25, "high": 0.75}


def test_align_labels_maps_numeric_keys() -> None:
    question = Question.score("q", ("low", "mid", "high"))
    assert align_labels({"0": 0.2, "1": 0.3, "2": 0.5}, question) == {
        "low": 0.2,
        "mid": 0.3,
        "high": 0.5,
    }


def test_align_labels_reports_a_mismatch() -> None:
    question = Question.choice("q", ("billing", "sales"))
    with pytest.raises(ProviderResponseError, match="match none of the question labels"):
        align_labels({"invoice": 0.6, "deal": 0.4}, question)


def test_align_labels_reports_a_length_mismatch() -> None:
    question = Question.score("q", ("low", "mid", "high"))
    with pytest.raises(ProviderResponseError, match="3 labels"):
        align_labels([0.5, 0.5], question)


def test_the_mock_is_deterministic() -> None:
    provider = MockProvider()
    question = Question.noul("Is this urgent?", tags=("triage",))
    first = provider.decide("same state", question)
    second = provider.decide("same state", question)
    assert first.distribution == second.distribution
    assert first.answer == second.answer


def test_the_mock_separates_states() -> None:
    provider = MockProvider()
    question = Question.noul("Is this urgent?")
    answers = {provider.decide(f"state-{i}", question).answer for i in range(40)}
    assert answers == {"yes", "no"}


def test_the_mock_distribution_sums_to_one() -> None:
    provider = MockProvider()
    decision = provider.decide("state", Question.choice("q", ("a", "b", "c")))
    assert sum(value for _, value in decision.distribution) == pytest.approx(1.0)


def test_the_mock_confidence_lands_in_the_useful_range() -> None:
    provider = MockProvider()
    question = Question.noul("Is this urgent?")
    confidences = [provider.decide(f"state-{i}", question).confidence for i in range(200)]
    assert all(0.5 <= value <= 1.0 for value in confidences)
    assert len(set(round(value, 3) for value in confidences)) > 50


def test_the_mock_health_check_always_passes() -> None:
    assert MockProvider().health()[0] is True


def test_a_typesafe_payload_carries_the_question_shape() -> None:
    provider = TypesafeProvider(api_key="k")
    choice = provider.payload("state", Question.choice("Which queue?", ("a", "b")))
    assert choice["type"] == "choice"
    assert choice["options"] == ["a", "b"]

    score = provider.payload("state", Question.score("How urgent?", ("low", "high")))
    assert score["type"] == "score"
    assert score["levels"] == ["low", "high"]

    noul = provider.payload("state", Question.noul("Is this urgent?", yes="needs a person"))
    assert noul["type"] == "noul"
    assert noul["yes"] == "needs a person"


def test_a_typesafe_provider_without_a_key_is_unhealthy() -> None:
    provider = TypesafeProvider(api_key="")
    healthy, note = provider.health()
    assert healthy is False
    assert "API key" in note


def test_a_typesafe_provider_with_a_key_is_healthy() -> None:
    assert TypesafeProvider(api_key="k").health()[0] is True


def test_a_typesafe_response_is_read() -> None:
    provider = TypesafeProvider(api_key="k")
    question = Question.noul("Is this urgent?")
    answer = provider.extract(
        {"answer": "yes", "probabilities": {"no": 0.1, "yes": 0.9}, "confidence": 0.9},
        question,
    )
    assert answer.answer == "yes"
    assert dict(answer.distribution)["yes"] == pytest.approx(0.9)


def test_a_missing_distribution_is_reported() -> None:
    provider = TypesafeProvider(api_key="k")
    with pytest.raises(ProviderResponseError, match="carried no distribution"):
        provider.extract({"answer": "yes"}, Question.noul("Is this urgent?"))


def test_openai_compat_reads_token_logprobs() -> None:
    import math

    provider = OpenAICompatProvider(model="local")
    question = Question.noul("Is this urgent?")
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "yes", "logprob": math.log(0.8)},
                                {"token": "no", "logprob": math.log(0.2)},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    answer = provider.extract(response, question)
    assert answer.answer == "yes"
    assert dict(answer.distribution)["yes"] == pytest.approx(0.8)


def test_openai_compat_maps_a_numeric_answer_to_a_level() -> None:
    import math

    provider = OpenAICompatProvider(model="local")
    question = Question.score("How urgent?", ("low", "mid", "high"))
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "3", "logprob": math.log(0.7)},
                                {"token": "1", "logprob": math.log(0.3)},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    answer = provider.extract(response, question)
    assert answer.answer == "high"


def test_openai_compat_reports_unmatched_tokens() -> None:
    import math

    provider = OpenAICompatProvider(model="local")
    question = Question.noul("Is this urgent?")
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"top_logprobs": [{"token": "maybe", "logprob": math.log(0.9)}]}
                    ]
                }
            }
        ]
    }
    with pytest.raises(ProviderResponseError, match="must be a single token"):
        provider.extract(response, question)


def test_openai_compat_reports_missing_logprobs() -> None:
    provider = OpenAICompatProvider(model="local")
    with pytest.raises(ProviderResponseError, match="no token logprobs"):
        provider.extract({"choices": [{}]}, Question.noul("Is this urgent?"))


def test_sglang_reads_a_probability_map() -> None:
    provider = SGLangProvider(model="local")
    question = Question.score("How urgent?", ("low", "mid", "high"))
    answer = provider.extract(
        {"meta_info": {"probabilities": {"low": 0.1, "mid": 0.2, "high": 0.7}}}, question
    )
    assert answer.answer == "high"


def test_sglang_reads_logprob_entries() -> None:
    import math

    provider = SGLangProvider(model="local")
    question = Question.noul("Is this urgent?")
    answer = provider.extract(
        {"probabilities": [[" yes", math.log(0.9)], [" no", math.log(0.1)]]}, question
    )
    assert answer.answer == "yes"


def test_sglang_payload_asks_for_no_new_tokens() -> None:
    provider = SGLangProvider(model="local")
    payload = provider.payload("state", Question.noul("Is this urgent?"))
    assert payload["max_new_tokens"] == 0
    assert payload["return_logprob"] is True


def test_sglang_reports_a_missing_distribution() -> None:
    provider = SGLangProvider(model="local")
    with pytest.raises(ProviderResponseError, match="carried no distribution"):
        provider.extract({"meta_info": {}}, Question.noul("Is this urgent?"))


def test_build_provider_selects_by_name() -> None:
    assert isinstance(build_provider("mock"), MockProvider)
    assert isinstance(build_provider("sglang"), SGLangProvider)
    assert isinstance(build_provider("openai"), OpenAICompatProvider)


def test_build_provider_reads_a_model_suffix() -> None:
    provider = build_provider("sglang:my-decision-model")
    assert provider.model == "my-decision-model"


def test_build_provider_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("nonesuch")


def test_decide_records_the_question_type_and_tags() -> None:
    provider = MockProvider()
    question = Question.score("How urgent?", ("low", "high"), tags=("urgency",))
    decision = provider.decide("state", question)
    assert decision.question_type is QuestionType.SCORE
    assert decision.tags == ("urgency",)
    assert decision.question_id == question.id
