"""The client: asking, caching, acting, and recording."""

from __future__ import annotations

import pytest

from metajev.budget import Budget
from metajev.client import Client
from metajev.ledger import Ledger
from metajev.policy import three_band_policy
from metajev.store import DecisionStore
from metajev.types import Action, Question

from conftest import CountingProvider, ground_truth


def a_question() -> Question:
    return Question.noul("Is this urgent?", yes="a person should look now")


def test_asking_a_new_question_calls_the_provider() -> None:
    provider = CountingProvider()
    with Client(provider=provider, store=DecisionStore(":memory:")) as client:
        answer = client.decide("state", a_question())
        assert answer.cached is False
        assert provider.calls == 1


def test_asking_the_same_question_again_does_not() -> None:
    provider = CountingProvider()
    with Client(provider=provider, store=DecisionStore(":memory:")) as client:
        client.decide("state", a_question())
        second = client.decide("state", a_question())
        assert second.cached is True
        assert provider.calls == 1


def test_the_cache_key_covers_the_state() -> None:
    provider = CountingProvider()
    with Client(provider=provider, store=DecisionStore(":memory:")) as client:
        client.decide("one", a_question())
        client.decide("two", a_question())
        assert provider.calls == 2


def test_the_cache_key_covers_the_question() -> None:
    provider = CountingProvider()
    with Client(provider=provider, store=DecisionStore(":memory:")) as client:
        client.decide("state", Question.noul("Is this urgent?"))
        client.decide("state", Question.noul("Is this important?"))
        assert provider.calls == 2


def test_a_decision_is_recorded_in_the_store() -> None:
    with Client(provider=CountingProvider(), store=DecisionStore(":memory:")) as client:
        answer = client.decide("state", a_question())
        assert client.store.get(answer.decision.key) == answer.decision


def test_a_decision_is_recorded_in_the_ledger() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(str(Path(tmp) / "ledger.jsonl"))
        with Client(
            provider=CountingProvider(), store=DecisionStore(":memory:"), ledger=ledger
        ) as client:
            client.decide("state", a_question())
            # one decision entry, and nothing else yet
            assert ledger.count() == 1
            assert ledger.verify() == 1


def test_recording_an_outcome_reaches_both_the_store_and_the_ledger() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(str(Path(tmp) / "ledger.jsonl"))
        with Client(
            provider=CountingProvider(), store=DecisionStore(":memory:"), ledger=ledger
        ) as client:
            answer = client.decide("state", a_question())
            client.record_outcome(answer, "no")
            assert client.store.outcome_for(answer.decision.key).label == "no"
            assert ledger.count() == 2


def test_a_policy_change_re_resolves_a_cached_decision() -> None:
    """The stored distribution is read again, and no call is made."""
    provider = CountingProvider(confidence=0.80)
    store = DecisionStore(":memory:")
    client = Client(provider=provider, store=store, policy=three_band_policy(accept=0.70))
    first = client.decide("state", a_question())
    assert first.action is Action.ACCEPT

    client.policy = three_band_policy("stricter", accept=0.90, review=0.5)
    second = client.decide("state", a_question())
    assert second.cached is True
    assert second.action is Action.REVIEW
    assert provider.calls == 1
    client.close()


def test_the_budget_is_charged_and_can_stop_a_run() -> None:
    provider = CountingProvider(cost_usd=0.4)
    budget = Budget(usd_limit=1.0)
    with Client(provider=provider, store=DecisionStore(":memory:"), budget=budget) as client:
        client.decide("one", a_question())
        client.decide("two", a_question())
        with pytest.raises(Exception, match="limit is"):
            client.decide("three", a_question())
    assert budget.calls == 2


def test_a_cached_answer_does_not_spend_the_budget() -> None:
    provider = CountingProvider(cost_usd=0.4)
    budget = Budget(usd_limit=1.0)
    with Client(provider=provider, store=DecisionStore(":memory:"), budget=budget) as client:
        client.decide("one", a_question())
        client.decide("one", a_question())
        assert budget.calls == 1
        assert budget.spent_usd == pytest.approx(0.4)


def test_decide_many_answers_each_pair() -> None:
    with Client(provider=CountingProvider(), store=DecisionStore(":memory:")) as client:
        answers = client.decide_many([(f"s{i}", a_question()) for i in range(4)])
        assert len(answers) == 4
        assert all(not answer.cached for answer in answers)


def test_a_routed_client_uses_the_router() -> None:
    cheap = CountingProvider(cost_usd=0.00001)
    cheap.name = "cheap"
    cheap.model = "cheap-1"
    pricey = CountingProvider(cost_usd=0.001)
    pricey.name = "pricey"
    pricey.model = "pricey-1"
    with Client(providers=[pricey, cheap], store=DecisionStore(":memory:")) as client:
        answer = client.decide("state", a_question())
        assert answer.route is not None
        assert answer.route.provider.name == "cheap"


def test_a_routed_client_records_the_decision() -> None:
    provider = CountingProvider()
    provider.name = "solo"
    with Client(providers=[provider], store=DecisionStore(":memory:")) as client:
        answer = client.decide("state", a_question())
        assert client.store.get(answer.decision.key) is not None
        assert provider.calls == 1


def test_a_client_without_a_provider_is_rejected() -> None:
    with pytest.raises(Exception, match="needs a provider"):
        Client()


def test_stats_and_calibration_read_the_history() -> None:
    with Client(provider=CountingProvider(), store=DecisionStore(":memory:")) as client:
        for index in range(30):
            state = f"ticket-{index}"
            answer = client.decide(state, a_question())
            client.record_outcome(answer, ground_truth(state))
        assert client.stats().decisions == 30
        assert client.stats().outcomes == 30
        summary = client.calibration()
        assert summary.n == 30


def test_answer_serialises_to_a_flat_record() -> None:
    with Client(provider=CountingProvider(), store=DecisionStore(":memory:")) as client:
        payload = client.decide("state", a_question()).to_dict()
        assert set(payload) == {
            "action",
            "answer",
            "confidence",
            "decision_key",
            "provider",
            "model",
            "cached",
            "reason",
        }


def test_storing_the_decision_survives_a_reopened_store(tmp_path) -> None:
    path = str(tmp_path / "decisions.db")
    provider = CountingProvider()
    with Client(provider=provider, store=path) as client:
        answer = client.decide("state", a_question())
        key = answer.decision.key

    with Client(provider=provider, store=path) as reopened:
        again = reopened.decide("state", a_question())
        assert again.cached is True
        assert again.decision.key == key
    assert provider.calls == 1
