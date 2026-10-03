"""The MCP runtime: the operations an MCP client can call.

These exercise the runtime directly rather than over the protocol, because the
transport is the MCP SDK's problem and the arithmetic is ours.
"""

from __future__ import annotations

import pytest

from metajev.errors import MetajevError
from metajev.mcp import Runtime, Settings


@pytest.fixture()
def runtime(tmp_path):
    settings = Settings(
        store_path=str(tmp_path / "mcp.db"),
        provider="mock",
        accept=0.85,
        review=0.55,
    )
    opened = Runtime(settings)
    yield opened
    opened.close()


def test_settings_default_to_the_home_store() -> None:
    settings = Settings.from_env({})
    assert settings.store_path.endswith(".metajev/decisions.db")
    assert settings.provider == "typesafe"
    assert settings.accept == 0.85
    assert settings.review == 0.55


def test_settings_read_the_environment() -> None:
    settings = Settings.from_env(
        {"METAJEV_STORE": "/tmp/x.db", "METAJEV_PROVIDER": "sglang", "METAJEV_POLICY": "0.7,0.4"}
    )
    assert settings.store_path == "/tmp/x.db"
    assert settings.provider == "sglang"
    assert settings.accept == 0.7
    assert settings.review == 0.4


def test_a_malformed_policy_is_rejected() -> None:
    with pytest.raises(MetajevError, match="must look like"):
        Settings.from_env({"METAJEV_POLICY": "nonsense"})


def test_an_inverted_policy_is_rejected() -> None:
    with pytest.raises(MetajevError, match="review < accept"):
        Settings.from_env({"METAJEV_POLICY": "0.4,0.9"})


def test_decide_returns_an_action_and_a_distribution(runtime: Runtime) -> None:
    result = runtime.decide("the invoice is negative", "Does this need a person?")
    assert result["action"] in {"accept", "review", "abstain", "escalate", "deny"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["distribution"].values()) - 1.0) < 1e-6
    assert result["decision_key"]


def test_decide_accepts_a_sharpened_yes(runtime: Runtime) -> None:
    result = runtime.decide(
        "state", "Does this need a person?", yes="a human should read it first"
    )
    assert result["decision_key"]


def test_a_choice_question_needs_options(runtime: Runtime) -> None:
    with pytest.raises(MetajevError, match="needs options"):
        runtime.decide("state", "Which queue?", kind="choice")


def test_a_score_question_needs_levels(runtime: Runtime) -> None:
    with pytest.raises(MetajevError, match="needs levels"):
        runtime.decide("state", "How urgent?", kind="score")


def test_an_unknown_question_type_is_reported(runtime: Runtime) -> None:
    with pytest.raises(MetajevError, match="unknown question type"):
        runtime.decide("state", "question", kind="mystery")


def test_a_choice_question_returns_a_distribution_over_its_options(runtime: Runtime) -> None:
    result = runtime.decide(
        "customer wants a refund", "Which queue?", kind="choice",
        options=["billing", "support", "sales"],
    )
    assert set(result["distribution"]) == {"billing", "support", "sales"}


def test_recording_an_outcome_is_reported_back(runtime: Runtime) -> None:
    result = runtime.decide("state", "Does this need a person?")
    recorded = runtime.record_outcome(result["decision_key"], result["answer"])
    assert recorded["recorded"] == result["answer"]
    assert recorded["outcomes_known"] == 1


def test_recording_against_an_unknown_decision_is_rejected(runtime: Runtime) -> None:
    with pytest.raises(MetajevError, match="no decision stored"):
        runtime.record_outcome("nope", "yes")


def test_recording_an_impossible_label_is_rejected(runtime: Runtime) -> None:
    result = runtime.decide("state", "Does this need a person?")
    with pytest.raises(MetajevError, match="is not one of"):
        runtime.record_outcome(result["decision_key"], "maybe")


def test_replay_calls_no_model(runtime: Runtime) -> None:
    for index in range(30):
        runtime.decide(f"state-{index}", "Does this need a person?")
    calls_before = runtime.client.budget.calls
    report = runtime.replay(accept=0.95, review=0.6)
    assert report["model_calls"] == 0
    assert runtime.client.budget.calls == calls_before


def test_replay_reports_the_trade(runtime: Runtime) -> None:
    for index in range(60):
        result = runtime.decide(f"state-{index}", "Does this need a person?")
        runtime.record_outcome(result["decision_key"], "yes" if index % 3 else "no")
    report = runtime.replay(accept=0.95, review=0.6)
    baseline = report["against_baseline"]
    assert baseline is not None
    assert baseline["errors_moved_off_accept"] >= 0
    assert baseline["correct_answers_diverted"] >= 0
    assert "action" in report["summary"]


def test_calibration_says_so_when_nothing_is_judged(runtime: Runtime) -> None:
    runtime.decide("state", "Does this need a person?")
    report = runtime.calibrate()
    assert report["observations"] == 0
    assert "record_outcome" in report["note"]


def test_calibration_measures_recorded_outcomes(runtime: Runtime) -> None:
    for index in range(80):
        result = runtime.decide(f"state-{index}", "Does this need a person?")
        runtime.record_outcome(result["decision_key"], "yes" if index % 4 else "no")
    report = runtime.calibrate()
    assert report["observations"] == 80
    assert 0.0 <= report["brier"] <= 1.0
    assert report["reading"] in {"overconfident", "underconfident", "calibrated"}
    assert report["reliability"]


def test_the_boundary_table_is_empty_without_outcomes(runtime: Runtime) -> None:
    runtime.decide("state", "Does this need a person?")
    table = runtime.boundary_table()
    assert table["observations"] == 0


def test_the_boundary_table_reports_both_sides(runtime: Runtime) -> None:
    for index in range(50):
        result = runtime.decide(f"state-{index}", "Does this need a person?")
        runtime.record_outcome(result["decision_key"], "yes" if index % 3 else "no")
    table = runtime.boundary_table(steps=5)
    assert table["model_calls"] == 0
    assert len(table["rows"]) == 6
    for row in table["rows"]:
        assert "errors_admitted" in row
        assert "right_turned_away" in row


def test_the_store_summary_describes_the_configuration(runtime: Runtime) -> None:
    runtime.decide("state", "Does this need a person?")
    summary = runtime.store_summary()
    assert summary["provider"] == "mock"
    assert summary["policy"] == {"accept": 0.85, "review": 0.55}
    assert summary["decisions"] == 1


def test_the_store_survives_a_restart(tmp_path) -> None:
    settings = Settings(
        store_path=str(tmp_path / "mcp.db"), provider="mock", accept=0.85, review=0.55
    )
    first = Runtime(settings)
    result = first.decide("state", "Does this need a person?")
    first.record_outcome(result["decision_key"], "yes")
    first.close()

    second = Runtime(settings)
    try:
        summary = second.store_summary()
        assert summary["decisions"] == 1
        assert summary["outcomes"] == 1
        # The recorded decision is reused rather than asked again.
        calls_before = second.client.budget.calls
        again = second.decide("state", "Does this need a person?")
        assert again["decision_key"] == result["decision_key"]
        assert second.client.budget.calls == calls_before
    finally:
        second.close()


def test_the_server_builds_and_exposes_tools(tmp_path) -> None:
    """The server registers the operations, when the SDK is available."""
    pytest.importorskip("mcp")
    from metajev.mcp import build_server

    settings = Settings(
        store_path=str(tmp_path / "srv.db"), provider="mock", accept=0.85, review=0.55
    )
    runtime = Runtime(settings)
    try:
        server = build_server(runtime)
        assert server is not None
    finally:
        runtime.close()
