"""The append-only record and its tamper detection."""

from __future__ import annotations

import json

import pytest

from metajev.errors import LedgerIntegrityError
from metajev.ledger import GENESIS, Ledger
from metajev.policy import three_band_policy
from metajev.types import Decision, QuestionType


def a_decision(key: str = "k1", confidence: float = 0.9) -> Decision:
    return Decision(
        key=key,
        question_id="q1",
        question_type=QuestionType.NOUL,
        state_hash="s",
        provider="p",
        model="m",
        answer="yes",
        confidence=confidence,
        distribution=(("no", 1.0 - confidence), ("yes", confidence)),
    )


def test_an_empty_ledger_verifies_and_sits_at_genesis(tmp_path) -> None:
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    assert ledger.verify() == 0
    assert ledger.head() == GENESIS


def test_appending_links_each_entry_to_the_one_before(tmp_path) -> None:
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    first = ledger.append("note", {"text": "one"})
    second = ledger.append("note", {"text": "two"})
    assert first.prev == GENESIS
    assert second.prev == first.hash
    assert ledger.verify() == 2


def test_sequence_numbers_advance_by_one(tmp_path) -> None:
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    for index in range(5):
        ledger.append("note", {"index": index})
    assert [entry.seq for entry in ledger.entries()] == [1, 2, 3, 4, 5]


def test_recording_a_decision_keeps_the_distribution(tmp_path) -> None:
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    decision = a_decision()
    ledger.record_decision(decision, three_band_policy(accept=0.85, review=0.55).resolve(decision))
    entry = next(ledger.entries())
    assert entry.kind == "decision"
    assert entry.body["decision"]["key"] == "k1"
    distribution = dict(entry.body["decision"]["distribution"])
    assert distribution["yes"] == pytest.approx(0.9)
    assert distribution["no"] == pytest.approx(0.1)
    assert entry.body["resolution"]["action"] == "accept"


def test_editing_an_entry_is_detected(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(str(path))
    ledger.append("note", {"amount": 10})
    ledger.append("note", {"amount": 20})

    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["body"]["amount"] = 999
    lines[0] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="entry 1 has hash"):
        ledger.verify()


def test_removing_an_entry_is_detected(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(str(path))
    for index in range(3):
        ledger.append("note", {"index": index})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="out of order"):
        ledger.verify()


def test_a_truncated_line_is_reported_with_its_number(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(str(path))
    ledger.append("note", {"index": 0})
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(LedgerIntegrityError, match="line 2"):
        ledger.verify()


def test_outcomes_are_recorded_as_their_own_entry_kind(tmp_path) -> None:
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    ledger.record_outcome("k1", "no", note="checked by hand")
    entry = next(ledger.entries())
    assert entry.kind == "outcome"
    assert entry.body == {"key": "k1", "label": "no", "note": "checked by hand"}


def test_reopening_a_ledger_continues_the_chain(tmp_path) -> None:
    path = str(tmp_path / "ledger.jsonl")
    first = Ledger(path)
    first.append("note", {"index": 0})
    head = first.head()

    second = Ledger(path)
    appended = second.append("note", {"index": 1})
    assert appended.prev == head
    assert second.verify() == 2


def test_count_matches_the_entries(tmp_path) -> None:
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    for index in range(4):
        ledger.append("note", {"index": index})
    assert ledger.count() == 4
