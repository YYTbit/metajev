"""A content-addressed store for decisions, and the outcomes that later judge them.

The store is the reason a policy can change without a model call. Decisions are
addressed by :func:`metajev.types.decision_key`, which hashes the state, the
question, and the model that answered. Policy inputs are absent from that key, so
re-resolving a stored decision under a new policy reads the same row and computes
a different :class:`~metajev.types.Action`.

Outcomes are kept in their own table because the truth generally arrives after the
decision does, and often never arrives at all. A decision without an outcome still
replays; it simply cannot contribute to calibration.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

from .errors import StoreError
from .types import Decision, Outcome, QuestionType, now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    key           TEXT PRIMARY KEY,
    question_id   TEXT NOT NULL,
    question_type TEXT NOT NULL,
    state_hash    TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    answer        TEXT NOT NULL,
    confidence    REAL NOT NULL,
    cost_usd      REAL NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_question ON decisions(question_id);
CREATE INDEX IF NOT EXISTS decisions_provider ON decisions(provider, model);

CREATE TABLE IF NOT EXISTS outcomes (
    key         TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1,
    recorded_at REAL NOT NULL DEFAULT 0,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outcomes_label ON outcomes(label);
"""


@dataclass(frozen=True)
class StoreStats:
    """A summary of what a store holds."""

    decisions: int
    outcomes: int
    total_cost_usd: float
    mean_confidence: float
    by_provider: Mapping[str, int]
    by_question_type: Mapping[str, int]


class DecisionStore:
    """A SQLite-backed store of decisions and outcomes.

    Pass ``":memory:"`` for an ephemeral store, or a path for one that survives the
    process. The store is safe to open from several processes; SQLite serialises
    writers.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        try:
            self._conn = sqlite3.connect(path, timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"could not open store at {path}: {exc}") from exc

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DecisionStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- decisions ---------------------------------------------------------

    def put(self, decision: Decision) -> bool:
        """Store a decision. Return True when it was new, False when it already existed.

        Storing an existing key is a no-op rather than an overwrite, because a
        decision is a record of what a model said. Re-asking the same question of
        the same model is expected to reproduce the same distribution, so keeping
        the first recording is the honest choice.
        """
        try:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO decisions "
                "(key, question_id, question_type, state_hash, provider, model, answer, "
                " confidence, cost_usd, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.key,
                    decision.question_id,
                    decision.question_type.value,
                    decision.state_hash,
                    decision.provider,
                    decision.model,
                    decision.answer,
                    decision.confidence,
                    decision.cost_usd,
                    decision.created_at or now(),
                    json.dumps(decision.as_dict(), ensure_ascii=False),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"could not store decision {decision.key}: {exc}") from exc
        return cursor.rowcount > 0

    def put_many(self, decisions: Iterable[Decision]) -> int:
        """Store several decisions. Return how many were new."""
        return sum(1 for decision in decisions if self.put(decision))

    def get(self, key: str) -> Decision | None:
        """Return the decision stored under ``key``, or None."""
        row = self._conn.execute("SELECT payload FROM decisions WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return Decision.from_dict(json.loads(row["payload"]))

    def require(self, key: str) -> Decision:
        """Return the decision stored under ``key``, raising when it is absent."""
        decision = self.get(key)
        if decision is None:
            raise StoreError(f"no decision stored under key {key}")
        return decision

    def has(self, key: str) -> bool:
        """Return whether a decision is stored under ``key``."""
        row = self._conn.execute("SELECT 1 FROM decisions WHERE key = ?", (key,)).fetchone()
        return row is not None

    def decisions(
        self,
        *,
        question_id: str | None = None,
        provider: str | None = None,
        question_type: QuestionType | None = None,
        limit: int | None = None,
    ) -> Iterator[Decision]:
        """Iterate stored decisions, optionally filtered."""
        clauses: list[str] = []
        params: list[Any] = []
        if question_id is not None:
            clauses.append("question_id = ?")
            params.append(question_id)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if question_type is not None:
            clauses.append("question_type = ?")
            params.append(question_type.value)
        sql = "SELECT payload FROM decisions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in self._conn.execute(sql, params):
            yield Decision.from_dict(json.loads(row["payload"]))

    def count(self) -> int:
        """Return how many decisions are stored."""
        return int(self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])

    # -- outcomes ----------------------------------------------------------

    def record_outcome(
        self,
        key: str,
        label: str,
        *,
        weight: float = 1.0,
        note: str = "",
    ) -> None:
        """Record the correct answer for a decision, replacing any earlier record."""
        outcome = Outcome(key=key, label=label, weight=weight, recorded_at=now(), note=note)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO outcomes (key, label, weight, recorded_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    outcome.key,
                    outcome.label,
                    outcome.weight,
                    outcome.recorded_at,
                    json.dumps(outcome.as_dict(), ensure_ascii=False),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"could not record outcome for {key}: {exc}") from exc

    def outcome_for(self, key: str) -> Outcome | None:
        """Return the outcome recorded for ``key``, or None."""
        row = self._conn.execute("SELECT payload FROM outcomes WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return Outcome.from_dict(json.loads(row["payload"]))

    def outcomes(self) -> Iterator[Outcome]:
        """Iterate every recorded outcome."""
        for row in self._conn.execute("SELECT payload FROM outcomes ORDER BY recorded_at"):
            yield Outcome.from_dict(json.loads(row["payload"]))

    def outcome_count(self) -> int:
        """Return how many outcomes are recorded."""
        return int(self._conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])

    def judged(self) -> Iterator[tuple[Decision, Outcome]]:
        """Iterate decisions that have a recorded outcome, paired with it."""
        sql = (
            "SELECT d.payload AS decision, o.payload AS outcome "
            "FROM decisions d JOIN outcomes o ON d.key = o.key "
            "ORDER BY d.created_at"
        )
        for row in self._conn.execute(sql):
            yield Decision.from_dict(json.loads(row["decision"])), Outcome.from_dict(
                json.loads(row["outcome"])
            )

    # -- reporting ---------------------------------------------------------

    def stats(self) -> StoreStats:
        """Summarise what the store holds."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd), 0) AS cost, "
            "COALESCE(AVG(confidence), 0) AS conf FROM decisions"
        ).fetchone()
        by_provider = {
            str(r["provider"]): int(r["n"])
            for r in self._conn.execute(
                "SELECT provider, COUNT(*) AS n FROM decisions GROUP BY provider ORDER BY n DESC"
            )
        }
        by_type = {
            str(r["question_type"]): int(r["n"])
            for r in self._conn.execute(
                "SELECT question_type, COUNT(*) AS n FROM decisions "
                "GROUP BY question_type ORDER BY n DESC"
            )
        }
        return StoreStats(
            decisions=int(row["n"]),
            outcomes=self.outcome_count(),
            total_cost_usd=float(row["cost"]),
            mean_confidence=float(row["conf"]),
            by_provider=by_provider,
            by_question_type=by_type,
        )

    # -- interchange -------------------------------------------------------

    def export_jsonl(self, path: str) -> int:
        """Write every decision and outcome to a JSONL file. Return rows written."""
        written = 0
        with open(path, "w", encoding="utf-8") as handle:
            for decision in self.decisions():
                handle.write(json.dumps({"kind": "decision", **decision.as_dict()}) + "\n")
                written += 1
            for outcome in self.outcomes():
                handle.write(json.dumps({"kind": "outcome", **outcome.as_dict()}) + "\n")
                written += 1
        return written

    def import_jsonl(self, path: str) -> tuple[int, int]:
        """Read a JSONL file produced by :meth:`export_jsonl`.

        Return the number of new decisions and the number of outcomes read.
        """
        new_decisions = 0
        outcomes = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                kind = payload.pop("kind", "decision")
                if kind == "decision":
                    if self.put(Decision.from_dict(payload)):
                        new_decisions += 1
                elif kind == "outcome":
                    outcome = Outcome.from_dict(payload)
                    self.record_outcome(
                        outcome.key,
                        outcome.label,
                        weight=outcome.weight,
                        note=outcome.note,
                    )
                    outcomes += 1
                else:
                    raise StoreError(f"unknown record kind {kind!r} in {path}")
        return new_decisions, outcomes
