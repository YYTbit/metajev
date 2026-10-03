"""An append-only, hash-chained record of what was decided and what happened next.

The decision store is a working database. Rows can be rewritten, outcomes can be
corrected, and nothing about a SQLite file proves that yesterday's row is the row
that was written yesterday. When a decision needs to be defensible after the fact,
that gap matters.

A ledger closes it. Each entry carries the hash of the entry before it, so removing
or editing any entry breaks every hash after it and :meth:`Ledger.verify` says
exactly where. The chain covers the decision, the action a policy took on it, and
the outcome once it is known, which together are the whole story of a decision.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from .errors import LedgerIntegrityError
from .policy import Resolution
from .types import Decision, content_hash, now

GENESIS = "0" * 64


@dataclass(frozen=True)
class Entry:
    """One link in the chain."""

    seq: int
    prev: str
    kind: str
    body: Mapping[str, Any]
    recorded_at: float
    hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "prev": self.prev,
            "kind": self.kind,
            "body": dict(self.body),
            "recorded_at": self.recorded_at,
            "hash": self.hash,
        }


def _entry_hash(seq: int, prev: str, kind: str, body: Mapping[str, Any], recorded_at: float) -> str:
    return content_hash(
        {
            "seq": seq,
            "prev": prev,
            "kind": kind,
            "body": body,
            "recorded_at": recorded_at,
        }
    )


class Ledger:
    """A JSONL file where each line commits to the line before it.

    Appending is the only write this class offers. There is no update and no delete,
    because a record that can be edited is not a record of what happened.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _last(self) -> tuple[int, str]:
        """Return the sequence number and hash of the final entry."""
        last: Entry | None = None
        for entry in self.entries():
            last = entry
        if last is None:
            return 0, GENESIS
        return last.seq, last.hash

    def append(self, kind: str, body: Mapping[str, Any]) -> Entry:
        """Commit one entry to the ledger and return it."""
        seq, prev = self._last()
        recorded_at = now()
        digest = _entry_hash(seq + 1, prev, kind, body, recorded_at)
        entry = Entry(
            seq=seq + 1,
            prev=prev,
            kind=kind,
            body=dict(body),
            recorded_at=recorded_at,
            hash=digest,
        )
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
        return entry

    def record_decision(self, decision: Decision, resolution: Resolution | None = None) -> Entry:
        """Commit a decision, together with the action a policy took on it."""
        body: dict[str, Any] = {"decision": decision.as_dict()}
        if resolution is not None:
            body["resolution"] = resolution.to_dict()
        return self.append("decision", body)

    def record_outcome(self, key: str, label: str, *, note: str = "") -> Entry:
        """Commit a ground-truth label for an earlier decision."""
        return self.append("outcome", {"key": key, "label": label, "note": note})

    def entries(self) -> Iterator[Entry]:
        """Iterate the ledger in the order entries were written."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerIntegrityError(
                        f"line {number} of {self.path} is not valid JSON: {exc}"
                    ) from exc
                yield Entry(
                    seq=int(payload["seq"]),
                    prev=str(payload["prev"]),
                    kind=str(payload["kind"]),
                    body=dict(payload["body"]),
                    recorded_at=float(payload["recorded_at"]),
                    hash=str(payload["hash"]),
                )

    def verify(self) -> int:
        """Walk the chain and return the number of entries when it holds.

        Raises :class:`~metajev.errors.LedgerIntegrityError` naming the first entry
        that does not match, so a tampered record is located rather than merely
        reported.
        """
        expected_seq = 1
        expected_prev = GENESIS
        count = 0
        for entry in self.entries():
            if entry.seq != expected_seq:
                raise LedgerIntegrityError(
                    f"entry {expected_seq} is missing or out of order, found seq {entry.seq}"
                )
            if entry.prev != expected_prev:
                raise LedgerIntegrityError(
                    f"entry {entry.seq} points at {entry.prev[:12]}, expected {expected_prev[:12]}"
                )
            recomputed = _entry_hash(
                entry.seq, entry.prev, entry.kind, entry.body, entry.recorded_at
            )
            if recomputed != entry.hash:
                raise LedgerIntegrityError(
                    f"entry {entry.seq} has hash {entry.hash[:12]}, recomputed {recomputed[:12]}"
                )
            expected_prev = entry.hash
            expected_seq += 1
            count += 1
        return count

    def head(self) -> str:
        """Return the hash of the final entry, or the genesis hash when empty."""
        return self._last()[1]

    def count(self) -> int:
        """Return how many entries the ledger holds."""
        return sum(1 for _ in self.entries())
