"""The two primitives metajev is built on: typed questions and policy-free decisions.

A :class:`Question` is a typed request for a judgement. A :class:`Decision` is what a
model answered, recorded together with everything needed to reproduce it. A decision
carries no policy: it records the distribution the model produced and nothing about
what anyone intends to do with it. Thresholds, review bands, and routing live in
:mod:`metajev.policy` and are applied when a decision is read.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class QuestionType(str, Enum):
    """The three question shapes Jev-class models answer.

    ``NOUL`` asks a yes/no question and returns the probability of yes.
    ``CHOICE`` asks which of a set of candidates applies and returns a distribution
    over the candidates. ``SCORE`` asks for a position on an ordered scale and
    returns a distribution over the levels.
    """

    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class Action(str, Enum):
    """What a policy decides should happen to a decision."""

    ACCEPT = "accept"
    REVIEW = "review"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"
    DENY = "deny"


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` so that equal values always produce equal strings."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def content_hash(obj: Any) -> str:
    """Return the SHA-256 of the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Question:
    """A typed request for a judgement.

    ``text`` states what is being asked. ``type`` selects the answer shape.
    ``options`` lists the candidates for a ``CHOICE`` question. ``levels`` lists
    the ordered levels for a ``SCORE`` question, lowest first. ``yes`` optionally
    sharpens what counts as an affirmative answer to a ``NOUL`` question, which
    measurably improves calibration. ``tags`` label the question so that a policy
    can address a family of questions at once.
    """

    text: str
    type: QuestionType = QuestionType.NOUL
    options: tuple[str, ...] = ()
    levels: tuple[str, ...] = ()
    yes: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a question needs non-empty text")
        if self.type is QuestionType.CHOICE and len(self.options) < 2:
            raise ValueError("a choice question needs at least two options")
        if self.type is QuestionType.SCORE and len(self.levels) < 2:
            raise ValueError("a score question needs at least two levels")
        if len(set(self.options)) != len(self.options):
            raise ValueError("choice options must be distinct")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("score levels must be distinct")

    @property
    def id(self) -> str:
        """A stable identifier for this question, derived from its content."""
        return content_hash(
            {
                "text": self.text,
                "type": self.type.value,
                "options": list(self.options),
                "levels": list(self.levels),
                "yes": self.yes,
            }
        )

    @property
    def labels(self) -> tuple[str, ...]:
        """Every label this question can answer with, in presentation order."""
        if self.type is QuestionType.NOUL:
            return ("no", "yes")
        if self.type is QuestionType.CHOICE:
            return self.options
        return self.levels

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type.value,
            "options": list(self.options),
            "levels": list(self.levels),
            "yes": self.yes,
            "tags": list(self.tags),
        }

    @classmethod
    def noul(cls, text: str, yes: str | None = None, tags: Sequence[str] = ()) -> "Question":
        """Build a yes/no question, optionally sharpening what yes means."""
        return cls(text=text, type=QuestionType.NOUL, yes=yes, tags=tuple(tags))

    @classmethod
    def choice(cls, text: str, options: Iterable[str], tags: Sequence[str] = ()) -> "Question":
        """Build a question that picks one of ``options``."""
        return cls(text=text, type=QuestionType.CHOICE, options=tuple(options), tags=tuple(tags))

    @classmethod
    def score(cls, text: str, levels: Iterable[str], tags: Sequence[str] = ()) -> "Question":
        """Build a question that places the state on an ordered scale."""
        return cls(text=text, type=QuestionType.SCORE, levels=tuple(levels), tags=tuple(tags))


@dataclass(frozen=True)
class Decision:
    """What a model answered, with everything needed to reproduce and audit it.

    ``distribution`` holds the full answer distribution as ordered ``(label,
    probability)`` pairs, so a later reader can recompute an argmax, an expected
    value, or a threshold comparison without calling the model again. ``confidence``
    is the probability the model assigned to the answer it led with. ``cost_usd``
    and ``latency_ms`` record what the call actually cost.

    ``tags`` is copied from the question so that a policy can address a family of
    questions without holding the original :class:`Question`. Tags are deliberately
    kept out of :func:`decision_key`, since they organise questions rather than
    determine answers.
    """

    key: str
    question_id: str
    question_type: QuestionType
    state_hash: str
    provider: str
    model: str
    answer: str
    confidence: float
    distribution: tuple[tuple[str, float], ...]
    tags: tuple[str, ...] = ()
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    created_at: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)

    def probability_of(self, label: str) -> float:
        """Return the probability the model assigned to ``label``, or 0.0."""
        for candidate, probability in self.distribution:
            if candidate == label:
                return probability
        return 0.0

    def expected_level_index(self) -> float:
        """For a score decision, return the probability-weighted level position."""
        total = 0.0
        for index, (_, probability) in enumerate(self.distribution):
            total += index * probability
        return total

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "question_id": self.question_id,
            "question_type": self.question_type.value,
            "state_hash": self.state_hash,
            "provider": self.provider,
            "model": self.model,
            "answer": self.answer,
            "confidence": self.confidence,
            "distribution": [list(pair) for pair in self.distribution],
            "tags": list(self.tags),
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "created_at": self.created_at,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Decision":
        return cls(
            key=str(payload["key"]),
            question_id=str(payload["question_id"]),
            question_type=QuestionType(payload["question_type"]),
            state_hash=str(payload["state_hash"]),
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            answer=str(payload["answer"]),
            confidence=float(payload["confidence"]),
            distribution=tuple((str(a), float(b)) for a, b in payload["distribution"]),
            tags=tuple(str(t) for t in payload.get("tags", ())),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            created_at=float(payload.get("created_at", 0.0)),
            meta=dict(payload.get("meta", {})),
        )


@dataclass(frozen=True)
class Outcome:
    """Ground truth recorded against a decision, once it is known.

    Outcomes are what turn a decision store into a calibration record. They are
    recorded separately from decisions because the truth usually arrives later,
    and sometimes never.
    """

    key: str
    label: str
    weight: float = 1.0
    recorded_at: float = 0.0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "weight": self.weight,
            "recorded_at": self.recorded_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Outcome":
        return cls(
            key=str(payload["key"]),
            label=str(payload["label"]),
            weight=float(payload.get("weight", 1.0)),
            recorded_at=float(payload.get("recorded_at", 0.0)),
            note=str(payload.get("note", "")),
        )


def state_hash(state: str) -> str:
    """Hash the state a question was asked about."""
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def decision_key(state_hash_value: str, question: Question, provider: str, model: str) -> str:
    """Address a decision by everything that determines it, and nothing else.

    The key deliberately excludes thresholds, budgets, and every other policy input,
    so that two runs which ask the same question of the same model about the same
    state address the same record even when their policies disagree.
    """
    return content_hash(
        {
            "state": state_hash_value,
            "question": question.id,
            "provider": provider,
            "model": model,
        }
    )


def now() -> float:
    """Current wall-clock time as a float, in one place so tests can monkeypatch it."""
    return time.time()
