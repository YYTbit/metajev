"""Policies turn recorded decisions into actions, at read time.

A decision records a probability distribution. It does not record what anyone
wanted to do about it. That separation is the point of this package, and this
module is where the second half lives.

A :class:`Rule` selects the decisions it governs and carries an ordered list of
:class:`Band` objects. Each band covers a half-open interval of some quantity
derived from the decision, and names the action to take there. Rules are consulted
in order and the first match wins. Values are read off the stored distribution, so
changing a boundary costs nothing beyond re-reading the store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .errors import MetajevError
from .types import Action, Decision, QuestionType, content_hash


@dataclass(frozen=True)
class Band:
    """A half-open interval ``[lo, hi)`` and the action to take inside it."""

    lo: float
    hi: float
    action: Action
    note: str = ""

    def __post_init__(self) -> None:
        if self.lo >= self.hi:
            raise ValueError(f"band lower bound {self.lo} must be below upper bound {self.hi}")

    def contains(self, value: float) -> bool:
        """Return whether ``value`` falls inside this band."""
        return self.lo <= value < self.hi

    @property
    def name(self) -> str:
        return f"[{self.lo:g},{self.hi:g})"

    def to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "action": self.action.value, "note": self.note}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Band":
        return cls(
            lo=float(payload["lo"]),
            hi=float(payload["hi"]),
            action=Action(payload["action"]),
            note=str(payload.get("note", "")),
        )


def _normalise(value: float) -> float:
    """Fold a probability that may sit a hair outside [0, 1] back inside it."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


@dataclass(frozen=True)
class Rule:
    """Selects decisions and maps a quantity derived from them onto actions.

    ``on`` names the quantity the bands are compared against. Three forms exist.

    ``confidence`` is the probability of the answer the model led with, always in
    ``[0, 1]``. ``label:<name>`` is the probability of a named label, always in
    ``[0, 1]``. ``expected`` is a position on the answer scale normalised to
    ``[0, 1]``, equal to the probability of yes for a yes/no question, the leading
    probability for a choice question, and the probability-weighted level position
    for a score question.

    The selectors ``question_types``, ``question_ids``, ``tags``, ``providers``,
    and ``models`` narrow which decisions the rule governs. Each is a membership
    test, and an empty selector matches everything.
    """

    name: str
    bands: tuple[Band, ...]
    on: str = "confidence"
    question_types: tuple[QuestionType, ...] = ()
    question_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError(f"rule {self.name!r} needs at least one band")
        if not self.on:
            raise ValueError(f"rule {self.name!r} needs an 'on' quantity")

    def matches(self, decision: Decision) -> bool:
        """Return whether this rule governs ``decision``."""
        if self.question_types and decision.question_type not in self.question_types:
            return False
        if self.question_ids and decision.question_id not in self.question_ids:
            return False
        if self.tags and not set(self.tags) & set(decision.tags):
            return False
        if self.providers and decision.provider not in self.providers:
            return False
        if self.models and decision.model not in self.models:
            return False
        return True

    def value_for(self, decision: Decision) -> float:
        """Read the quantity named by ``on`` off a decision, normalised to [0, 1]."""
        if self.on == "confidence":
            return _normalise(decision.confidence)
        if self.on == "expected":
            if decision.question_type is QuestionType.SCORE:
                span = len(decision.distribution) - 1
                if span <= 0:
                    return _normalise(decision.confidence)
                return _normalise(decision.expected_level_index() / span)
            return _normalise(decision.confidence)
        if self.on.startswith("label:"):
            label = self.on[len("label:") :]
            if not label:
                raise MetajevError(f"rule {self.name!r} names an empty label")
            return _normalise(decision.probability_of(label))
        raise MetajevError(
            f"rule {self.name!r} asks for unknown quantity {self.on!r}; "
            "expected 'confidence', 'expected', or 'label:<name>'"
        )

    def band_for(self, value: float) -> Band | None:
        """Return the first band containing ``value``, or None."""
        for band in self.bands:
            if band.contains(value):
                return band
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "on": self.on,
            "bands": [band.to_dict() for band in self.bands],
            "question_types": [t.value for t in self.question_types],
            "question_ids": list(self.question_ids),
            "tags": list(self.tags),
            "providers": list(self.providers),
            "models": list(self.models),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Rule":
        return cls(
            name=str(payload["name"]),
            bands=tuple(Band.from_dict(b) for b in payload["bands"]),
            on=str(payload.get("on", "confidence")),
            question_types=tuple(
                QuestionType(t) for t in payload.get("question_types", ())
            ),
            question_ids=tuple(str(q) for q in payload.get("question_ids", ())),
            tags=tuple(str(t) for t in payload.get("tags", ())),
            providers=tuple(str(p) for p in payload.get("providers", ())),
            models=tuple(str(m) for m in payload.get("models", ())),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class Resolution:
    """What a policy decided should happen to one decision."""

    key: str
    action: Action
    policy: str
    rule: str
    band: str
    value: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "action": self.action.value,
            "policy": self.policy,
            "rule": self.rule,
            "band": self.band,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Policy:
    """An ordered set of rules, plus the action to take when none of them match.

    A policy holds no state and no model handle. Constructing one is free, which is
    what makes comparing several of them against the same recorded history cheap.
    """

    name: str
    rules: tuple[Rule, ...]
    default: Action = Action.REVIEW

    def resolve(self, decision: Decision) -> Resolution:
        """Apply this policy to a decision."""
        for rule in self.rules:
            if not rule.matches(decision):
                continue
            value = rule.value_for(decision)
            band = rule.band_for(value)
            if band is None:
                continue
            return Resolution(
                key=decision.key,
                action=band.action,
                policy=self.name,
                rule=rule.name,
                band=band.name,
                value=value,
                reason=f"{rule.name} {rule.on}={value:.4f} in {band.name}",
            )
        return Resolution(
            key=decision.key,
            action=self.default,
            policy=self.name,
            rule="",
            band="",
            value=_normalise(decision.confidence),
            reason=f"no rule matched, default {self.default.value}",
        )

    def resolve_all(self, decisions: Iterable[Decision]) -> list[Resolution]:
        """Apply this policy to several decisions."""
        return [self.resolve(decision) for decision in decisions]

    @property
    def fingerprint(self) -> str:
        """A stable hash of the rules, independent of what the policy is called.

        Record this alongside a resolution to know which rules produced it. The name
        is excluded so that renaming a policy does not invalidate every resolution
        already attributed to it, and so that two people who write the same bands
        agree on a fingerprint without having to agree on a label first.
        """
        payload = self.to_dict()
        payload.pop("name", None)
        return content_hash(payload)[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "default": self.default.value,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Policy":
        return cls(
            name=str(payload["name"]),
            rules=tuple(Rule.from_dict(r) for r in payload.get("rules", ())),
            default=Action(payload.get("default", "review")),
        )

    @classmethod
    def from_json(cls, text: str) -> "Policy":
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        """Render the policy as a short human-readable block."""
        lines = [f"policy {self.name} (fingerprint {self.fingerprint})"]
        for rule in self.rules:
            selectors = []
            if rule.question_types:
                selectors.append("types=" + ",".join(t.value for t in rule.question_types))
            if rule.question_ids:
                selectors.append(f"questions={len(rule.question_ids)}")
            if rule.tags:
                selectors.append("tags=" + ",".join(rule.tags))
            if rule.providers:
                selectors.append("providers=" + ",".join(rule.providers))
            scope = " ".join(selectors) if selectors else "all decisions"
            lines.append(f"  {rule.name} on {rule.on} [{scope}]")
            for band in rule.bands:
                lines.append(f"    {band.name:<14} {band.action.value}")
        lines.append(f"  default: {self.default.value}")
        return "\n".join(lines)


def three_band_policy(
    name: str = "default",
    *,
    accept: float = 0.85,
    review: float = 0.55,
    below: Action = Action.ESCALATE,
) -> Policy:
    """Build the policy shape most integrations end up writing by hand.

    Confident answers are accepted, middling ones go to a human, and everything
    below the review floor takes a fallback action. Stating it once makes the two
    boundaries the only things worth arguing about.
    """
    if not 0.0 <= review < accept <= 1.0:
        raise ValueError("expected 0 <= review < accept <= 1")
    return Policy(
        name=name,
        rules=(
            Rule(
                name="confidence",
                on="confidence",
                bands=(
                    Band(accept, 1.0001, Action.ACCEPT, "confident"),
                    Band(review, accept, Action.REVIEW, "worth a look"),
                    Band(0.0, review, below, "no confident answer"),
                ),
            ),
        ),
        default=Action.REVIEW,
    )


def label_policy(
    label: str,
    name: str | None = None,
    *,
    accept: float = 0.7,
    review: float = 0.4,
    below: Action = Action.ACCEPT,
) -> Policy:
    """Build a policy that acts on the probability of one named label.

    This is the shape a gate takes. The question is whether some property holds,
    and the bands say what to do as that probability rises.
    """
    return Policy(
        name=name or f"label-{label}",
        rules=(
            Rule(
                name=f"p({label})",
                on=f"label:{label}",
                bands=(
                    Band(accept, 1.0001, Action.DENY, "the property holds"),
                    Band(review, accept, Action.REVIEW, "borderline"),
                    Band(0.0, review, below, "the property is absent"),
                ),
            ),
        ),
        default=Action.REVIEW,
    )
