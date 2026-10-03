"""Shared machinery for providers that talk to an HTTP endpoint.

Servers that answer Jev-style questions disagree about field names. One returns
``probabilities``, another ``probs``, a third nests the whole distribution under
``meta_info``. Rather than pick a winner, this module reads a list of candidate
paths in order and coerces whatever it finds into a distribution over the question's
own labels.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..errors import ProviderResponseError
from ..types import Question
from ._http import dig, first_present, post_json
from .base import Provider, ProviderAnswer

DEFAULT_DISTRIBUTION_PATHS = (
    "probabilities",
    "distribution",
    "probs",
    "scores",
    "meta_info.probabilities",
    "meta_info.distribution",
    "data.probabilities",
)
DEFAULT_ANSWER_PATHS = ("answer", "choice", "label", "result", "data.answer")
DEFAULT_CONFIDENCE_PATHS = ("confidence", "probability", "prob", "data.confidence")


def align_labels(raw: Any, question: Question) -> dict[str, float]:
    """Coerce a raw distribution into a mapping keyed by the question's labels.

    Handles the three shapes servers actually emit. A mapping keyed by label is
    taken as is, with case-only differences resolved. A mapping keyed by numeric
    index is matched to labels by position. A plain list of numbers is matched to
    labels by position.
    """
    labels = question.labels
    if isinstance(raw, Mapping):
        keys = [str(key) for key in raw.keys()]
        lowered = {key.lower(): key for key in keys}
        out: dict[str, float] = {}
        for label in labels:
            source = label if label in raw else lowered.get(label.lower())
            if source is not None:
                out[label] = float(raw[source])
        if out:
            return out
        numeric = all(key.lstrip("-").isdigit() for key in keys)
        if numeric and len(keys) == len(labels):
            return {label: float(raw[key]) for label, key in zip(labels, sorted(keys, key=int))}
        # Fall through: report what arrived so the caller can see the mismatch.
        raise ProviderResponseError(
            f"distribution keys {keys} match none of the question labels {list(labels)}"
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
        if len(values) == len(labels):
            return {label: float(value) for label, value in zip(labels, values)}
        raise ProviderResponseError(
            f"distribution has {len(values)} entries, question has {len(labels)} labels"
        )
    raise ProviderResponseError(f"distribution is a {type(raw).__name__}, expected object or list")


class HttpProvider(Provider):
    """A provider backed by a JSON HTTP endpoint.

    Subclasses supply the request body through :meth:`payload` and read the answer
    through :meth:`extract`. Everything else, including error handling and label
    alignment, is shared.
    """

    def __init__(
        self,
        url: str,
        *,
        name: str | None = None,
        model: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 20.0,
        cost_usd: float = 0.0,
        distribution_paths: Iterable[str] = DEFAULT_DISTRIBUTION_PATHS,
        answer_paths: Iterable[str] = DEFAULT_ANSWER_PATHS,
        confidence_paths: Iterable[str] = DEFAULT_CONFIDENCE_PATHS,
    ) -> None:
        super().__init__(name=name, model=model)
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.cost_usd = cost_usd
        self.distribution_paths = tuple(distribution_paths)
        self.answer_paths = tuple(answer_paths)
        self.confidence_paths = tuple(confidence_paths)

    def payload(self, state: str, question: Question) -> dict[str, Any]:
        """Build the request body. Subclasses override this."""
        raise NotImplementedError

    def _answer(self, state: str, question: Question) -> ProviderAnswer:
        response = post_json(
            self.url, self.payload(state, question), headers=self.headers, timeout=self.timeout
        )
        return self.extract(response, question)

    def extract(self, response: Mapping[str, Any], question: Question) -> ProviderAnswer:
        """Read a distribution, an answer, and a confidence out of a response body."""
        raw = first_present(response, self.distribution_paths)
        if raw is None:
            raise ProviderResponseError(
                f"{self.name} response carried no distribution at any of "
                f"{list(self.distribution_paths)}; body keys were {sorted(response.keys())}"
            )
        distribution = align_labels(raw, question)

        answer = first_present(response, self.answer_paths)
        if not isinstance(answer, str) or answer not in distribution:
            answer = max(distribution, key=lambda label: distribution[label])

        confidence = first_present(response, self.confidence_paths)
        meta: dict[str, Any] = {"url": self.url}
        usage = dig(response, "usage", None)
        if isinstance(usage, Mapping):
            meta["usage"] = dict(usage)
        return ProviderAnswer(
            answer=answer,
            distribution=tuple(distribution.items()),
            cost_usd=self.cost_usd,
            meta=meta,
        )

    def health(self) -> tuple[bool, str]:
        """Report whether the endpoint is configured, without calling it."""
        if not self.url:
            return False, "no url configured"
        return True, f"configured for {self.url}"

    def describe(self) -> dict[str, Any]:
        described = super().describe()
        described.update({"url": self.url, "cost_usd": self.cost_usd})
        return described


class BearerAuth:
    """Build an Authorization header from an API key, when one is present."""

    @staticmethod
    def headers(api_key: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}
