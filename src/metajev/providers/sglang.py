"""A provider for prefill-only decision servers built on SGLang.

Some open decision models answer without decoding anything. The server runs the
prompt through the model, reads the distribution over the label tokens at the
final position, and returns it. No token is generated, so latency is one forward
pass and cost is one prompt.

The route and the location of the distribution inside the response differ between
server builds. The path list below is tried in order, and ``distribution_paths``
can replace it when a deployment puts the numbers somewhere else.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

from ..errors import ProviderResponseError
from ..types import Question
from ._http import first_present
from .base import ProviderAnswer
from .http_base import HttpProvider

DEFAULT_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000/generate")

SGLANG_DISTRIBUTION_PATHS = (
    "probabilities",
    "meta_info.probabilities",
    "meta_info.output_token_logprobs_dist",
    "meta_info.distribution",
    "meta_info.label_probs",
    "output.probabilities",
)


class SGLangProvider(HttpProvider):
    """A provider for a prefill-only SGLang server.

    ``label_prefix`` is prepended to each label when the server expects the
    continuation under a leading space, which is how most tokenizers encode a
    mid-sentence token.
    """

    name = "sglang"
    model = "local"
    DEFAULT_ENDPOINT = DEFAULT_URL

    def __init__(
        self,
        url: str | None = None,
        *,
        api_key: str | None = None,
        model: str = "local",
        timeout: float = 30.0,
        cost_usd: float = 0.0,
        label_prefix: str = " ",
        distribution_paths: tuple[str, ...] = SGLANG_DISTRIBUTION_PATHS,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("SGLANG_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        super().__init__(
            url or DEFAULT_URL,
            name="sglang",
            model=model,
            headers=headers,
            timeout=timeout,
            cost_usd=cost_usd,
            distribution_paths=distribution_paths,
        )
        self.label_prefix = label_prefix
        self.extra_body = dict(extra_body or {})

    def payload(self, state: str, question: Question) -> dict[str, Any]:
        prompt = (
            f"{state}\n\nQuestion: {question.text}\n"
            f"Answer with one of: {', '.join(question.labels)}."
        )
        body: dict[str, Any] = {
            "text": prompt,
            "max_new_tokens": 0,
            "return_logprob": True,
            "logprob_start_len": 0,
            "labels": [f"{self.label_prefix}{label}" for label in question.labels],
        }
        body.update(self.extra_body)
        return body

    def extract(self, response: Mapping[str, Any], question: Question) -> ProviderAnswer:
        raw = first_present(response, self.distribution_paths)
        if raw is None:
            raise ProviderResponseError(
                f"sglang response carried no distribution at any of "
                f"{list(self.distribution_paths)}; body keys were {sorted(response.keys())}"
            )
        if isinstance(raw, list):
            # Servers that return logprobs rather than probabilities.
            converted = _from_logprob_entries(raw, self.label_prefix, question)
            if converted:
                answer = max(converted, key=lambda label: converted[label])
                return ProviderAnswer(
                    answer=answer,
                    distribution=tuple(converted.items()),
                    cost_usd=self.cost_usd,
                    meta={"url": self.url},
                )
        return super().extract(response, question)

    def health(self) -> tuple[bool, str]:
        return True, f"configured for {self.url}"


def _from_logprob_entries(
    entries: list[Any], prefix: str, question: Question
) -> dict[str, float]:
    """Read label probabilities out of a list of token logprob entries."""
    lookup = {f"{prefix}{label}".lower(): label for label in question.labels}
    lookup.update({label.lower(): label for label in question.labels})
    masses: dict[str, float] = {}
    for entry in entries:
        token = None
        logprob = None
        if isinstance(entry, Mapping):
            token = entry.get("token") or entry.get("label") or entry.get("text")
            logprob = entry.get("logprob", entry.get("log_prob"))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            token, logprob = entry[0], entry[1]
        if token is None or logprob is None:
            continue
        label = lookup.get(str(token).strip().lower())
        if label is None:
            continue
        masses[label] = masses.get(label, 0.0) + math.exp(float(logprob))
    return masses


__all__ = ["SGLangProvider", "SGLANG_DISTRIBUTION_PATHS"]
