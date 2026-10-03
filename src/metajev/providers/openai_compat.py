"""A provider for any OpenAI-compatible endpoint serving an open decision model.

Several open Jev-class models ship behind a stock ``/chat/completions`` route. The
distribution is recovered from the token logprobs of the first generated position:
the model is asked to name one label, decoding is capped at a single token, and the
probability mass on each label token is read straight out of ``top_logprobs``.

This works when every label is a single token in the serving tokenizer, which holds
for short labels like ``yes``, ``no``, ``invoice``, or a numbered scale. Labels that
tokenise into several pieces are rejected at call time with a message naming the
offending label, so the failure is legible rather than a silent misread.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Mapping

from ..errors import ProviderResponseError
from ..types import Question
from .http_base import HttpProvider

DEFAULT_URL = os.environ.get("OPENAI_COMPAT_URL", "http://127.0.0.1:8080/v1/chat/completions")

PROMPT_TEMPLATE = """You are answering one typed question about a state.

State:
{state}

Question: {question}

Reply with exactly one of the following labels, and nothing else:
{labels}"""


class OpenAICompatProvider(HttpProvider):
    """A provider for an OpenAI-compatible chat completions route.

    ``fractional_cost`` lets a local deployment report a nonzero cost, which is
    useful when the model runs on rented hardware and the budget should account
    for it. The default of zero matches a model on your own machine.
    """

    name = "openai-compat"
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
        top_logprobs: int = 20,
        prompt_template: str = PROMPT_TEMPLATE,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENAI_COMPAT_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        super().__init__(url or DEFAULT_URL, name="openai-compat", model=model,
                          headers=headers, timeout=timeout, cost_usd=cost_usd)
        self.top_logprobs = top_logprobs
        self.prompt_template = prompt_template
        self.extra_body = dict(extra_body or {})

    def payload(self, state: str, question: Question) -> dict[str, Any]:
        labels = "\n".join(f"- {label}" for label in question.labels)
        prompt = self.prompt_template.format(
            state=state, question=question.text, labels=labels
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        body.update(self.extra_body)
        return body

    def extract(self, response: Mapping[str, Any], question: Question) -> Any:
        entries = _top_logprobs(response)
        if not entries:
            raise ProviderResponseError(
                "response carried no token logprobs; the server must be started with "
                "logprobs support and the request must set max_tokens to 1"
            )

        lookup = {label.lower(): label for label in question.labels}
        masses: dict[str, float] = {}
        for entry in entries:
            token = str(entry.get("token", "")).strip().lower()
            if not token:
                continue
            label = lookup.get(token)
            if label is None:
                # A model may answer "1" for the first option on a scale.
                if token.rstrip(".:)").isdigit():
                    index = int(token.rstrip(".:)")) - 1
                    if 0 <= index < len(question.labels):
                        label = question.labels[index]
                if label is None:
                    continue
            logprob = entry.get("logprob")
            if logprob is None:
                continue
            masses[label] = masses.get(label, 0.0) + math.exp(float(logprob))

        if not masses:
            seen = [str(e.get("token", "")).strip() for e in entries[:10]]
            raise ProviderResponseError(
                f"none of the returned tokens {seen} match the question labels "
                f"{list(question.labels)}; each label must be a single token"
            )

        answer = max(masses, key=lambda label: masses[label])
        return self._assemble(answer, masses, response)

    def _assemble(
        self, answer: str, masses: Mapping[str, float], response: Mapping[str, Any]
    ) -> Any:
        from .base import ProviderAnswer

        meta: dict[str, Any] = {"url": self.url}
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            meta["usage"] = dict(usage)
        return ProviderAnswer(
            answer=answer,
            distribution=tuple(masses.items()),
            cost_usd=self.cost_usd,
            meta=meta,
        )

    def health(self) -> tuple[bool, str]:
        return True, f"configured for {self.url}"


def _top_logprobs(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pull the first position's top logprobs out of a chat completion response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first = choices[0]
    if not isinstance(first, Mapping):
        return []
    logprobs = first.get("logprobs")
    if not isinstance(logprobs, Mapping):
        return []
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        entry = content[0]
        if isinstance(entry, Mapping):
            top = entry.get("top_logprobs")
            if isinstance(top, list):
                return [e for e in top if isinstance(e, dict)]
    # Older servers put the alternatives at the top level of `logprobs`.
    top = logprobs.get("top_logprobs")
    if isinstance(top, list) and top and isinstance(top[0], list):
        return [e for e in top[0] if isinstance(e, dict)]
    return []


__all__ = ["OpenAICompatProvider", "PROMPT_TEMPLATE", "Iterable"]
