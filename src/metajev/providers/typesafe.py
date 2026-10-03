"""The hosted TypeSafe Jev endpoint.

This is the reference backend. It answers a typed question about a state and
returns a probability distribution, with no generated text. The request shape
follows the documented ``/v1/systemone`` contract, and the response reader is
tolerant about field names because deployments differ.
"""

from __future__ import annotations

import os
from typing import Any

from ..types import Question, QuestionType
from .http_base import BearerAuth, HttpProvider

DEFAULT_URL = os.environ.get("JEV_API_URL", "https://api.typesafe.ai/v1/systemone")


class TypesafeProvider(HttpProvider):
    """A provider for the hosted Jev endpoint.

    The API key is read from the ``api_key`` argument, then from ``JEV_API_KEY``.
    A missing key is not an error at construction time, so a stack can be built and
    inspected before credentials are wired in; the call itself will fail with a
    clear message if the key never arrives.
    """

    name = "typesafe"
    model = "jev"
    DEFAULT_ENDPOINT = DEFAULT_URL

    def __init__(
        self,
        url: str | None = None,
        *,
        api_key: str | None = None,
        model: str = "jev",
        timeout: float = 20.0,
        cost_usd: float = 0.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("JEV_API_KEY")
        merged = BearerAuth.headers(key)
        merged.update(headers or {})
        super().__init__(
            url or DEFAULT_URL,
            name="typesafe",
            model=model,
            headers=merged,
            timeout=timeout,
            cost_usd=cost_usd,
        )
        self.has_key = bool(key)

    def payload(self, state: str, question: Question) -> dict[str, Any]:
        body: dict[str, Any] = {
            "state": state,
            "question": question.text,
            "type": question.type.value,
        }
        if question.type is QuestionType.CHOICE:
            body["options"] = list(question.options)
        elif question.type is QuestionType.SCORE:
            body["levels"] = list(question.levels)
        elif question.type is QuestionType.NOUL and question.yes:
            body["yes"] = question.yes
        return body

    def health(self) -> tuple[bool, str]:
        if not self.has_key:
            return False, "no API key; set JEV_API_KEY or pass api_key"
        return True, f"configured for {self.url}"

    def describe(self) -> dict[str, Any]:
        described = super().describe()
        described["has_key"] = self.has_key
        return described
