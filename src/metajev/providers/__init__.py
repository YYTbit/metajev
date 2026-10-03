"""Providers that answer typed questions, and the registry that builds them."""

from __future__ import annotations

from .base import Provider, ProviderAnswer, normalise_distribution
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider
from .sglang import SGLangProvider
from .typesafe import TypesafeProvider

__all__ = [
    "Provider",
    "ProviderAnswer",
    "normalise_distribution",
    "MockProvider",
    "OpenAICompatProvider",
    "SGLangProvider",
    "TypesafeProvider",
    "PROVIDER_TYPES",
    "build_provider",
]


PROVIDER_TYPES: dict[str, type[Provider]] = {
    "typesafe": TypesafeProvider,
    "openai-compat": OpenAICompatProvider,
    "openai": OpenAICompatProvider,
    "sglang": SGLangProvider,
    "mock": MockProvider,
}


def build_provider(spec: str, **kwargs: object) -> Provider:
    """Build a provider from a short spec such as ``typesafe`` or ``sglang:my-model``.

    Everything after a colon is passed as the model name. Extra keyword arguments
    go to the provider constructor, so ``build_provider("typesafe", timeout=5)``
    works as expected.
    """
    name, _, model = spec.partition(":")
    key = name.strip().lower()
    if key not in PROVIDER_TYPES:
        raise ValueError(
            f"unknown provider {name!r}; known providers are {sorted(PROVIDER_TYPES)}"
        )
    if model:
        kwargs = {**kwargs, "model": model}
    return PROVIDER_TYPES[key](**kwargs)
