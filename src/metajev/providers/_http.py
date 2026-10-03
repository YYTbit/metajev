"""A small JSON-over-HTTP helper, so every provider fails the same way.

Providers in this package talk to several servers, and each of them can be missing,
slow, unauthorised, or answering with something that is not JSON. Folding that into
one function keeps the provider modules readable and keeps error reporting uniform.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from ..errors import ProviderAuthError, ProviderResponseError, ProviderUnavailable

DEFAULT_TIMEOUT = 20.0


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded response body."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    request.add_header("User-Agent", "metajev/0.1")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    return _send(request, timeout)


def get_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """GET ``url`` and return the decoded response body."""
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    request.add_header("User-Agent", "metajev/0.1")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    return _send(request, timeout)


def _send(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:400]
        except Exception:  # pragma: no cover - the body may be unreadable
            detail = ""
        if exc.code in (401, 403):
            raise ProviderAuthError(
                f"{request.full_url} rejected the credentials with HTTP {exc.code}. {detail}".strip()
            ) from exc
        raise ProviderUnavailable(
            f"{request.full_url} answered HTTP {exc.code}. {detail}".strip()
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderUnavailable(f"{request.full_url} could not be reached: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderUnavailable(f"{request.full_url} timed out after {timeout}s") from exc

    if not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(
            f"{request.full_url} returned a body that is not JSON: {raw[:200]}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ProviderResponseError(
            f"{request.full_url} returned {type(decoded).__name__}, expected an object"
        )
    return decoded


def dig(obj: Any, path: str | None, default: Any = None) -> Any:
    """Read a nested value by dotted path, with numeric segments indexing lists.

    ``dig(payload, "meta_info.probabilities")`` and ``dig(payload, "choices.0.text")``
    both work. A missing segment yields ``default`` rather than raising, so a
    provider can try several paths and take the first that resolves.
    """
    if not path:
        return obj
    current = obj
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return default
            current = current[segment]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(segment)
            except ValueError:
                return default
            if index < 0 or index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current


def first_present(obj: Any, paths: tuple[str, ...], default: Any = None) -> Any:
    """Return the first path that resolves to something, in order."""
    for path in paths:
        value = dig(obj, path, None)
        if value is not None:
            return value
    return default
