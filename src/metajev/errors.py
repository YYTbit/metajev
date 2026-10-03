"""Exception types raised by metajev."""

from __future__ import annotations


class MetajevError(Exception):
    """Base class for every error metajev raises."""


class ProviderError(MetajevError):
    """A provider failed to produce a decision."""


class ProviderUnavailable(ProviderError):
    """A provider could not be reached, or timed out."""


class ProviderAuthError(ProviderError):
    """A provider rejected the credentials it was given."""


class ProviderResponseError(ProviderError):
    """A provider answered, but the answer could not be read as a decision."""


class BudgetExceeded(MetajevError):
    """A call was refused because it would pass a configured budget limit."""


class LedgerIntegrityError(MetajevError):
    """A ledger failed hash-chain verification."""


class StoreError(MetajevError):
    """The decision store could not be read or written."""
