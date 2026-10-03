"""metajev keeps decisions and the policies that act on them apart.

A decision model answers a typed question about a state and returns a distribution.
What you do about that distribution is a separate question, and mixing the two is
why changing a threshold usually means re-running everything.

This package records the distribution under a key derived from the state, the
question, and the model, and nothing else. Thresholds, review bands, routing, and
spending limits live in a policy that reads the store. Moving a threshold becomes a
pass over recorded numbers instead of a round of model calls, and the cost of the
change is reported alongside the benefit.

    import metajev

    client = metajev.Client(
        provider=metajev.TypesafeProvider(api_key="..."),
        store="decisions.db",
        policy=metajev.three_band_policy(accept=0.85, review=0.55),
    )

    answer = client.decide("the invoice total is negative", metajev.Question.noul(
        "Does this message need a human?",
        yes="a person should read this before anything happens",
    ))
    print(answer.action, answer.decision.confidence)

    client.record_outcome(answer, "yes")

    # Later, with no model calls:
    report = client.replay(metajev.three_band_policy(accept=0.70, review=0.40))
    print(report.render())
"""

from .budget import Budget, BudgetSnapshot
from .calibrate import (
    Calibration,
    Observation,
    ReliabilityBin,
    brier,
    expected_calibration_error,
    observations_from_store,
    reliability,
    summarise,
    threshold_table,
)
from .client import Answer, Client
from .errors import (
    BudgetExceeded,
    LedgerIntegrityError,
    MetajevError,
    ProviderAuthError,
    ProviderError,
    ProviderResponseError,
    ProviderUnavailable,
    StoreError,
)
from .ledger import Entry, Ledger
from .policy import Band, Policy, Resolution, Rule, label_policy, three_band_policy
from .providers import (
    MockProvider,
    OpenAICompatProvider,
    Provider,
    ProviderAnswer,
    SGLangProvider,
    TypesafeProvider,
    build_provider,
)
from .replay import Bucket, Flip, ReplayReport, compare, replay, sweep
from .router import ProviderScore, RouteDecision, Router, score_providers
from .store import DecisionStore, StoreStats
from .types import (
    Action,
    Decision,
    Outcome,
    Question,
    QuestionType,
    content_hash,
    decision_key,
    state_hash,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # types
    "Action",
    "Decision",
    "Outcome",
    "Question",
    "QuestionType",
    "content_hash",
    "decision_key",
    "state_hash",
    # store
    "DecisionStore",
    "StoreStats",
    # policy
    "Band",
    "Policy",
    "Resolution",
    "Rule",
    "label_policy",
    "three_band_policy",
    # client
    "Answer",
    "Client",
    # replay
    "Bucket",
    "Flip",
    "ReplayReport",
    "compare",
    "replay",
    "sweep",
    # calibration
    "Calibration",
    "Observation",
    "ReliabilityBin",
    "brier",
    "expected_calibration_error",
    "observations_from_store",
    "reliability",
    "summarise",
    "threshold_table",
    # routing
    "ProviderScore",
    "RouteDecision",
    "Router",
    "score_providers",
    # budget
    "Budget",
    "BudgetSnapshot",
    # ledger
    "Entry",
    "Ledger",
    # providers
    "MockProvider",
    "OpenAICompatProvider",
    "Provider",
    "ProviderAnswer",
    "SGLangProvider",
    "TypesafeProvider",
    "build_provider",
    # errors
    "BudgetExceeded",
    "LedgerIntegrityError",
    "MetajevError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderUnavailable",
    "StoreError",
]
