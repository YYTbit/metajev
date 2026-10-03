# metajev

Decision models answer typed questions with probability distributions. What you do
about those distributions is a separate question, and most integrations answer both
in the same function, with a threshold written next to the API call.

metajev keeps them apart. It records the distribution under a key built from the
state, the question, and the model, and from nothing else. Thresholds, review bands,
routing, and spending limits live in a policy that reads the record. Moving a
threshold then costs a pass over stored numbers.

```
pip install metajev
```

## The problem

A Jev-class model returns a distribution in a single forward pass and generates no
tokens. Wiring one into a pipeline looks like this:

```python
result = jev.ask(state, "Does this ticket need a person?")
if result.probability > 0.8:          # <- the policy
    escalate(result)
```

The number `0.8` is now load-bearing and invisible. Three things follow from where
it sits.

Changing it means re-running the model over everything, because the only copy of
the probability was the return value of a call that already happened. Comparing two
thresholds means two full runs, so in practice nobody compares them and the first
number anyone typed survives.

The probability is discarded after the comparison. When you later want to know
whether the model was right, the evidence is gone, and the only way to find out is
another run.

Every integration repeats this. Search the ecosystem and you find `0.30` here,
`0.7 / 0.4 / 0.5` there, a review band somewhere else, each one written by hand,
each one in a different place, and one project advertising "re-scoring for free when
the policy changes" as a feature, which is a fair description of what it is.

## The idea

A decision is a pure function of the state, the question, and the model. Thresholds
are not inputs to it. So they do not belong in its cache key.

metajev computes

```
key = sha256(state, question, provider, model)
```

and stores the full distribution under it. A policy is a separate object with bands
and actions. Reading a decision means looking it up and handing it to the policy.
Nothing in that path calls a model, so the policy can change as often as you like.

```python
import metajev

client = metajev.Client(
    provider=metajev.TypesafeProvider(),           # or sglang, openai-compat, mock
    store="decisions.db",
    policy=metajev.three_band_policy(accept=0.85, review=0.55),
)

answer = client.decide("the invoice total is negative", metajev.Question.noul(
    "Does this need a person before anything happens?",
    yes="a human should read this before any automated reply goes out",
))
# Action.REVIEW, confidence 0.58, key 8f3c...
```

A week later, with a different boundary and no model calls:

```python
report = client.replay(metajev.three_band_policy(accept=0.93, review=0.60))
print(report.render())
```

## What a boundary change costs

Moving a threshold is a trade, and a report that only counted what it caught would
be advertising. `replay` reports both sides, from a run over 500 tickets where the
model was judged once each and the truth was recorded afterwards:

```
policy strict (fingerprint 365bcbe2ca3c22d7)
replayed 500 decisions, 500 with a recorded outcome

action       count    share  judged  correct  accuracy
accept          95    19.0%      95       77     0.811
review         320    64.0%     320      196     0.613
escalate        85    17.0%      85       31     0.365

precision among accepted decisions: 0.811

against baseline policy loose
  actions that change:        318
  errors moved off accept:    75
  errors newly admitted:       0
  correct answers diverted:   158
```

Tightening from 0.70 to 0.90 moves 75 wrong answers out of `accept`. It also moves
158 right answers out to a human. Whether that is worth it depends on what a wrong
automatic reply costs against a delayed right one, which is a question about your
business and not about the model. What metajev does is make the exchange rate
visible, in one pass, for the price of reading a file.

Sweeping the boundary produces the whole curve:

```
  accept   changed  accepted  precision  errors caught  right lost
------------------------------------------------------------------
    0.60        87       415      0.658              0           0
    0.70         0       328      0.716              0           0
    0.80       113       215      0.786             47          66
    0.85       177       151      0.821             66         111
    0.90       233        95      0.811             75         158
    0.95       286        42      0.810             85         201
```

Run it yourself with no network and no key:

```
python examples/triage.py
```

## Providers

One interface, four backends, chosen per question.

| Provider | For | Distribution from |
|---|---|---|
| `TypesafeProvider` | the hosted Jev endpoint | the `probabilities` field of the response |
| `SGLangProvider` | prefill-only decision servers | a configurable path inside `meta_info` |
| `OpenAICompatProvider` | open Jev-class models behind `/chat/completions` | the token logprobs of the first position |
| `MockProvider` | tests, demos, dry runs | a hash of the inputs, offline and deterministic |

Servers disagree about where the numbers live, so each provider reads a list of
candidate paths in order and coerces what it finds onto the question's own labels.
A provider that returns `{"0": 0.2, "1": 0.8}` for a two-level score question is
understood, and so is one that returns a bare list. A provider that returns
something genuinely unreadable is reported with the keys it did send, rather than
silently producing a wrong answer.

`build_provider("sglang:my-model")` constructs one from a short spec.

## Policies

A policy is an ordered list of rules, each with bands and selectors.

```python
policy = metajev.Policy(
    name="triage",
    rules=[
        metajev.Rule(
            name="ticket urgency",
            on="label:yes",                     # or "confidence" or "expected"
            tags=("triage",),                   # only questions tagged triage
            bands=[
                metajev.Band(0.90, 1.01, metajev.Action.ESCALATE),
                metajev.Band(0.40, 0.90, metajev.Action.REVIEW),
                metajev.Band(0.00, 0.40, metajev.Action.ACCEPT),
            ],
        ),
    ],
    default=metajev.Action.REVIEW,
)
```

`on` names the quantity the bands are compared against. `confidence` is the
probability of the answer the model led with, which asks whether the model knows
when it knows. `label:yes` is the probability of a named label, which asks whether
that label's probability can be read as a frequency. `expected` is a position on the
answer scale normalised to `[0, 1]`, which is what an ordered score question needs.

Selectors narrow a rule to a family of questions by tag, type, provider, or model.
A policy with no matching rule takes its `default`.

Policies serialise to JSON, and carry a fingerprint that covers the rules and not
the name, so renaming one does not invalidate the resolutions already attributed to
it.

## Calibration

Probabilities are useful when they can be read as frequencies. Point a client at a
store where outcomes have been recorded and it will say whether they can.

```
calibration for confidence
  observations:      500
  brier score:       0.2332   (lower is better, 0.25 is a coin at p=0.5)
  expected cal err:  0.1504
  mean predicted:    0.7584
  observed accuracy: 0.6080
  gap:               +0.1504  (overconfident)

  bin               count  predicted   actual      gap
  [0.5,0.6)            85      0.551    0.365   +0.187
  [0.6,0.7)            87      0.648    0.437   +0.211
  [0.7,0.8)           113      0.749    0.584   +0.165
  [0.8,0.9)           120      0.847    0.767   +0.081
  [0.9,1.0)            95      0.944    0.811   +0.134
```

`threshold_table` turns the same data into the decision view: for each candidate
boundary, how many wrong answers it admits and how many right answers it turns away.

## Routing

A client built from several providers routes each question to the cheapest one whose
observed accuracy on that question clears a floor. Observed accuracy comes from the
store, so routing sharpens as outcomes accumulate, and a provider with no recorded
outcomes is not penalised for being new. A `Budget` caps spending and call count, and
is checked before a call rather than after.

```python
client = metajev.Client(
    providers=[metajev.TypesafeProvider(), metajev.SGLangProvider()],
    store="decisions.db",
    budget=metajev.Budget(usd_limit=5.0, call_limit=50_000),
)
```

## Ledger

A store is a working database and its rows can be rewritten. When a decision has to
be defensible afterwards, `metajev.Ledger` writes an append-only JSONL record where
each entry commits to the hash of the one before it. Editing or dropping an entry
breaks every hash after it, and `verify()` names the first entry that fails.

```
metajev ledger verify decisions.jsonl
```

## Command line

```
metajev ask "the invoice total is negative" \
    --question "Does this need a person?" --yes "a human should read this first" \
    --provider typesafe --store decisions.db

metajev calibrate --store decisions.db --on confidence
metajev sweep     --store decisions.db --values 0.6,0.7,0.8,0.9,0.95
metajev replay    --store decisions.db --preset strict --baseline-preset balanced
metajev threshold --store decisions.db --steps 20
metajev ledger verify decisions.jsonl
metajev doctor --provider typesafe --store decisions.db
```

`metajev ask` with `--provider mock` runs offline, which is the fastest way to check
a question's wording before it goes into a pipeline.

## Design notes

**Why tags are not in the key.** Tags organise questions for policy authors. Two
questions that differ only in tags are the same question, and asking a model twice
because someone relabelled a tag would be a waste.

**Why a repeated decision is not overwritten.** A decision is a record of what a
model said. Re-asking the same question of the same model is expected to reproduce
the same distribution, so the first recording is kept and the duplicate is reported
as not new.

**Why a missing label is padded with zero rather than dropped.** A score question's
levels are positional. Dropping a level a provider omitted shifts every level after
it and silently corrupts the expected position, so omitted levels carry zero mass and
keep their place.

**Why the mock exists.** A pipeline that needs a paid key before it can be run is a
pipeline nobody runs. The mock answers from a hash of its inputs, so a fresh clone
can exercise the whole path offline and in a test.

## Limits

Storage grows with distinct questions rather than with calls, so a workload that
generates a new state on every request gains little from caching. Decisions are
immutable, so a distribution that a model would now produce differently is not
refreshed; record an outcome and re-ask if the model changed.

## Related work

[jev-cascade](https://github.com/fstandhartinger/jev-cascade) and
[hermes-jev-skills](https://github.com/kerpopule/hermes-jev-skills) route requests
across decision models. metajev is complementary to routing: it records what a
model answered and lets the action be recomputed later.

## License

MIT. Copyright YYTbit.
