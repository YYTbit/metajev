# Draft posts

Two versions, for two different registers. Neither has been posted.

## Show HN

**Title**

```
Show HN: Metajev – record typed decisions, change the threshold without re-running the model
```

**Body**

Most code that uses a decision model looks like this:

```python
result = jev.ask(state, "Does this ticket need a person?")
if result.probability > 0.8:
    escalate(result)
```

The model returns a calibrated probability in one forward pass and generates no
tokens. The `0.8` is load-bearing and nothing in the program knows it exists. Change
it and you have to re-run the model over everything, because the only copy of the
probability was a return value from a call that already happened. So nobody changes
it, and the first number anyone typed survives.

The observation is small. Ask a model the same question about the same state and it
produces the same distribution, so a decision is a function of the state, the
question, and the model. Which boundary you later compare it against is not an input
to that function, and has no business in its cache key.

Metajev records the full distribution under `sha256(state, question, provider,
model)` and keeps thresholds, review bands, and routing in a policy object that
reads the record. "What if the line were somewhere else" becomes a pass over stored
numbers.

The part I care about is that the replay report shows both sides. On a run of 500
tickets, moving the boundary from 0.70 to 0.90 keeps 75 wrong answers out of
automatic handling and sends 158 right ones to a person. Whether that is worth it is
a business question. Seeing the exchange rate in one pass, for free, is the point.

No dependencies, an offline provider so a fresh clone runs the whole path without a
key, and a self-contained HTML report with a slider that drags the boundary across a
recorded history in the browser.

`pip install "metajev @ git+https://github.com/YYTbit/metajev"` while it is not on
PyPI yet. Longer write-up in `docs/why-decisions-and-policies-are-different.md`.

## Reddit

**Title**

```
The threshold in your decision-model code is load-bearing and invisible
```

**Body**

Short version of something I kept running into. Code that uses a decision model
usually ends up as `if result.probability > 0.8: act()`. The model is cheap and
fast, and the `0.8` decides everything.

Three consequences. You cannot change the threshold without re-running the model
over all your data, so in practice you do not change it. You cannot learn anything
from it, because the probability is compared and then discarded, so six weeks later
nobody can say how often a 0.9 was actually right. And every project in the
ecosystem reimplements the same threshold-plus-review-band logic by hand, in a
different file, with different numbers.

The fix is one observation: a decision is a pure function of the state, the
question, and the model. The threshold is not one of its inputs, so it does not
belong in the cache key. Record the distribution, keep the policy separate, and
changing the boundary becomes a read instead of a re-run.

I built a small library around that. The bit that made it click for me is the
report. It shows what a boundary change catches and what it costs, because a tool
that only counted the errors it caught would be selling something:

```
  accept   changed  accepted  precision  errors caught  right lost
    0.70         0       328      0.716              0           0
    0.85       177       151      0.821             66         111
    0.90       233        95      0.811             75         158
```

75 wrong answers kept out of automatic handling, 158 right ones sent to a human.
Whether that trade is good depends on your business. Seeing it in one pass, with no
model calls, is the difference between choosing a threshold and inheriting one.

Repo and a longer write-up in the comments.

## Notes for posting

- Post the Show HN in the morning US Eastern on a weekday.
- The first comment should link the longer essay and mention the offline provider,
  since the fastest way to judge the idea is to run it with no key.
- If someone asks why not just cache the model call: caching the call still couples
  you to a threshold, because the cache key has to include it or you get the wrong
  answer back. Caching the decision is what lets the policy move.
