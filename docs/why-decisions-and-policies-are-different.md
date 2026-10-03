# Your threshold is load-bearing and invisible

Here is the code almost everyone writes when they wire a decision model into a
pipeline.

```python
result = jev.ask(state, "Does this ticket need a person?")
if result.probability > 0.8:
    escalate(result)
```

The model returns a probability in one forward pass and generates no tokens. The
call is cheap. The number `0.8` is doing a lot of work and nothing in the program
knows it exists.

Three things follow from where that number sits.

## It cannot be changed without paying again

The only copy of the probability was the return value of a call that already
happened. Change `0.8` to `0.9` and the new boundary has to be applied to data you
no longer have, so you re-run the model over everything.

In practice this means nobody changes it. The first number anyone typed survives,
not because it was right, but because testing a second one costs a full pass and
nobody has a reason to spend one.

## Nothing can be learned from it

The probability is discarded after the comparison. Six weeks later, when someone
asks whether the classifier is any good, the evidence is gone. You cannot ask how
often a `0.9` was actually correct, because the `0.9` was thrown away.

The cost of that is easy to underestimate. It is not that you lack a metric. It is
that you cannot answer the question that decides whether the threshold means
anything, which is whether a reported `0.9` happens four times in five or six.

## Everyone reimplements the same three lines

Search any ecosystem that has a decision model in it and you find the same
vocabulary invented over and over. A `0.30` here, a `0.7 / 0.4 / 0.5` there, a
review band in a third place, each written by hand, each in a different file. One
project advertises "re-scoring for free when the policy changes" as a feature,
which is an accurate description of how rare it is.

## A decision is a pure function. A threshold is not an input to it.

This is the whole observation.

Ask a model the same question about the same state and it produces the same
distribution. That distribution is a function of three things: the state, the
question, and the model. Which boundary you later compare it against is not one of
them.

So the boundary has no business in the cache key.

```
key = sha256(state, question, provider, model)
```

Record the full distribution under that key. Keep thresholds, review bands, and
routing in a separate object that reads the record. Then the operation people
actually want, "what if the line were somewhere else", becomes a pass over stored
numbers.

```python
answer = client.decide(ticket, QUESTION)      # one model call, recorded

report = client.replay(accept=0.90)           # a different boundary, no calls
print(report.render())
```

## What it buys

The report has to show both sides or it is advertising. A boundary that diverts
errors also diverts correct answers, and a tool that counts only the first is
selling something.

Here is a real run. Five hundred tickets, each judged once by a model whose
confidence orders tickets correctly but runs about 0.15 hot, with the truth
recorded afterwards.

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

Moving the boundary from 0.70 to 0.90 keeps 75 wrong answers out of automatic
handling and sends 158 right answers to a person. Whether that trade is worth
making depends on what a wrong automatic reply costs against a delayed right one,
which is a question about your business and not about the model.

What the tool does is make the exchange rate visible in one pass, for the price of
reading a file. That is the difference between choosing a threshold and inheriting
one.

The same history, applied to the question of whether the probabilities can be read
as frequencies at all:

```
calibration for confidence
  observations:      500
  brier score:       0.2332
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

A model that says 0.94 and is right 0.81 of the time is still useful, since the
ordering holds. It is not useful if you read 0.94 as a frequency, and the report
says which one you have.

## The extra step that makes it pay

If all you do is cache decisions, you have made the boundary changeable and
learned nothing. The value shows up when outcomes are recorded.

An outcome is the truth for a decision, recorded whenever you learn it, which is
usually later and sometimes never. With outcomes, three things become possible
that were not before.

The precision of an accepted decision becomes a measurement instead of an
assumption. A boundary can be chosen against the errors it admits and the right
answers it turns away, from data. And a routing layer can send each question to
the cheapest provider whose observed accuracy on that question clears a bar,
which improves on its own as outcomes accumulate.

None of that needs the model to be called again. It needs the answers to have been
kept.

## When this does not help

Storage grows with distinct questions, so a workload that generates a brand new
state on every request gains little from caching. The first pass still costs what
it costs, and replay only pays for itself on the second policy.

Decisions are immutable, so a distribution that a model would now produce
differently is not refreshed. If the model changed, the recorded decision is a
record of what the old one said. That is a property, not a defect, but it means
the store is not a cache you can invalidate and expect the same keys to refill.

## The shape of the thing

A decision model is a fast, cheap, well-calibrated answer to a typed question. The
thing that decides what to do about the answer is a policy, and it changes for
reasons that have nothing to do with the model: a cost target moves, a review
team's capacity changes, an incident makes one kind of error more expensive.

Coupling those two on one line of code is what makes the cheap part expensive. It
is a small change to keep them apart, and it turns the boundary from a number
nobody dares touch into a parameter you can sweep in a second.

---

`metajev` is a small library that does this, with no dependencies, an offline
provider so a fresh clone can run the whole path without a key, and an interactive
report that lets you drag the boundary across a recorded history in a browser.
