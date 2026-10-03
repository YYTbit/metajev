"""The ``metajev`` command line.

Two things are worth a shell. Asking one question and seeing the answer, which is
how a question is debugged before it goes into a pipeline. And replaying a stored
history under a new policy, which is the operation this package exists to make
cheap and is genuinely unpleasant to do from inside a program.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .budget import Budget
from .calibrate import observations_from_store, summarise, threshold_table
from .errors import MetajevError
from .ledger import Ledger
from .policy import Band, Policy, Rule, label_policy, three_band_policy
from .providers import build_provider
from .replay import replay, sweep
from .store import DecisionStore
from .types import Action, Question, QuestionType

PRESETS = {
    "strict": lambda: three_band_policy("strict", accept=0.92, review=0.70),
    "balanced": lambda: three_band_policy("balanced", accept=0.85, review=0.55),
    "lenient": lambda: three_band_policy("lenient", accept=0.70, review=0.35),
}


def _load_policy(args: argparse.Namespace) -> Policy:
    """Resolve a policy from the command line, a file, or a preset."""
    if getattr(args, "policy_file", None):
        with open(args.policy_file, "r", encoding="utf-8") as handle:
            return Policy.from_json(handle.read())
    if getattr(args, "label", None):
        return label_policy(
            args.label,
            accept=getattr(args, "accept", 0.7),
            review=getattr(args, "review", 0.4),
        )
    preset = getattr(args, "preset", None) or "balanced"
    factory = PRESETS.get(preset)
    if factory is None:
        raise MetajevError(f"unknown preset {preset!r}; known presets are {sorted(PRESETS)}")
    return factory()


def _build_question(args: argparse.Namespace) -> Question:
    kind = QuestionType(args.type)
    if kind is QuestionType.CHOICE:
        options = [part.strip() for part in args.options.split(",") if part.strip()]
        return Question.choice(args.question, options, tags=_tags(args))
    if kind is QuestionType.SCORE:
        levels = [part.strip() for part in args.levels.split(",") if part.strip()]
        return Question.score(args.question, levels, tags=_tags(args))
    return Question.noul(args.question, yes=args.yes, tags=_tags(args))


def _tags(args: argparse.Namespace) -> tuple[str, ...]:
    raw = getattr(args, "tag", None) or []
    return tuple(raw)


# -- commands --------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    question = _build_question(args)
    provider = build_provider(args.provider)
    with DecisionStore(args.store) as store:
        policy = _load_policy(args)
        from .client import Client

        client = Client(provider=provider, store=store, policy=policy, budget=Budget())
        answer = client.decide(args.state, question)
        payload = answer.to_dict()
        payload["distribution"] = [list(pair) for pair in answer.decision.distribution]
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"action      {answer.action.value}")
            print(f"answer      {answer.decision.answer}")
            print(f"confidence  {answer.decision.confidence:.4f}")
            print(f"provider    {answer.decision.provider}:{answer.decision.model}")
            print(f"cached      {answer.cached}")
            print(f"reason      {answer.resolution.reason}")
            print("distribution")
            for label, probability in answer.decision.distribution:
                bar = "#" * int(round(probability * 40))
                print(f"  {label:<20}{probability:>7.3f}  {bar}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    with DecisionStore(args.store) as store:
        store.record_outcome(args.key, args.label, note=args.note or "")
        print(f"recorded {args.label} for {args.key[:16]}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    with DecisionStore(args.store) as store:
        policy = _load_policy(args)
        baseline = None
        if args.baseline_preset:
            baseline = PRESETS[args.baseline_preset]()
        elif args.baseline_file:
            with open(args.baseline_file, "r", encoding="utf-8") as handle:
                baseline = Policy.from_json(handle.read())
        report = replay(store, policy, baseline=baseline, examples=args.examples)
        print(report.render())
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    values = [float(part) for part in args.values.split(",") if part.strip()]
    if not values:
        raise MetajevError("--values needs at least one number")

    def make_policy(value: float) -> Policy:
        if args.on == "label":
            return label_policy(args.label, accept=value, review=min(value, args.review))
        return three_band_policy(f"accept={value:g}", accept=value, review=min(value, args.review))

    with DecisionStore(args.store) as store:
        baseline = PRESETS[args.baseline_preset]() if args.baseline_preset else None
        rows = sweep(store, make_policy, values, baseline=baseline)
        header = (
            f"{'accept':>8}{'changed':>9}{'accepted':>10}{'precision':>11}"
            f"{'err diverted':>14}{'err admitted':>13}{'right lost':>12}"
        )
        print(header)
        print("-" * len(header))
        for value, report in rows:
            accepted = report.bucket(Action.ACCEPT)
            print(
                f"{value:>8.2f}"
                f"{(report.flips if report.flips is not None else 0):>9}"
                f"{(accepted.count if accepted else 0):>10}"
                f"{(report.accept_precision if report.accept_precision is not None else 0):>11.3f}"
                f"{(report.errors_diverted if report.errors_diverted is not None else 0):>14}"
                f"{(report.errors_admitted if report.errors_admitted is not None else 0):>13}"
                f"{(report.correct_diverted if report.correct_diverted is not None else 0):>12}"
            )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    with DecisionStore(args.store) as store:
        observations, unjudged = observations_from_store(store, on=args.on)
        summary = summarise(observations, name=args.on, bins=args.bins)
        print(summary.render())
        if unjudged:
            print("")
            print(f"{unjudged} stored decisions have no recorded outcome and were skipped")
    return 0


def cmd_threshold(args: argparse.Namespace) -> int:
    steps = int(args.steps)
    if steps < 1:
        raise MetajevError("--steps must be at least 1")
    thresholds = [index / steps for index in range(steps + 1)]
    with DecisionStore(args.store) as store:
        observations, _ = observations_from_store(store, on=args.on)
        if not observations:
            print("no judged decisions in this store")
            return 0
        rows = threshold_table(observations, thresholds)
        header = f"{'threshold':>10}{'admitted':>10}{'coverage':>10}{'precision':>11}{'errors':>8}{'lost':>7}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['threshold']:>10.2f}{int(row['admitted']):>10}{row['coverage']:>10.2f}"
                f"{row['precision']:>11.3f}{int(row['errors_admitted']):>8}"
                f"{int(row['right_turned_away']):>7}"
            )
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    ledger = Ledger(args.path)
    if args.ledger_command == "verify":
        try:
            count = ledger.verify()
        except MetajevError as exc:
            print(f"FAILED  {exc}")
            return 1
        print(f"ok      {count} entries, head {ledger.head()[:16]}")
        return 0
    if args.ledger_command == "head":
        print(ledger.head())
        return 0
    if args.ledger_command == "count":
        print(ledger.count())
        return 0
    raise MetajevError(f"unknown ledger command {args.ledger_command!r}")


def cmd_providers(args: argparse.Namespace) -> int:
    from .providers import PROVIDER_TYPES

    print(f"{'name':<16}{'class':<26}{'default endpoint'}")
    for name, cls in sorted(PROVIDER_TYPES.items()):
        endpoint = getattr(cls, "DEFAULT_ENDPOINT", "") or "offline, no endpoint"
        print(f"{name:<16}{cls.__name__:<26}{endpoint}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print("providers")
    for spec in args.provider or ["mock"]:
        try:
            provider = build_provider(spec)
            ok, note = provider.health()
            mark = "ok  " if ok else "warn"
            print(f"  {mark} {spec:<24}{note}")
        except Exception as exc:  # noqa: BLE001 - doctor reports rather than raises
            print(f"  fail {spec:<24}{exc}")
    if args.store:
        with DecisionStore(args.store) as store:
            stats = store.stats()
            print("")
            print("store")
            print(f"  path                {args.store}")
            print(f"  decisions           {stats.decisions}")
            print(f"  outcomes            {stats.outcomes}")
            print(f"  spent               {stats.total_cost_usd:.6f} USD")
            print(f"  mean confidence     {stats.mean_confidence:.4f}")
            for name, count in stats.by_provider.items():
                print(f"    {name:<18}{count}")
    if args.ledger_path:
        ledger = Ledger(args.ledger_path)
        print("")
        print("ledger")
        try:
            count = ledger.verify()
            print(f"  ok                  {count} entries, head {ledger.head()[:16]}")
        except MetajevError as exc:
            print(f"  failed              {exc}")
            return 1
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    with DecisionStore(args.store) as store:
        written = store.export_jsonl(args.out)
        print(f"wrote {written} records to {args.out}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    with DecisionStore(args.store) as store:
        decisions, outcomes = store.import_jsonl(args.path)
        print(f"read {decisions} new decisions and {outcomes} outcomes from {args.path}")
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    policy = _load_policy(args)
    if args.json:
        print(policy.to_json())
    else:
        print(policy.describe())
    return 0


# -- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metajev",
        description="Record typed decisions, then change the policy without re-running the model.",
    )
    parser.add_argument("--version", action="version", version="metajev 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask one typed question and show the decision")
    ask.add_argument("state", help="the text to judge")
    ask.add_argument("--question", required=True, help="what to ask about the state")
    ask.add_argument("--type", default="noul", choices=[t.value for t in QuestionType])
    ask.add_argument("--options", default="", help="comma separated, for a choice question")
    ask.add_argument("--levels", default="", help="comma separated low to high, for a score question")
    ask.add_argument("--yes", default=None, help="sharpen what counts as yes")
    ask.add_argument("--tag", action="append", help="label the question, repeatable")
    ask.add_argument("--provider", default="mock", help="provider spec, such as typesafe or sglang:my-model")
    ask.add_argument("--store", default=":memory:", help="decision store path")
    ask.add_argument("--preset", default="balanced", choices=sorted(PRESETS))
    ask.add_argument("--policy-file", default=None)
    ask.add_argument("--label", default=None, help="resolve on the probability of this label")
    ask.add_argument("--accept", type=float, default=0.7)
    ask.add_argument("--review", type=float, default=0.4)
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    record = sub.add_parser("record", help="record the truth for a stored decision")
    record.add_argument("key", help="the decision key")
    record.add_argument("label", help="the correct label")
    record.add_argument("--store", required=True)
    record.add_argument("--note", default="")
    record.set_defaults(func=cmd_record)

    rep = sub.add_parser("replay", help="re-resolve stored decisions under a new policy")
    rep.add_argument("--store", required=True)
    rep.add_argument("--policy-file", default=None)
    rep.add_argument("--label", default=None)
    rep.add_argument("--preset", default="balanced", choices=sorted(PRESETS))
    rep.add_argument("--accept", type=float, default=0.7)
    rep.add_argument("--review", type=float, default=0.4)
    rep.add_argument("--baseline-preset", default=None, choices=sorted(PRESETS))
    rep.add_argument("--baseline-file", default=None)
    rep.add_argument("--examples", type=int, default=5)
    rep.set_defaults(func=cmd_replay)

    sw = sub.add_parser("sweep", help="replay one boundary across a range of values")
    sw.add_argument("--store", required=True)
    sw.add_argument("--values", required=True, help="comma separated, such as 0.6,0.7,0.8,0.9")
    sw.add_argument("--on", default="confidence", choices=["confidence", "label"])
    sw.add_argument("--label", default="yes")
    sw.add_argument("--review", type=float, default=0.4)
    sw.add_argument("--baseline-preset", default=None, choices=sorted(PRESETS))
    sw.set_defaults(func=cmd_sweep)

    cal = sub.add_parser("calibrate", help="measure how well probabilities match outcomes")
    cal.add_argument("--store", required=True)
    cal.add_argument("--on", default="confidence")
    cal.add_argument("--bins", type=int, default=10)
    cal.set_defaults(func=cmd_calibrate)

    thr = sub.add_parser("threshold", help="show what each threshold would admit")
    thr.add_argument("--store", required=True)
    thr.add_argument("--on", default="confidence")
    thr.add_argument("--steps", type=int, default=20)
    thr.set_defaults(func=cmd_threshold)

    led = sub.add_parser("ledger", help="inspect and verify a ledger")
    led.add_argument("ledger_command", choices=["verify", "head", "count"])
    led.add_argument("path")
    led.set_defaults(func=cmd_ledger)

    prov = sub.add_parser("providers", help="list the available providers")
    prov.set_defaults(func=cmd_providers)

    doc = sub.add_parser("doctor", help="check providers, store, and ledger")
    doc.add_argument("--provider", action="append", default=None)
    doc.add_argument("--store", default=None)
    doc.add_argument("--ledger-path", default=None)
    doc.set_defaults(func=cmd_doctor)

    exp = sub.add_parser("export", help="write the store to JSONL")
    exp.add_argument("--store", required=True)
    exp.add_argument("--out", required=True)
    exp.set_defaults(func=cmd_export)

    imp = sub.add_parser("import", help="read a JSONL file into the store")
    imp.add_argument("--store", required=True)
    imp.add_argument("path")
    imp.set_defaults(func=cmd_import)

    pol = sub.add_parser("policy", help="print a policy and its fingerprint")
    pol.add_argument("--policy-file", default=None)
    pol.add_argument("--label", default=None)
    pol.add_argument("--preset", default="balanced", choices=sorted(PRESETS))
    pol.add_argument("--accept", type=float, default=0.7)
    pol.add_argument("--review", type=float, default=0.4)
    pol.add_argument("--json", action="store_true")
    pol.set_defaults(func=cmd_policy)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MetajevError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
