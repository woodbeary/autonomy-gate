"""The ``autonomy-gate`` command line.

Exit codes: 0 when the step passes (GREEN, clean lint, re-test passed), 1 when it doesn't
(RED, lint errors, re-test failed, no threshold clears the bars), 2 for bad input.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .actuator import Guard, NotEarned
from .baseline import LogisticBaseline
from .clocktest import clock_dict, file_sha256, format_clock, measure_clock
from .golive import check, format_check
from .killswitch import FileKillSwitch
from .ledger import Ledger
from .lint import lint_file
from .lock import DataReuseError, GateLock, LockError, agent_fingerprint
from .records import (
    DataError,
    disjoint,
    load_assisted,
    load_labels,
    load_records,
    parse_ts,
    split_provenance,
    write_jsonl,
)
from .replay import (
    assisted_report,
    format_assisted,
    format_report,
    replay,
    report_dict,
    run_agent,
    stratified_sample,
    time_split,
)
from .shadow import Shadow, load_shadow
from .synthetic import SEED, FileOutbox, write_examples
from .threshold import Bars


class UsageError(Exception):
    pass


def load_object(spec: str) -> Any:
    """Resolve ``module:attr`` or ``path/to/file.py:attr``. A class is instantiated."""
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise UsageError(f"expected module:attr or file.py:attr, got {spec!r}")
    if target.endswith(".py"):
        path = Path(target)
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise UsageError(f"can't load {target}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise UsageError(f"{target} has no {attr!r}") from exc
    return obj() if inspect.isclass(obj) else obj


def _bars(args: argparse.Namespace, lock: GateLock | None = None) -> Bars:
    if getattr(args, "bars", None):
        return Bars.load(args.bars)
    return lock.bars if lock is not None else Bars()


def cmd_synth(args: argparse.Namespace) -> int:
    for path in write_examples(args.directory, seed=args.seed):
        print(path)
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    ok = True
    for path in args.files:
        result = lint_file(path, labels=not args.export)
        status = "ok" if result.ok else "FAIL"
        print(f"{status}  {path}: {result.records} records")
        for e in result.errors:
            print(f"  error: {e}")
        for n in result.notes:
            print(f"  note: {n}")
        ok = ok and result.ok
    return 0 if ok else 1


def cmd_labels(args: argparse.Namespace) -> int:
    split = split_provenance(load_records(args.export))
    write_jsonl(args.output, (r.to_dict() for r in split.human))
    assisted = f"{len(split.assisted)} made by a person who saw the agent's suggestion"
    if args.assisted:
        write_jsonl(args.assisted, (r.to_dict() for r in split.assisted))
        kept = f"kept {assisted} apart in {args.assisted}"
    else:
        kept = f"dropped {assisted} (keep them with --assisted)"
    print(
        f"wrote {len(split.human)} human decisions to {args.output}; {kept}; dropped "
        f"{len(split.agent)} made by the agent and {len(split.undecided)} undecided"
    )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    records = load_labels(args.labels)
    assisted = load_assisted(args.assisted) if args.assisted else []
    disjoint(records, assisted)
    bars = _bars(args)
    agent = load_object(args.candidate)
    clock = not args.strip_clock
    baseline = None
    sample = records
    if args.baseline:
        train, sample = time_split(records, args.train_fraction)
        baseline = LogisticBaseline().fit(train)
        print(
            f"Baseline fitted on the oldest {len(train)} decisions; "
            f"replaying the newer {len(sample)}."
        )
    if args.sample:
        sample = stratified_sample(sample, args.sample, seed=args.seed)
    calls = run_agent(agent, sample, clock=clock, workers=args.workers, timeout_s=args.timeout)
    report = replay(sample, calls, bars, clock=clock, baseline=baseline)
    helped = None
    if assisted:
        helped_calls = run_agent(
            agent, assisted, clock=clock, workers=args.workers, timeout_s=args.timeout
        )
        helped = assisted_report(assisted, helped_calls, report.threshold)
    if args.json:
        out = report_dict(report)
        if helped is not None:
            out["assisted"] = {
                **helped.confusion.to_dict(bars.confidence),
                "n": helped.n,
                "agree": helped.agree,
            }
        print(json.dumps(out, indent=2))
    else:
        print(format_report(report))
        if helped is not None:
            print("\n" + format_assisted(helped, bars.confidence))
    if args.lock:
        if not clock:
            raise UsageError("refusing to lock a replay with the clock stripped")
        previous = GateLock.load(args.previous_lock) if args.previous_lock else None
        lock = GateLock.from_replay(
            [*records, *assisted],
            report,
            fingerprint=agent_fingerprint(agent, args.candidate, clock),
            previous=previous,
        )
        lock.save(args.lock)
        print(
            f"\nLocked threshold {lock.threshold:.4g} in {args.lock}; the gate has now seen "
            f"everything through {lock.seen_through}."
        )
    return 0 if report.candidate.meets_bars else 1


def cmd_clock_test(args: argparse.Namespace) -> int:
    prices = (args.usd_per_mtok_in, args.usd_per_mtok_out)
    if (prices[0] is None) != (prices[1] is None):
        raise UsageError("pass both --usd-per-mtok-in and --usd-per-mtok-out, or neither")
    records = load_labels(args.labels)
    if args.sample:
        records = stratified_sample(records, args.sample, seed=args.seed)
    agent = load_object(args.candidate)
    report = measure_clock(
        agent,
        records,
        workers=args.workers,
        timeout_s=args.timeout,
        usd_per_mtok=None if prices[0] is None else (prices[0], prices[1]),
        candidate=args.candidate,
        meta={
            "candidate": args.candidate,
            "labels": Path(args.labels).name,
            "labels_sha256": file_sha256(args.labels),
            "sample": args.sample,
            "seed": args.seed if args.sample else None,
        },
    )
    out = clock_dict(report)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(out, indent=2) if args.json else format_clock(report))
    if args.out and not args.json:
        print(f"\nWrote {args.out}.")
    return 0


def cmd_retest(args: argparse.Namespace) -> int:
    lock = GateLock.load(args.lock)
    records = load_labels(args.heldout)
    agent = load_object(args.candidate)
    fp = agent_fingerprint(agent, args.candidate, True)
    if fp != lock.fingerprint:
        raise UsageError("this candidate's fingerprint differs from the locked one; re-tune first")
    lock.check_fresh(records, allow_older=args.allow_older)
    calls = run_agent(agent, records, clock=True, workers=args.workers, timeout_s=args.timeout)
    result = lock.run_retest(records, calls, allow_older=args.allow_older)
    lock.save(args.lock)
    if args.json:
        print(
            json.dumps({k: v for k, v in result.items() if k not in ("ids", "content")}, indent=2)
        )
    else:
        verdict = "PASSED" if result["passed"] else "FAILED: " + "; ".join(result["failures"])
        p, r = result["precision"], result["recall"]
        print(
            f"Held-out re-test at the frozen threshold {lock.threshold:.4g}, "
            f"{result['n']} new decisions:"
        )
        print(
            f"  approves {result['tp'] + result['fp']}, right {result['tp']}: precision "
            f"{'n/a' if p is None else f'{p:.3f}'} [{result['precision_ci'][0]:.3f}, "
            f"{result['precision_ci'][1]:.3f}]"
        )
        print(
            f"  catches {result['tp']} of {result['positives']}: recall "
            f"{'n/a' if r is None else f'{r:.3f}'} [{result['recall_ci'][0]:.3f}, "
            f"{result['recall_ci'][1]:.3f}]"
        )
        print(f"  {verdict}. Recorded in {args.lock}; this lock takes no other held-out set.")
    return 0 if result["passed"] else 1


def cmd_shadow(args: argparse.Namespace) -> int:
    requests = load_records(args.requests)
    if args.until:
        until = parse_ts(args.until, "--until")
        requests = [r for r in requests if r.decided_at < until]
    agent = load_object(args.candidate)
    fp = agent_fingerprint(agent, args.candidate, True)
    threshold = None
    if args.lock:
        lock = GateLock.load(args.lock)
        threshold = lock.threshold
    shadow = Shadow(agent, args.log, fingerprint=fp, threshold=threshold, timeout_s=args.timeout)
    entries = shadow.run(requests)
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["action"]] = counts.get(e["action"], 0) + 1
    shown = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing new"
    print(f"Shadowed {len(entries)} requests into {args.log} ({shown}). Nothing was sent.")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    lock = GateLock.load(args.lock) if args.lock else None
    bars = _bars(args, lock)
    assisted = load_assisted(args.assisted) if args.assisted else []
    report = check(load_shadow(args.shadow), load_labels(args.labels), bars, lock, assisted)
    if args.json:
        rows = [r.__dict__ for r in report.rows]
        print(json.dumps({**report.summary(), "rows": rows}, indent=2))
    else:
        print(format_check(report))
    if lock is not None:
        lock.record_check(report.summary())
        lock.save(args.lock)
    return 0 if report.passed else 1


def cmd_switch(args: argparse.Namespace) -> int:
    switch = FileKillSwitch(args.file)
    if args.state != "status":
        switch.set(args.state == "on")
    state = switch.state()
    print(f"{'on (agent may act)' if state.running else 'off'}: {state.detail}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    lock = GateLock.load(args.lock)
    agent = load_object(args.candidate)
    fp = agent_fingerprint(agent, args.candidate, True)
    sender = None
    if args.live:
        if args.outbox:
            sender = FileOutbox(args.outbox, crash=args.demo_crash)
        elif args.sender:
            sender = load_object(args.sender)
        else:
            raise UsageError("--live needs --outbox or --sender")
    elif args.demo_crash:
        raise UsageError("--demo-crash needs --live")
    requests = load_records(args.requests)
    with Ledger(args.ledger) as ledger:
        guard = Guard(
            agent,
            ledger,
            FileKillSwitch(args.switch),
            lock,
            fingerprint=fp,
            sender=sender,
            max_sends_per_cycle=args.cap,
            timeout_s=args.timeout,
        )
        pending = [r for r in requests if not ledger.has(r.id)]
        batch = pending[: args.batch] if args.batch else pending
        report = guard.cycle(batch)
        mode = "live" if guard.live else "live off (approvals are held)"
        print(
            f"Cycle: {mode}; kill switch {'on' if report.switch_on else 'off'}; "
            f"{len(batch)} new requests."
        )
        for item_id, state, why in report.outcomes:
            print(f"  {item_id}  {state:<30}  {why}")
        print("Ledger: " + ", ".join(f"{k} {v}" for k, v in sorted(ledger.counts().items())))
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        rows = [ledger.get(args.item)] if args.item else ledger.rows()
        for row in rows:
            if row is None:
                print(f"{args.item} is not in the ledger")
                return 1
            print(
                f"{row.item_id}  {row.state:<30}  {row.reason:<16}  "
                f"{row.request_id or '-':<27}  {row.receipt or '-'}"
            )
            if args.item:
                for at, state, detail in ledger.events(row.item_id):
                    print(f"    {at}  {state}  {detail or ''}")
        if not args.item:
            print("Totals: " + ", ".join(f"{k} {v}" for k, v in sorted(ledger.counts().items())))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autonomy-gate",
        description="Decide when an LLM agent has earned the right to act.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def agent_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--candidate",
            required=True,
            help="module:attr or file.py:attr; a callable taking a request dict",
        )
        sp.add_argument("--timeout", type=float, default=None, help="seconds per model call")

    s = sub.add_parser("synth", help="write the synthetic example files")
    s.add_argument("directory")
    s.add_argument("--seed", type=int, default=SEED)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("lint", help="check the clock, leakage and provenance of decision files")
    s.add_argument("files", nargs="+")
    s.add_argument(
        "--export",
        action="store_true",
        help="a raw export: allow decisions the agent made (labels must not)",
    )
    s.set_defaults(func=cmd_lint)

    s = sub.add_parser("labels", help="keep only human decisions from an export")
    s.add_argument("export")
    s.add_argument("-o", "--output", required=True)
    s.add_argument(
        "--assisted",
        help="also write decisions made by a person who saw the agent's suggestion here",
    )
    s.set_defaults(func=cmd_labels)

    s = sub.add_parser("replay", help="replay labeled decisions; choose and lock a threshold")
    s.add_argument("labels")
    agent_opts(s)
    s.add_argument("--bars")
    s.add_argument(
        "--baseline",
        action="store_true",
        help="fit a logistic baseline on older decisions and compare on newer ones",
    )
    s.add_argument("--train-fraction", type=float, default=0.5)
    s.add_argument("--sample", type=int, help="replay a stratified sample of this size")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--strip-clock",
        action="store_true",
        help="leave decidedAt out of every request, to see what the model does without it",
    )
    s.add_argument("--lock", help="write the chosen threshold to this lock file")
    s.add_argument("--previous-lock", help="carry everything an earlier lock saw into the new one")
    s.add_argument(
        "--assisted",
        help="decisions by a person who saw the agent's suggestion; reported apart",
    )
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser(
        "clock-test", help="ask the same decisions with and without decidedAt; compare"
    )
    s.add_argument("labels")
    agent_opts(s)
    s.add_argument("--sample", type=int, help="use a stratified sample of this size")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--usd-per-mtok-in", type=float, help="input price, USD per million tokens")
    s.add_argument("--usd-per-mtok-out", type=float, help="output price, USD per million tokens")
    s.add_argument("--out", help="write the JSON report here")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_clock_test)

    s = sub.add_parser("retest", help="score newer decisions once at the frozen threshold")
    s.add_argument("heldout")
    agent_opts(s)
    s.add_argument("--lock", required=True)
    s.add_argument(
        "--allow-older",
        action="store_true",
        help="accept decisions older than the newest the gate has seen",
    )
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_retest)

    s = sub.add_parser("shadow", help="run the agent on live requests and log; never sends")
    s.add_argument("requests")
    agent_opts(s)
    s.add_argument("--log", required=True)
    s.add_argument("--lock")
    s.add_argument("--until", help="only requests decided before this ISO time")
    s.set_defaults(func=cmd_shadow)

    s = sub.add_parser("check", help="RED or GREEN against the go-live bars")
    s.add_argument("--shadow", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--bars")
    s.add_argument("--lock", help="also check the re-test and the candidate; record the result")
    s.add_argument(
        "--assisted",
        help="decisions by a person who saw the agent's suggestion; reported, never a bar",
    )
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("switch", help="turn the file kill switch on or off")
    s.add_argument("file")
    s.add_argument("state", choices=("on", "off", "status"))
    s.set_defaults(func=cmd_switch)

    s = sub.add_parser("run", help="one guarded cycle over new requests")
    s.add_argument("requests")
    agent_opts(s)
    s.add_argument("--lock", required=True)
    s.add_argument("--ledger", required=True)
    s.add_argument("--switch", required=True, help="kill switch file; must say `run` to act")
    s.add_argument("--live", action="store_true", help="actually send approvals (off by default)")
    s.add_argument("--outbox", help="the built-in file sender, for the demo")
    s.add_argument("--sender", help="module:attr of an object with send() and lookup()")
    s.add_argument("--cap", type=int, help="most sends per cycle; the rest wait as would_approve")
    s.add_argument("--batch", type=int, help="take at most this many new requests")
    s.add_argument(
        "--demo-crash",
        choices=("before_write", "after_write"),
        help="demo only: end the process inside the outbox's send",
    )
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("ledger", help="show the ledger")
    s.add_argument("ledger")
    s.add_argument("--item")
    s.set_defaults(func=cmd_ledger)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DataError, LockError, UsageError, NotEarned, ValueError, OSError) as exc:
        label = "refused" if isinstance(exc, DataReuseError | NotEarned) else "error"
        print(f"autonomy-gate: {label}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
