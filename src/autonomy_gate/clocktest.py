"""Measure what the clock is worth: the same decisions, asked with and without ``decidedAt``.

``autonomy-gate clock-test`` runs the agent twice over the same human decisions: once with
``decidedAt`` in every request, once without it, and nothing else different. It compares
the model's own answers (no threshold, so nothing tuned on one arm favors it), pairs the
two arms on the person's approvals with an exact McNemar test, and splits recall by
whether the request uses relative time words, which is where a missing clock should show.

Both arms get the same instructions, so instructions that say every request carries
``decidedAt`` would tell the bare arm about a field it doesn't have and could push it toward
holds; the adapters' default instructions only say what to do if it is there. Everything
else in the request stays, including any other timestamp. The synthetic requests carry
``postedAt``, usually minutes to a few hours before the decision, so on them the bare arm
means no ``decidedAt``, not no time at all.

It records what a reader needs to trust or repeat the number: the model and effort the
adapter reports, the date, how many decisions, a hash of the label file, the candidate's
fingerprint, and the tokens each arm used, priced if you pass prices. It claims nothing on
its own: run it on a real model and commit the report.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .lint import RELATIVE_TIME
from .lock import agent_fingerprint
from .metrics import Confusion, confusion_at, mcnemar_exact, p95
from .policy import Agent, Call
from .records import Record, require_human_labels
from .replay import approve_score, run_agent
from .usage import usage_delta


@dataclass(frozen=True)
class Arm:
    clock: bool
    confusion: Confusion
    relative: Confusion
    other: Confusion
    errors: int
    p95_latency_s: float | None
    usage: dict[str, int] | None


@dataclass(frozen=True)
class ClockReport:
    meta: dict[str, Any]
    with_clock: Arm
    without_clock: Arm
    only_with: int
    only_without: int
    p_value: float
    relative_n: int
    other_n: int
    cost_usd: dict[str, float] | None = None
    unpriced: dict[str, int] = field(default_factory=dict)


def uses_relative_time(record: Record) -> bool:
    return bool(RELATIVE_TIME.search(record.text))


def _snapshot(agent: Agent) -> dict[str, int] | None:
    meter = getattr(agent, "usage", None)
    return meter.snapshot() if meter is not None and hasattr(meter, "snapshot") else None


def _arm(
    clock: bool,
    records: Sequence[Record],
    calls: Sequence[Call],
    usage: dict[str, int] | None,
) -> Arm:
    items = [(approve_score(c), bool(r.label)) for r, c in zip(records, calls, strict=True)]
    relative = [i for i, r in zip(items, records, strict=True) if uses_relative_time(r)]
    other = [i for i, r in zip(items, records, strict=True) if not uses_relative_time(r)]
    return Arm(
        clock=clock,
        confusion=confusion_at(items, 0.0),
        relative=confusion_at(relative, 0.0),
        other=confusion_at(other, 0.0),
        errors=sum(1 for c in calls if c.failed),
        p95_latency_s=p95([c.latency_s for c in calls]),
        usage=usage,
    )


def _cost(
    arms: Mapping[str, Arm], usd_per_mtok: tuple[float, float] | None
) -> tuple[dict[str, float] | None, dict[str, int]]:
    if usd_per_mtok is None or any(a.usage is None for a in arms.values()):
        return None, {}
    usd_in, usd_out = usd_per_mtok
    cost = {
        name: (a.usage["input_tokens"] * usd_in + a.usage["output_tokens"] * usd_out) / 1e6
        for name, a in arms.items()
        if a.usage is not None
    }
    cost["total"] = sum(cost.values())
    unpriced = {
        k: sum(a.usage[k] for a in arms.values() if a.usage is not None)
        for k in ("cache_read_tokens", "cache_write_tokens")
    }
    return cost, {k: v for k, v in unpriced.items() if v}


def measure_clock(
    agent: Agent,
    records: Sequence[Record],
    *,
    workers: int = 1,
    timeout_s: float | None = None,
    usd_per_mtok: tuple[float, float] | None = None,
    candidate: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> ClockReport:
    """Ask every record twice, with the clock and without it, and compare the answers.

    ``candidate`` is the ``--candidate`` spec the agent was loaded from. With it, or with an
    agent that has its own ``fingerprint``, the report carries the same candidate fingerprint
    ``replay --lock`` would record; with neither, the fingerprint is None.
    """
    records = require_human_labels(records, where="clock-test")
    before = _snapshot(agent)
    clocked = run_agent(agent, records, clock=True, workers=workers, timeout_s=timeout_s)
    middle = _snapshot(agent)
    bare = run_agent(agent, records, clock=False, workers=workers, timeout_s=timeout_s)
    after = _snapshot(agent)
    with_usage = usage_delta(before, middle) if before is not None and middle else None
    without_usage = usage_delta(middle, after) if middle is not None and after else None
    arms = {
        "with_clock": _arm(True, records, clocked, with_usage),
        "without_clock": _arm(False, records, bare, without_usage),
    }
    caught = [
        (approve_score(a) is not None, approve_score(b) is not None)
        for r, a, b in zip(records, clocked, bare, strict=True)
        if r.label
    ]
    only_with = sum(1 for w, wo in caught if w and not wo)
    only_without = sum(1 for w, wo in caught if wo and not w)
    describe = getattr(agent, "describe", None)
    relative_n = sum(1 for r in records if uses_relative_time(r))
    cost, unpriced = _cost(arms, usd_per_mtok)
    return ClockReport(
        meta={
            "autonomy_gate": __version__,
            "ran_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "agent": describe() if callable(describe) else None,
            "fingerprint": (
                agent_fingerprint(agent, candidate or "", True)
                if candidate or getattr(agent, "fingerprint", None)
                else None
            ),
            "n": len(records),
            "positives": sum(1 for r in records if r.label),
            "workers": workers,
            "timeout_s": timeout_s,
            "usd_per_mtok": list(usd_per_mtok) if usd_per_mtok else None,
            **dict(meta or {}),
        },
        with_clock=arms["with_clock"],
        without_clock=arms["without_clock"],
        only_with=only_with,
        only_without=only_without,
        p_value=mcnemar_exact(only_with, only_without),
        relative_n=relative_n,
        other_n=len(records) - relative_n,
        cost_usd=cost,
        unpriced=unpriced,
    )


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rate(c: Confusion, confidence: float) -> str:
    p = "n/a" if c.precision is None else f"{c.precision:.3f}"
    r = "n/a" if c.recall is None else f"{c.recall:.3f}"
    return (
        f"approves {c.flagged}, right {c.tp}: precision {p} {c.precision_interval(confidence)}; "
        f"catches {c.tp} of {c.positives}: recall {r} {c.recall_interval(confidence)}"
    )


def _recall(c: Confusion) -> str:
    return "n/a" if c.recall is None else f"{c.tp}/{c.positives} = {c.recall:.3f}"


def _tokens(usage: Mapping[str, int] | None) -> str:
    if usage is None:
        return "no usage reported by this candidate"
    return (
        f"{usage['calls']} calls, {usage['input_tokens']} in, {usage['output_tokens']} out "
        f"({usage['reasoning_tokens']} reasoning)"
    )


def format_clock(report: ClockReport, confidence: float = 0.95) -> str:
    m = report.meta
    agent = m.get("agent") or {}
    w, wo = report.with_clock, report.without_clock
    lines = [
        f"Clock test: {m['n']} human decisions ({m['positives']} approvals), each asked twice, "
        "with decidedAt and without it. The model's own answers, no threshold.",
        f"Candidate: {m.get('candidate', 'n/a')} (model {agent.get('model', 'n/a')}, effort "
        f"{agent.get('effort', 'n/a')}). Ran {m['ran_at']}.",
        "",
        f"  with clock     {_rate(w.confusion, confidence)}",
        f"  without clock  {_rate(wo.confusion, confidence)}",
        "",
        f"Person approvals caught only with the clock: {report.only_with}; only without it: "
        f"{report.only_without}; exact McNemar p = {report.p_value:.3g}",
        f"Requests with relative time words ({report.relative_n}): recall {_recall(w.relative)} "
        f"with the clock, {_recall(wo.relative)} without.",
        f"Other requests ({report.other_n}): recall {_recall(w.other)} with the clock, "
        f"{_recall(wo.other)} without.",
        f"Model errors, counted as holds: {w.errors} with the clock, {wo.errors} without.",
        f"Tokens: with the clock {_tokens(w.usage)}; without {_tokens(wo.usage)}.",
    ]
    if report.cost_usd is not None:
        c = report.cost_usd
        lines.append(
            f"Cost: ${c['with_clock']:.4f} with the clock, ${c['without_clock']:.4f} without, "
            f"${c['total']:.4f} in all (input and output tokens at the prices given)."
        )
        if report.unpriced:
            lines.append(f"Not priced: {report.unpriced}.")
    else:
        lines.append(
            "Cost: not computed; pass --usd-per-mtok-in and --usd-per-mtok-out with the "
            "model's prices."
        )
    return "\n".join(lines)


def clock_dict(report: ClockReport, confidence: float = 0.95) -> dict[str, Any]:
    def arm(a: Arm) -> dict[str, Any]:
        return {
            "as_answered": a.confusion.to_dict(confidence),
            "relative_time": a.relative.to_dict(confidence),
            "other": a.other.to_dict(confidence),
            "errors": a.errors,
            "p95_latency_s": a.p95_latency_s,
            "usage": a.usage,
        }

    return {
        "format": "autonomy-gate.clock-test/1",
        "meta": report.meta,
        "arms": {"with_clock": arm(report.with_clock), "without_clock": arm(report.without_clock)},
        "paired": {
            "only_with_clock": report.only_with,
            "only_without_clock": report.only_without,
            "mcnemar_p": report.p_value,
        },
        "relative_time_requests": report.relative_n,
        "other_requests": report.other_n,
        "cost_usd": report.cost_usd,
        "unpriced_tokens": report.unpriced,
    }
