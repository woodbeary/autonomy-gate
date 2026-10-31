"""Replay labeled decisions through the agent and a baseline, and say what would have happened.

The agent sees each request as it was at ``decidedAt``, clock included, and never the
decision. Its approvals are thresholded on confidence; the threshold is the lowest one
whose precision clears the bars on these labels. Everything is reported as whole counts
first ("75 of 87") and rates second, because a rate without its denominator can't be
checked.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .metrics import Confusion, confusion_at, mcnemar_exact, p95
from .policy import APPROVE, HOLD, SKIP, Agent, Call, Verdict, apply_threshold, decide
from .records import Record, require_human_labels
from .threshold import Bars, OperatingPoint, choose_threshold, threshold_for_precision


def run_agent(
    agent: Agent,
    records: Sequence[Record],
    *,
    clock: bool = True,
    workers: int = 1,
    timeout_s: float | None = None,
) -> list[Call]:
    """Call the agent once per record, in order. Failures come back as holds."""
    requests = [r.agent_request(clock=clock) for r in records]

    def one(request: dict[str, Any]) -> Call:
        return decide(agent, request, timeout_s=timeout_s)

    if workers <= 1:
        return [one(q) for q in requests]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, requests))


def approve_score(call: Call) -> float | None:
    """The confidence behind an approval, or None if the agent didn't approve.

    An approval with no confidence counts as 1.0, the same rule the runtime uses.
    """
    if call.failed or call.verdict.action != APPROVE:
        return None
    return 1.0 if call.verdict.confidence is None else call.verdict.confidence


def time_split(
    records: Sequence[Record], train_fraction: float
) -> tuple[list[Record], list[Record]]:
    """Oldest ``train_fraction`` for fitting, the newer rest for replay."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    ordered = sorted(records, key=lambda r: (r.decided_at, r.id))
    cut = round(len(ordered) * train_fraction)
    return ordered[:cut], ordered[cut:]


def stratified_sample(records: Sequence[Record], n: int, seed: int = 0) -> list[Record]:
    """``n`` records with the same share of human approvals as the whole set, in time order."""
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    approved = [r for r in records if r.label]
    skipped = [r for r in records if not r.label]
    k = round(n * len(approved) / len(records))
    picked = rng.sample(approved, k) + rng.sample(skipped, n - k)
    return sorted(picked, key=lambda r: (r.decided_at, r.id))


@dataclass(frozen=True)
class ArmReport:
    """One scorer on a labeled set, at the threshold chosen for the bars."""

    name: str
    n: int
    positives: int
    errors: int
    bars: Bars
    choice: OperatingPoint | None

    @property
    def failures(self) -> list[str]:
        if self.choice is None:
            return ["no threshold clears the precision bars"]
        return self.bars.failures(self.choice.confusion)

    @property
    def meets_bars(self) -> bool:
        return not self.failures


def evaluate_arm(
    name: str, items: Sequence[tuple[float | None, bool]], bars: Bars, errors: int = 0
) -> ArmReport:
    return ArmReport(
        name=name,
        n=len(items),
        positives=sum(1 for _, y in items if y),
        errors=errors,
        bars=bars,
        choice=choose_threshold(items, bars),
    )


@dataclass(frozen=True)
class Matched:
    """The baseline held to the candidate's precision, on the same decisions."""

    target_precision: float
    baseline: OperatingPoint | None
    candidate_only: int
    baseline_only: int
    p_value: float


@dataclass(frozen=True)
class Disagreement:
    id: str
    text: str
    agent: Verdict
    human_action: str
    basis: str


@dataclass(frozen=True)
class ReasonRow:
    reason: str
    n: int
    approve: int
    hold: int
    skip: int
    human_approved: int


@dataclass(frozen=True)
class ReplayReport:
    clock: bool
    candidate: ArmReport
    as_answered: Confusion
    finals: list[Verdict]
    matrix: dict[tuple[str, str], int]
    by_reason: list[ReasonRow]
    content_only: Confusion | None
    skip_agree: int
    skip_total: int
    p95_latency_s: float | None
    misses: list[Disagreement]
    false_approvals: list[Disagreement]
    baseline: ArmReport | None = None
    matched: Matched | None = None

    @property
    def threshold(self) -> float | None:
        return self.candidate.choice.threshold if self.candidate.choice else None


def _caught(items: Sequence[tuple[float | None, bool]], threshold: float | None) -> list[bool]:
    return [threshold is not None and s is not None and s >= threshold for s, y in items if y]


def replay(
    records: Sequence[Record],
    calls: Sequence[Call],
    bars: Bars,
    *,
    clock: bool = True,
    baseline: Callable[[Record], float] | None = None,
    worst: int = 5,
) -> ReplayReport:
    """Score the agent's calls against human labels and choose its threshold.

    ``records`` must all be human decisions. With ``baseline``, the baseline is scored on
    the same records, gets its own threshold, and is also held to the candidate's
    precision so the two compare on recall alone, with an exact McNemar test on the
    approvals one caught and the other missed.
    """
    records = require_human_labels(records, where="replay")
    if len(records) != len(calls) or any(
        r.id != c.id for r, c in zip(records, calls, strict=False)
    ):
        raise ValueError("calls must line up with records, one per record, in order")
    items = [(approve_score(c), bool(r.label)) for r, c in zip(records, calls, strict=True)]
    errors = sum(1 for c in calls if c.failed)
    cand = evaluate_arm("candidate", items, bars, errors)
    threshold = cand.choice.threshold if cand.choice else None
    finals = [_final(c, threshold) for c in calls]

    matrix: Counter[tuple[str, str]] = Counter()
    rows: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    skip_agree = skip_total = 0
    misses: list[Disagreement] = []
    false_approvals: list[Disagreement] = []
    for r, c, v in zip(records, calls, finals, strict=True):
        human = r.decision.action  # type: ignore[union-attr]
        matrix[(v.action, human)] += 1
        row = rows[c.verdict.reason]
        row[0] += 1
        row[{APPROVE: 1, HOLD: 2, SKIP: 3}[v.action]] += 1
        row[4] += 1 if r.label else 0
        if v.action == SKIP:
            skip_total += 1
            skip_agree += 0 if r.label else 1
            if r.label:
                misses.append(Disagreement(r.id, r.text, v, human, r.decision.basis))  # type: ignore[union-attr]
        if v.action == APPROVE and not r.label:
            false_approvals.append(Disagreement(r.id, r.text, v, human, r.decision.basis))  # type: ignore[union-attr]

    def by_confidence(d: Disagreement) -> tuple[float, str]:
        return (-(d.agent.confidence or 0.0), d.id)

    content = [
        (s, y) for (s, y), r in zip(items, records, strict=True) if r.decision.basis == "content"
    ]  # type: ignore[union-attr]
    report = ReplayReport(
        clock=clock,
        candidate=cand,
        as_answered=confusion_at(items, 0.0),
        finals=finals,
        matrix=dict(matrix),
        by_reason=sorted(
            (ReasonRow(k, *v) for k, v in rows.items()), key=lambda row: (-row.n, row.reason)
        ),
        content_only=confusion_at(content, threshold) if len(content) < len(items) else None,
        skip_agree=skip_agree,
        skip_total=skip_total,
        p95_latency_s=p95([c.latency_s for c in calls]),
        misses=sorted(misses, key=by_confidence)[:worst],
        false_approvals=sorted(false_approvals, key=by_confidence)[:worst],
    )
    if baseline is None:
        return report
    base_items = [(_safe(baseline, r), bool(r.label)) for r in records]
    base = evaluate_arm("baseline", base_items, bars, sum(1 for s, _ in base_items if s is None))
    matched = None
    if cand.choice is not None:
        target = cand.choice.confusion.precision or 0.0
        point = threshold_for_precision(base_items, target)
        c_caught = _caught(items, threshold)
        b_caught = _caught(base_items, point.threshold if point else None)
        only_c = sum(1 for x, y in zip(c_caught, b_caught, strict=True) if x and not y)
        only_b = sum(1 for x, y in zip(c_caught, b_caught, strict=True) if y and not x)
        matched = Matched(target, point, only_c, only_b, mcnemar_exact(only_c, only_b))
    return ReplayReport(**{**report.__dict__, "baseline": base, "matched": matched})


def _safe(scorer: Callable[[Record], float], record: Record) -> float | None:
    try:
        return float(scorer(record))
    except Exception:
        return None


def _final(call: Call, threshold: float | None) -> Verdict:
    """The verdict at the chosen threshold, or the model's own answer if none was chosen."""
    return apply_threshold(call.verdict, threshold)


def _rate(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _point(c: Confusion, confidence: float) -> str:
    return (
        f"approves {c.flagged}, right {c.tp}: precision {_rate(c.precision)} "
        f"{c.precision_interval(confidence)}; catches {c.tp} of {c.positives}: "
        f"recall {_rate(c.recall)} {c.recall_interval(confidence)}"
    )


def _arm(arm: ArmReport) -> list[str]:
    errors = f" ({arm.errors} model errors, counted as holds)" if arm.errors else ""
    if arm.choice is None:
        return [f"{arm.name:<10}no threshold clears the precision bars: gate stays closed{errors}"]
    verdict = "clears the bars" if arm.meets_bars else "misses: " + "; ".join(arm.failures)
    return [
        f"{arm.name:<10}threshold {arm.choice.threshold:.4g}: "
        f"{_point(arm.choice.confusion, arm.bars.confidence)}",
        f"{'':<10}{verdict}{errors}",
    ]


def _short(text: str, width: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def format_report(report: ReplayReport) -> str:
    cand = report.candidate
    bars = cand.bars
    pct = f"{bars.confidence:.0%}"
    lines = [
        f"{cand.n} human decisions, {cand.positives} approvals. "
        + ("Requests carried decidedAt." if report.clock else "CLOCK STRIPPED: no decidedAt."),
        f"Bars: precision >= {bars.min_precision:.2f} with Wilson {pct} lower bound >= "
        f"{bars.min_precision_lower_bound:.2f}; at least {bars.min_overlaps} decisions and "
        f"{bars.min_overlap_approvals} approvals"
        + (f"; recall >= {bars.min_recall:.2f}" if bars.min_recall else "")
        + ".",
        "",
        f"{'as asked':<10}the model's own answers, no threshold: "
        f"{_point(report.as_answered, bars.confidence)}",
        *_arm(cand),
    ]
    if report.baseline is not None:
        lines += _arm(report.baseline)
    m = report.matched
    if m is not None:
        lines += ["", f"Baseline held to the candidate's precision ({m.target_precision:.3f}):"]
        if m.baseline is None:
            lines.append("  no baseline threshold reaches it")
        else:
            lines.append(
                f"  threshold {m.baseline.threshold:.4g}: "
                f"{_point(m.baseline.confusion, bars.confidence)}"
            )
        lines.append(
            f"  approvals only the candidate caught: {m.candidate_only}; only the baseline: "
            f"{m.baseline_only}; exact McNemar p = {m.p_value:.3g}"
        )
    if report.content_only is not None and cand.choice is not None:
        lines += [
            "",
            "Content labels only (people deciding from the request, not from facts the agent",
            f"couldn't know): {_point(report.content_only, bars.confidence)}",
        ]
    where = (
        f"at threshold {report.threshold:.4g}"
        if report.threshold is not None
        else "as answered (no threshold cleared the bars)"
    )
    lines += [
        "",
        f"Agent {where} (rows) vs person (columns):",
        f"{'':>10}{'approve':>9}{'skip':>7}",
    ]
    for action in (APPROVE, HOLD, SKIP):
        a = report.matrix.get((action, "approve"), 0)
        s = report.matrix.get((action, "skip"), 0)
        lines.append(f"{action:>10}{a:>9}{s:>7}")
    if report.skip_total:
        lines.append(
            f"Skips: {report.skip_agree} of {report.skip_total} were skips for the person too."
        )
    lines += [
        "",
        "By the agent's reason:",
        f"  {'reason':<18}{'n':>5}{'approve':>9}{'hold':>6}{'skip':>6}{'person approved':>17}",
    ]
    for row in report.by_reason:
        lines.append(
            f"  {row.reason:<18}{row.n:>5}{row.approve:>9}{row.hold:>6}{row.skip:>6}"
            f"{row.human_approved:>17}"
        )
    if report.misses:
        lines += ["", "Worst misses (the person approved, the agent skipped):"]
        lines += [
            f"  {d.id}  skip/{d.agent.reason} {(d.agent.confidence or 0):.2f}  {_short(d.text)}"
            for d in report.misses
        ]
    if report.false_approvals:
        lines += ["", "Worst false approvals (the agent approved, the person skipped):"]
        lines += [
            f"  {d.id}  {(d.agent.confidence or 0):.2f} [{d.basis}]  {_short(d.text)}"
            for d in report.false_approvals
        ]
    if report.p95_latency_s is not None:
        lines += ["", f"p95 time per decision in this replay: {report.p95_latency_s:.3g} s"]
    return "\n".join(lines)


def report_dict(report: ReplayReport) -> dict[str, Any]:
    def arm(a: ArmReport | None) -> dict[str, Any] | None:
        if a is None:
            return None
        return {
            "n": a.n,
            "positives": a.positives,
            "errors": a.errors,
            "threshold": a.choice.threshold if a.choice else None,
            "at_threshold": a.choice.confusion.to_dict(a.bars.confidence) if a.choice else None,
            "meets_bars": a.meets_bars,
            "failures": a.failures,
        }

    out: dict[str, Any] = {
        "clock": report.clock,
        "as_answered": report.as_answered.to_dict(),
        "candidate": arm(report.candidate),
        "baseline": arm(report.baseline),
        "matrix": {f"{a}/{h}": n for (a, h), n in sorted(report.matrix.items())},
        "by_reason": [row.__dict__ for row in report.by_reason],
        "content_only": report.content_only.to_dict() if report.content_only else None,
        "skips": {"agree": report.skip_agree, "total": report.skip_total},
        "p95_latency_s": report.p95_latency_s,
    }
    if report.matched is not None:
        m = report.matched
        out["matched"] = {
            "target_precision": m.target_precision,
            "baseline_threshold": m.baseline.threshold if m.baseline else None,
            "baseline_at_target": m.baseline.confusion.to_dict() if m.baseline else None,
            "candidate_only": m.candidate_only,
            "baseline_only": m.baseline_only,
            "mcnemar_p": m.p_value,
        }
    return out
