"""The go-live check: does the shadow record clear every bar?

It joins the shadow log to human decisions on the same requests (the overlaps) and checks
the bars one by one. It reports RED and names each failing bar until all of them pass.
Only human decisions count: the label file goes through the same provenance check as
every other label path, so the agent can't agree with itself.

Approvals and skips are judged separately. Approvals gate going live. Skips (closing a
request without a person) stay advisory until they reach their own bar; until then the
runtime routes them to a person with the agent's suggestion attached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .lock import GateLock
from .metrics import Confusion, p95, wilson
from .policy import APPROVE, SKIP
from .records import Record, parse_ts, require_human_labels
from .threshold import Bars


@dataclass(frozen=True)
class BarResult:
    name: str
    value: str
    need: str
    ok: bool


@dataclass(frozen=True)
class CheckReport:
    rows: list[BarResult]
    confusion: Confusion
    entries: int
    skip_overlaps: int
    skip_agree: int
    skips_earned: bool
    bars: Bars

    @property
    def failing(self) -> list[BarResult]:
        return [r for r in self.rows if not r.ok]

    @property
    def passed(self) -> bool:
        return not self.failing

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failing": [f"{r.name} {r.value}, need {r.need}" for r in self.failing],
            "skips_earned": self.skips_earned,
            "overlaps": self.confusion.n,
            "approvals": self.confusion.flagged,
            "precision": self.confusion.precision,
            "recall": self.confusion.recall,
            "shadow_entries": self.entries,
        }


def check(
    shadow: Sequence[Mapping[str, Any]],
    labels: Sequence[Record],
    bars: Bars,
    lock: GateLock | None = None,
) -> CheckReport:
    labels = require_human_labels(labels, where="check")
    human = {r.id: r for r in labels}
    rows: list[BarResult] = []

    times = [parse_ts(e["decidedAt"], "decidedAt") for e in shadow]
    days = (max(times) - min(times)).total_seconds() / 86400 if times else 0.0
    rows.append(
        BarResult(
            "days of shadow",
            f"{days:.1f}",
            f">= {bars.min_shadow_days:g}",
            days >= bars.min_shadow_days,
        )
    )

    overlaps = [(e, human[e["id"]]) for e in shadow if e["id"] in human]
    tp = sum(1 for e, r in overlaps if e["action"] == APPROVE and r.label)
    fp = sum(1 for e, r in overlaps if e["action"] == APPROVE and not r.label)
    fn = sum(1 for e, r in overlaps if e["action"] != APPROVE and r.label)
    c = Confusion(tp, fp, fn, len(overlaps) - tp - fp - fn)
    rows.append(
        BarResult(
            "overlaps with a person", str(c.n), f">= {bars.min_overlaps}", c.n >= bars.min_overlaps
        )
    )
    rows.append(
        BarResult(
            "person approvals in overlaps",
            str(c.positives),
            f">= {bars.min_overlap_approvals}",
            c.positives >= bars.min_overlap_approvals,
        )
    )
    if c.precision is None:
        rows.append(
            BarResult("precision", "n/a (no approvals)", f">= {bars.min_precision:.2f}", False)
        )
        rows.append(
            BarResult(
                f"Wilson {bars.confidence:.0%} lower bound",
                "n/a",
                f">= {bars.min_precision_lower_bound:.2f}",
                False,
            )
        )
    else:
        low = c.precision_interval(bars.confidence).low
        rows.append(
            BarResult(
                "precision",
                f"{c.tp}/{c.flagged} = {c.precision:.3f}",
                f">= {bars.min_precision:.2f}",
                c.precision >= bars.min_precision,
            )
        )
        rows.append(
            BarResult(
                f"Wilson {bars.confidence:.0%} lower bound",
                f"{low:.3f}",
                f">= {bars.min_precision_lower_bound:.2f}",
                low >= bars.min_precision_lower_bound,
            )
        )
    recall = c.recall
    rows.append(
        BarResult(
            "recall",
            "n/a" if recall is None else f"{c.tp}/{c.positives} = {recall:.3f}",
            f">= {bars.min_recall:.2f}",
            (recall or 0.0) >= bars.min_recall,
        )
    )

    latency = p95([float(e["latency_s"]) for e in shadow if e.get("latency_s") is not None])
    rows.append(
        BarResult(
            "p95 decision time",
            "n/a" if latency is None else f"{latency:.3g} s",
            f"<= {bars.max_p95_latency_s:g} s",
            latency is not None and latency <= bars.max_p95_latency_s,
        )
    )
    failed = sum(1 for e in shadow if e.get("error"))
    share = failed / len(shadow) if shadow else 1.0
    rows.append(
        BarResult(
            "model unavailable",
            f"{failed}/{len(shadow)} = {share:.1%}",
            f"<= {bars.max_unavailable:.0%}",
            bool(shadow) and share <= bars.max_unavailable,
        )
    )

    if lock is not None:
        others = sum(1 for e in shadow if e.get("fingerprint") != lock.fingerprint)
        rows.append(
            BarResult(
                "shadow entries from the locked candidate",
                f"{len(shadow) - others}/{len(shadow)}",
                "all",
                others == 0,
            )
        )
        state = "missing" if lock.retest is None else ("passed" if lock.retest_passed else "failed")
        rows.append(BarResult("held-out re-test", state, "passed", lock.retest_passed))

    skips = [(e, r) for e, r in overlaps if e["action"] == SKIP]
    skip_agree = sum(1 for _, r in skips if not r.label)
    skips_earned = (
        len(skips) >= bars.skip_min_overlaps
        and len(skips) > 0
        and skip_agree / len(skips) >= bars.skip_min_precision
    )
    return CheckReport(rows, c, len(shadow), len(skips), skip_agree, skips_earned, bars)


def format_check(report: CheckReport) -> str:
    failing = report.failing
    head = (
        "GREEN: every bar passes."
        if report.passed
        else f"RED: {len(failing)} of {len(report.rows)} bars fail."
    )
    lines = [head]
    width = max(len(r.name) for r in report.rows)
    for r in report.rows:
        lines.append(
            f"  {'ok  ' if r.ok else 'FAIL'}  {r.name:<{width}}  {r.value:<18}  need {r.need}"
        )
    b = report.bars
    n, agree = report.skip_overlaps, report.skip_agree
    rate = f"{agree}/{n} = {agree / n:.3f}" if n else "none yet"
    low = wilson(agree, n, b.confidence).low if n else 0.0
    if report.skips_earned:
        lines.append(f"Skips: earned ({rate} on skip overlaps); they may close requests.")
    else:
        lines.append(
            f"Skips: advisory ({rate}, lower bound {low:.3f}); closing a request without a "
            f"person needs >= {b.skip_min_precision:.2f} on >= {b.skip_min_overlaps}."
        )
    return "\n".join(lines)
