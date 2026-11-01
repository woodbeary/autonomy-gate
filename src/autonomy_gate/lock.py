"""The gate lock: a frozen threshold, what it was tuned on, and the results that earn it.

``autonomy-gate replay --lock`` writes it. ``autonomy-gate retest`` reads it, refuses any
held-out decision the gate has already seen, scores the rest at the frozen threshold, and
records the result. ``autonomy-gate check --lock`` records the shadow verdict. The runtime
acts only on a lock whose re-test and shadow check both passed, for the same candidate.

A lock takes one held-out result. After a re-test, a changed prompt or threshold needs a
new lock and new decisions, or the held-out set quietly becomes tuning data.

Ids and contents are stored as truncated SHA-256 hashes, so a lock can be committed
without the decisions in it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .metrics import confusion_at
from .policy import Call
from .records import Record, iso, parse_ts, require_human_labels
from .replay import ReplayReport, approve_score
from .threshold import Bars

FORMAT = "autonomy-gate.lock/1"


class LockError(Exception):
    """The lock is missing, malformed, or can't take this operation."""


class DataReuseError(LockError):
    """Held-out decisions overlap what the gate was tuned or already tested on."""


def _h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def fingerprint(*parts: Any) -> str:
    """A stable hash of whatever defines the candidate: model, prompt, schema, version.

    The lock stores it when you tune. Shadow logs carry it, and the check and the runtime
    refuse a candidate whose fingerprint differs from the one that was tested.
    """
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def agent_fingerprint(agent: Any, spec: str, clock: bool) -> str:
    """The candidate fingerprint a lock records for ``agent``.

    It is the agent's own ``fingerprint`` attribute if it has one, else the ``--candidate``
    spec it was loaded from, together with whether its requests carry the clock.
    """
    return fingerprint({"agent": getattr(agent, "fingerprint", None) or spec, "clock": clock})


@dataclass
class GateLock:
    threshold: float
    bars: Bars
    fingerprint: str
    tuning: dict[str, Any]
    seen_through: str
    tuning_ids: set[str] = field(default_factory=set)
    tuning_content: set[str] = field(default_factory=set)
    spent_ids: set[str] = field(default_factory=set)
    spent_content: set[str] = field(default_factory=set)
    created_at: str = field(default_factory=_now)
    retest: dict[str, Any] | None = None
    golive: dict[str, Any] | None = None

    @classmethod
    def from_replay(
        cls,
        seen: Sequence[Record],
        report: ReplayReport,
        *,
        fingerprint: str,
        previous: GateLock | None = None,
    ) -> GateLock:
        """Freeze the threshold a replay chose.

        ``seen`` is everything the gate looked at while tuning: the replayed records and
        anything a baseline was fitted on. Pass ``previous`` when you re-tune after a
        re-test, so everything that lock saw, its held-out set included, stays off-limits.
        """
        arm = report.candidate
        if arm.choice is None:
            raise LockError("no threshold clears the bars, so there is nothing to lock")
        if not seen:
            raise LockError("no decisions")
        newest = max(r.decided_at for r in seen)
        spent_ids: set[str] = set()
        spent_content: set[str] = set()
        if previous is not None:
            spent_ids = previous.spent_ids | previous.tuning_ids
            spent_content = previous.spent_content | previous.tuning_content
            newest = max(newest, parse_ts(previous.seen_through))
            if previous.retest:
                spent_ids |= set(previous.retest.get("ids", []))
                spent_content |= set(previous.retest.get("content", []))
                newest = max(newest, parse_ts(previous.retest["newest"]))
        c = arm.choice.confusion
        return cls(
            threshold=arm.choice.threshold,
            bars=arm.bars,
            fingerprint=fingerprint,
            tuning={
                "n": arm.n,
                "positives": arm.positives,
                "errors": arm.errors,
                "meets_bars": arm.meets_bars,
                "clock": report.clock,
                **c.to_dict(arm.bars.confidence),
            },
            seen_through=iso(newest),
            tuning_ids={_h(r.id) for r in seen},
            tuning_content={r.content_hash[:32] for r in seen},
            spent_ids=spent_ids,
            spent_content=spent_content,
        )

    def check_fresh(self, records: Sequence[Record], *, allow_older: bool = False) -> None:
        """Raise :class:`DataReuseError` unless every record is new to this gate.

        New means an id the gate hasn't seen, content (clock and request) it hasn't seen
        under another id, and, unless ``allow_older``, a ``decidedAt`` after the newest
        decision it has seen.
        """
        seen_ids = self.tuning_ids | self.spent_ids
        seen_content = self.tuning_content | self.spent_content
        reused = [r.id for r in records if _h(r.id) in seen_ids]
        copied = [
            r.id for r in records if r.content_hash[:32] in seen_content and r.id not in reused
        ]
        problems: list[str] = []
        if reused:
            problems.append(f"{len(reused)} ids were already used: {_show(reused)}")
        if copied:
            problems.append(
                f"{len(copied)} records repeat requests already used under other ids: "
                f"{_show(copied)}"
            )
        if not allow_older:
            cutoff = parse_ts(self.seen_through)
            old = [r.id for r in records if r.decided_at <= cutoff]
            if old:
                problems.append(
                    f"{len(old)} records are not newer than {self.seen_through}, the newest "
                    f"the gate has seen: {_show(old)}"
                )
        if problems:
            raise DataReuseError(
                "refusing to re-test on data the gate has already seen; "
                + "; ".join(problems)
                + ". Collect decisions made after the lock was written."
            )

    def run_retest(
        self, records: Sequence[Record], calls: Sequence[Call], *, allow_older: bool = False
    ) -> dict[str, Any]:
        """Score held-out decisions at the frozen threshold and record the result.

        The threshold is never re-chosen here. Raises :class:`LockError` if this lock
        already holds a re-test, and :class:`DataReuseError` if any record isn't new.
        """
        if self.retest is not None:
            raise LockError(
                "this lock already holds a held-out result; tune a new lock with "
                "`replay --previous-lock` and re-test on decisions it hasn't seen"
            )
        if not records:
            raise LockError("no held-out decisions")
        records = require_human_labels(records, where="retest")
        self.check_fresh(records, allow_older=allow_older)
        if [r.id for r in records] != [c.id for c in calls]:
            raise LockError("calls must line up with records")
        items = [(approve_score(c), bool(r.label)) for r, c in zip(records, calls, strict=True)]
        c = confusion_at(items, self.threshold)
        failures = self.bars.failures(c)
        self.retest = {
            "ran_at": _now(),
            "n": c.n,
            "positives": c.positives,
            "errors": sum(1 for call in calls if call.failed),
            **c.to_dict(self.bars.confidence),
            "passed": not failures,
            "failures": failures,
            "newest": iso(max(r.decided_at for r in records)),
            "ids": sorted(_h(r.id) for r in records),
            "content": sorted(r.content_hash[:32] for r in records),
        }
        return self.retest

    def record_check(self, result: Mapping[str, Any]) -> None:
        """Keep the latest shadow check. A RED check replaces an earlier GREEN one."""
        self.golive = {"checked_at": _now(), **dict(result)}

    @property
    def retest_passed(self) -> bool:
        return bool(self.retest and self.retest.get("passed"))

    @property
    def shadow_passed(self) -> bool:
        return bool(self.golive and self.golive.get("passed"))

    @property
    def skips_earned(self) -> bool:
        return bool(self.golive and self.golive.get("skips_earned"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "created_at": self.created_at,
            "fingerprint": self.fingerprint,
            "threshold": self.threshold,
            "bars": self.bars.to_dict(),
            "seen_through": self.seen_through,
            "tuning": self.tuning,
            "tuning_ids": sorted(self.tuning_ids),
            "tuning_content": sorted(self.tuning_content),
            "spent_ids": sorted(self.spent_ids),
            "spent_content": sorted(self.spent_content),
            "retest": self.retest,
            "golive": self.golive,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> GateLock:
        if obj.get("format") != FORMAT:
            raise LockError(f"not a {FORMAT} file")
        try:
            return cls(
                threshold=float(obj["threshold"]),
                bars=Bars.from_dict(obj["bars"]),
                fingerprint=str(obj["fingerprint"]),
                tuning=dict(obj["tuning"]),
                seen_through=obj["seen_through"],
                tuning_ids=set(obj["tuning_ids"]),
                tuning_content=set(obj["tuning_content"]),
                spent_ids=set(obj.get("spent_ids", [])),
                spent_content=set(obj.get("spent_content", [])),
                created_at=obj["created_at"],
                retest=obj.get("retest"),
                golive=obj.get("golive"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LockError(f"malformed lock: {exc}") from exc

    def save(self, path: str | Path) -> None:
        """Write atomically, so a crash never leaves half a lock."""
        path = Path(path)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".lock-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> GateLock:
        try:
            with open(path, encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise LockError(f"can't read lock {path}: {exc}") from exc
        return cls.from_dict(obj)


def _show(ids: list[str]) -> str:
    return ", ".join(ids[:5]) + (" ..." if len(ids) > 5 else "")
