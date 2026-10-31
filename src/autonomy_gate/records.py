"""Logged decisions: what the agent saw, when, and who decided.

One JSON object per line::

    {"id": "lead-00012",
     "decidedAt": "2026-03-04T15:20:00Z",
     "capturedAt": "2026-03-04T15:20:00Z",
     "request": {"text": "...", "postedAt": "...", "features": {"distance_miles": 4.2}},
     "decision": {"action": "approve", "decidedBy": "human", "basis": "content"}}

``decidedAt`` is the clock. It is when the decision is made, and it goes into every request
the agent sees, because a model asked about "today's sale" with no date judges it against
its own idea of today.

``capturedAt`` is when the request snapshot was taken. A snapshot taken after the decision
can carry facts nobody had at decision time.

``decision`` is present once someone has decided. ``decidedBy`` says who: ``human`` or
``agent``. Only human decisions are labels. ``basis`` says whether the person decided from
the request itself (``content``) or from facts the agent could not have known, such as a
full schedule (``operational``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

HUMAN = "human"
AGENT = "agent"
PROVENANCE = (HUMAN, AGENT)
LABEL_ACTIONS = ("approve", "skip")
BASES = ("content", "operational")


class DataError(ValueError):
    """A record or file is malformed."""


class ProvenanceError(DataError):
    """An agent-made decision reached a place that only takes human decisions."""


def parse_ts(value: Any, field: str = "timestamp") -> datetime:
    """Parse an ISO 8601 timestamp that names its timezone (``Z`` or an offset)."""
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value)
        except ValueError as exc:
            raise DataError(f"{field} {value!r} is not an ISO 8601 timestamp") from exc
    else:
        raise DataError(f"{field} must be an ISO 8601 string")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise DataError(f"{field} {value!r} has no timezone; write it with Z or an offset")
    return ts


def iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class Decision:
    """A final decision on a request, and who made it."""

    action: str
    decided_by: str
    basis: str = "content"
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.decided_by not in PROVENANCE:
            raise DataError(f"decidedBy must be one of {PROVENANCE}, got {self.decided_by!r}")
        if self.action not in LABEL_ACTIONS:
            raise DataError(f"decision action must be one of {LABEL_ACTIONS}")
        if self.basis not in BASES:
            raise DataError(f"basis must be one of {BASES}")

    @property
    def approved(self) -> bool:
        return self.action == "approve"

    @classmethod
    def from_dict(cls, obj: Any) -> Decision:
        if not isinstance(obj, Mapping):
            raise DataError("decision must be an object")
        if "decidedBy" not in obj:
            raise DataError("decision has no decidedBy: say whether a human or the agent made it")
        return cls(
            action=obj.get("action"),
            decided_by=obj["decidedBy"],
            basis=obj.get("basis", "content"),
            reason=obj.get("reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        out = {"action": self.action, "decidedBy": self.decided_by, "basis": self.basis}
        if self.reason is not None:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class Record:
    """One request the agent could act on. Frozen all the way down."""

    id: str
    decided_at: datetime
    request: Mapping[str, Any]
    captured_at: datetime | None = None
    decision: Decision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise DataError("id must be a non-empty string")
        object.__setattr__(self, "decided_at", parse_ts(self.decided_at, "decidedAt"))
        if self.captured_at is not None:
            object.__setattr__(self, "captured_at", parse_ts(self.captured_at, "capturedAt"))
        if not isinstance(self.request, Mapping):
            raise DataError(f"{self.id}: request must be an object")
        if "decidedAt" in self.request:
            raise DataError(f"{self.id}: put decidedAt on the record, not inside request")
        object.__setattr__(self, "request", _freeze(self.request))

    @property
    def text(self) -> str:
        return str(self.request.get("text", ""))

    @property
    def features(self) -> Mapping[str, Any]:
        features = self.request.get("features")
        return features if isinstance(features, Mapping) else MappingProxyType({})

    @property
    def label(self) -> bool | None:
        """True if a person approved, False if a person skipped, None otherwise."""
        if self.decision is None or self.decision.decided_by != HUMAN:
            return None
        return self.decision.approved

    def agent_request(self, *, clock: bool = True) -> dict[str, Any]:
        """What the agent is shown: a fresh copy of the request, plus the clock.

        It never includes the decision. ``clock=False`` leaves out ``decidedAt``; it exists
        only to reproduce what a model does without one.
        """
        out: dict[str, Any] = {"id": self.id}
        if clock:
            out["decidedAt"] = iso(self.decided_at)
        out.update(_thaw(self.request))
        return out

    @property
    def content_hash(self) -> str:
        """SHA-256 of what the agent saw and when. The id and the decision are left out."""
        payload = json.dumps(
            {"decidedAt": iso(self.decided_at), "request": _thaw(self.request)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, obj: Any) -> Record:
        if not isinstance(obj, Mapping):
            raise DataError("each line must be a JSON object")
        if "id" not in obj:
            raise DataError("missing id")
        if "decidedAt" not in obj:
            raise DataError(f"{obj['id']}: no decidedAt; every request needs its clock")
        decision = obj.get("decision")
        return cls(
            id=obj["id"],
            decided_at=obj["decidedAt"],
            request=obj.get("request") or {},
            captured_at=obj.get("capturedAt"),
            decision=None if decision is None else Decision.from_dict(decision),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "decidedAt": iso(self.decided_at)}
        if self.captured_at is not None:
            out["capturedAt"] = iso(self.captured_at)
        out["request"] = _thaw(self.request)
        if self.decision is not None:
            out["decision"] = self.decision.to_dict()
        return out

    def without_decision(self) -> Record:
        return Record(self.id, self.decided_at, self.request, self.captured_at, None)


def read_jsonl(path: str | Path) -> Iterator[tuple[int, Any]]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno}: not valid JSON ({exc.msg})") from exc


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_records(path: str | Path) -> list[Record]:
    """Read records of any provenance. Ids must be unique."""
    out: list[Record] = []
    seen: set[str] = set()
    for lineno, obj in read_jsonl(path):
        try:
            record = Record.from_dict(obj)
        except DataError as exc:
            raise DataError(f"{path}:{lineno}: {exc}") from exc
        if record.id in seen:
            raise DataError(f"{path}:{lineno}: duplicate id {record.id!r}")
        seen.add(record.id)
        out.append(record)
    return out


def require_human_labels(records: Sequence[Record], where: str = "labels") -> list[Record]:
    """Return ``records`` if every one carries a human decision; raise otherwise.

    Every path that turns decisions into labels or agreement numbers goes through here, so
    the agent's own decisions can never become the thing it is graded against.
    """
    for r in records:
        if r.decision is None:
            raise DataError(f"{where}: {r.id} has no decision")
        if r.decision.decided_by != HUMAN:
            raise ProvenanceError(
                f"{where}: {r.id} was decided by {r.decision.decided_by!r}. Only human "
                "decisions are labels; filter the export with `autonomy-gate labels`."
            )
    return list(records)


def load_labels(path: str | Path) -> list[Record]:
    """Read a label file. Fails if any record isn't a human decision."""
    return require_human_labels(load_records(path), where=str(path))


def split_provenance(records: Sequence[Record]) -> tuple[list[Record], list[Record], list[Record]]:
    """Split an export into (human-decided, agent-decided, undecided)."""
    human: list[Record] = []
    agent: list[Record] = []
    undecided: list[Record] = []
    for r in records:
        if r.decision is None:
            undecided.append(r)
        elif r.decision.decided_by == HUMAN:
            human.append(r)
        else:
            agent.append(r)
    return human, agent, undecided
