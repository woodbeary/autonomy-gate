"""Check a decisions file before anything is scored on it.

Errors (the file can't be used as it is):

* **no clock**: a record has no ``decidedAt``, or one without a timezone. The agent would
  judge "today" against its own idea of today.
* **leakage**: ``capturedAt`` is after ``decidedAt``. The request snapshot was taken after
  the decision, so it can hold facts nobody had at decision time (a reply count that kept
  growing, a status that changed).
* **provenance**: a decision without ``decidedBy``, or, in a label file, a decision the
  agent made or one a person made after seeing the agent's suggestion
  (``human_assisted``). The agent's own decisions must never become its labels, and an
  assisted decision isn't an independent one.
* malformed JSON, a bad action, a duplicate id.

Notes (worth knowing, not errors): how many requests use relative time words, which is
how many depend on the clock.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import AGENT, HUMAN_ASSISTED, DataError, Decision, Record, parse_ts, read_jsonl

RELATIVE_TIME = re.compile(
    r"\b(today|tonight|tomorrow|this (morning|afternoon|evening|weekend|week)|"
    r"right now|asap|by \d{1,2}\s?(am|pm)|before \d{1,2}\s?(am|pm)|ends at \d{1,2})\b",
    re.IGNORECASE,
)


@dataclass
class LintResult:
    path: str
    records: int = 0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def lint_file(path: str | Path, *, labels: bool = True, limit: int = 20) -> LintResult:
    """Lint one JSONL file. ``labels=False`` allows agent-made decisions (a raw export)."""
    result = LintResult(str(path))
    seen: set[str] = set()
    relative = agent_made = assisted = 0
    errors: list[str] = []
    try:
        rows = list(read_jsonl(path))
    except (OSError, DataError) as exc:
        result.errors.append(str(exc))
        return result
    for lineno, obj in rows:
        result.records += 1
        where = f"{path}:{lineno}"
        problems = _check(obj)
        rid = obj.get("id") if isinstance(obj, Mapping) else None
        if isinstance(rid, str):
            where += f" ({rid})"
            if rid in seen:
                problems.append(f"duplicate id {rid!r}")
            seen.add(rid)
        decision = obj.get("decision") if isinstance(obj, Mapping) else None
        if isinstance(decision, Mapping) and decision.get("decidedBy") == AGENT:
            agent_made += 1
            if labels:
                problems.append(
                    "provenance: decided by the agent; agent decisions are never labels"
                )
        if isinstance(decision, Mapping) and decision.get("decidedBy") == HUMAN_ASSISTED:
            assisted += 1
            if labels:
                problems.append(
                    "provenance: decided by a person who saw the agent's suggestion; it isn't "
                    "an independent label (`autonomy-gate labels --assisted` keeps it apart)"
                )
        request = obj.get("request") if isinstance(obj, Mapping) else None
        if isinstance(request, Mapping) and RELATIVE_TIME.search(str(request.get("text", ""))):
            relative += 1
        errors += [f"{where}: {p}" for p in problems]
    result.errors = errors[:limit] + (
        [f"... and {len(errors) - limit} more"] if len(errors) > limit else []
    )
    if relative:
        result.notes.append(
            f"{relative} of {result.records} requests use relative time (today, tonight, "
            "by 5pm...); they can only be judged against decidedAt"
        )
    if agent_made and not labels:
        result.notes.append(
            f"{agent_made} decisions were made by the agent; `autonomy-gate labels` drops them"
        )
    if assisted and not labels:
        result.notes.append(
            f"{assisted} decisions were made by a person who saw the agent's suggestion; "
            "`autonomy-gate labels --assisted` keeps them apart"
        )
    return result


def _check(obj: Any) -> list[str]:
    if not isinstance(obj, Mapping):
        return ["not a JSON object"]
    problems: list[str] = []
    if not isinstance(obj.get("id"), str) or not obj.get("id"):
        problems.append("no id")
    decided_at = None
    if "decidedAt" not in obj:
        problems.append("no clock: decidedAt is missing")
    else:
        try:
            decided_at = parse_ts(obj["decidedAt"], "decidedAt")
        except DataError as exc:
            problems.append(f"no clock: {exc}")
    request = obj.get("request")
    if not isinstance(request, Mapping):
        problems.append("request must be an object")
    elif "decidedAt" in request:
        problems.append("decidedAt belongs on the record, not inside request")
    if obj.get("capturedAt") is not None:
        try:
            captured = parse_ts(obj["capturedAt"], "capturedAt")
        except DataError as exc:
            problems.append(str(exc))
        else:
            if decided_at is not None and captured > decided_at:
                problems.append(
                    f"leakage: request captured {captured - decided_at} after the decision"
                )
    if obj.get("decision") is not None:
        try:
            Decision.from_dict(obj["decision"])
        except DataError as exc:
            problems.append(f"provenance: {exc}" if "decidedBy" in str(exc) else str(exc))
    if not problems:
        try:
            Record.from_dict(obj)
        except DataError as exc:
            problems.append(str(exc))
    return problems
