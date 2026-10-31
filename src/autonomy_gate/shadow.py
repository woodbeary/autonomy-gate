"""Shadow mode: run the agent on live requests, write down what it would have done, send nothing.

:class:`Shadow` has no sender parameter and no slot to hold one, and this module's
imports can't reach the actuator or any network code (``tests/test_isolation.py`` walks
them). The agent gets a copy of each request and nothing else.

Each line of the shadow log says what the gate would have done: the final action after
the frozen threshold and the rails, the model's own answer, how long the call took, and
whether it failed. When a person later decides the same request, that pair is an
"overlap", and ``autonomy-gate check`` scores the overlaps against the go-live bars.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .policy import DEFAULT_RAILS, Agent, Rail, decide, final_verdict
from .records import DataError, Record, iso, read_jsonl


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Shadow:
    """Score live requests at the frozen threshold and log them. Never acts."""

    __slots__ = ("_agent", "_fingerprint", "_log", "_mutex", "_rails", "_threshold", "_timeout")

    def __init__(
        self,
        agent: Agent,
        log: str | Path,
        *,
        fingerprint: str,
        threshold: float | None = None,
        rails: Sequence[Rail] = DEFAULT_RAILS,
        timeout_s: float | None = None,
    ) -> None:
        self._agent = agent
        self._log = Path(log)
        self._fingerprint = fingerprint
        self._threshold = threshold
        self._rails = tuple(rails)
        self._timeout = timeout_s
        self._mutex = threading.Lock()

    def observe(self, record: Record) -> dict[str, Any]:
        request = record.agent_request(clock=True)
        call = decide(self._agent, request, timeout_s=self._timeout)
        verdict = final_verdict(request, call, self._threshold, self._rails)
        entry = {
            "mode": "shadow",
            "id": record.id,
            "decidedAt": iso(record.decided_at),
            "action": verdict.action,
            "reason": verdict.reason,
            "confidence": verdict.confidence,
            "model": call.verdict.to_dict(),
            "latency_s": round(call.latency_s, 6),
            "error": call.error,
            "fingerprint": self._fingerprint,
            "loggedAt": _now(),
        }
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
        with self._mutex, open(self._log, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        return entry

    def run(self, records: Iterable[Record]) -> list[dict[str, Any]]:
        """Observe every record not already in the log, in order."""
        done = {e["id"] for e in load_shadow(self._log)} if self._log.exists() else set()
        return [self.observe(r) for r in records if r.id not in done]


def load_shadow(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lineno, obj in read_jsonl(path):
        if not isinstance(obj, dict) or obj.get("mode") != "shadow" or "id" not in obj:
            raise DataError(f"{path}:{lineno}: not a shadow log line")
        out.append(obj)
    return out
