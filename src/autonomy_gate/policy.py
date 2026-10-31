"""The decision policy: pure functions, no I/O, nothing that can send.

An agent is any callable that takes a request dict and returns a verdict: ``{"action":
"approve" | "hold" | "skip", "reason": "<code>", "confidence": 0..1}``. "approve" means act
now, "hold" means leave it for a person, "skip" means close it without acting.

Three rules make the policy fail closed:

* Severity runs approve < hold < skip. Rails can only move a verdict up that order, never
  back toward approve.
* Anything that goes wrong in the model call (an exception, a timeout, output that isn't a
  valid verdict) becomes a hold.
* An approval below the frozen confidence threshold becomes a hold.

``tests/test_isolation.py`` checks that this module can't reach network or send code.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from typing import Any

APPROVE = "approve"
HOLD = "hold"
SKIP = "skip"
ACTIONS = (APPROVE, HOLD, SKIP)
SEVERITY = {APPROVE: 0, HOLD: 1, SKIP: 2}

Agent = Callable[[dict[str, Any]], Any]


class VerdictError(ValueError):
    """The agent returned something that isn't a valid verdict."""


@dataclass(frozen=True)
class Verdict:
    action: str
    reason: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise VerdictError(f"action must be one of {ACTIONS}, got {self.action!r}")
        if not isinstance(self.reason, str) or not self.reason:
            raise VerdictError("reason must be a non-empty string")
        if self.confidence is not None:
            c = self.confidence
            if isinstance(c, bool) or not isinstance(c, int | float) or not math.isfinite(c):
                raise VerdictError("confidence must be a finite number")
            if not 0 <= c <= 1:
                raise VerdictError(f"confidence {c} is outside [0, 1]")
            object.__setattr__(self, "confidence", float(c))

    @property
    def severity(self) -> int:
        return SEVERITY[self.action]

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "confidence": self.confidence}


def parse_verdict(value: Any) -> Verdict:
    if isinstance(value, Verdict):
        return value
    if not isinstance(value, Mapping):
        raise VerdictError(f"expected a verdict object, got {type(value).__name__}")
    return Verdict(
        action=value.get("action"),
        reason=value.get("reason"),
        confidence=value.get("confidence"),
    )


@dataclass(frozen=True)
class Call:
    """One model call: the verdict it produced (a hold if it failed) and how long it took."""

    id: str
    verdict: Verdict
    latency_s: float
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


def decide(
    agent: Agent,
    request: Mapping[str, Any],
    *,
    timeout_s: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Call:
    """Call the agent once. Any failure becomes ``hold`` with reason ``model_error``.

    With ``timeout_s`` the call runs in a worker thread and a late answer is ignored. The
    agent gets its own copy of the request, so it can't change what is logged.
    """
    request_copy = _copy(request)
    rid = str(request.get("id", ""))
    start = clock()
    try:
        if timeout_s is None:
            raw = agent(request_copy)
        else:
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                raw = pool.submit(agent, request_copy).result(timeout=timeout_s)
            finally:
                pool.shutdown(wait=False)
        verdict = parse_verdict(raw)
    except FutureTimeout:
        return Call(
            rid, Verdict(HOLD, "model_error"), clock() - start, f"timeout after {timeout_s} s"
        )
    except Exception as exc:
        return Call(
            rid, Verdict(HOLD, "model_error"), clock() - start, f"{type(exc).__name__}: {exc}"
        )
    return Call(rid, verdict, clock() - start)


def _copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_copy(v) for v in value]
    return value


def apply_threshold(verdict: Verdict, threshold: float | None) -> Verdict:
    """An approval needs confidence at or above the frozen threshold, or it becomes a hold.

    An approval with no confidence counts as 1.0, the same rule the replay used to choose
    the threshold. ``threshold=None`` applies no threshold.
    """
    if verdict.action != APPROVE or threshold is None:
        return verdict
    confidence = 1.0 if verdict.confidence is None else verdict.confidence
    if confidence < threshold:
        return replace(verdict, action=HOLD, reason="below_threshold")
    return verdict


Rail = Callable[[Mapping[str, Any], Verdict], Verdict | None]


def apply_rails(request: Mapping[str, Any], verdict: Verdict, rails: Sequence[Rail]) -> Verdict:
    """Run every rail; the most severe verdict wins.

    A rail returns a verdict or None (no opinion). A rail that tries to make the verdict
    less cautious is ignored, and a rail that raises turns the verdict into a hold.
    """
    current = verdict
    for rail in rails:
        try:
            proposed = rail(request, current)
        except Exception:
            proposed = Verdict(HOLD, f"rail_error:{getattr(rail, '__name__', 'rail')}")
        if proposed is not None and proposed.severity > current.severity:
            current = proposed
    return current


def require_clock(request: Mapping[str, Any], verdict: Verdict) -> Verdict | None:
    """Hold any approval whose request carried no ``decidedAt``."""
    if verdict.action == APPROVE and not request.get("decidedAt"):
        return Verdict(HOLD, "no_clock", verdict.confidence)
    return None


DEFAULT_RAILS: tuple[Rail, ...] = (require_clock,)


def final_verdict(
    request: Mapping[str, Any],
    call: Call,
    threshold: float | None,
    rails: Sequence[Rail] = DEFAULT_RAILS,
) -> Verdict:
    return apply_rails(request, apply_threshold(call.verdict, threshold), rails)
