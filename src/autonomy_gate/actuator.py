"""The guarded actuator: the only code in the package that can act.

:class:`Guard` runs one cycle at a time over new requests:

1. Settle anything a crash left in ``sending``: look it up by request id; if the lookup
   can't confirm it, close it ``unconfirmed_never_resubmitted``. Never send it again.
2. Read the kill switch. Unreadable means off.
3. Send deferred approvals (``would_approve``) from their stored verdicts, without asking
   the model again, while the per-cycle cap has room.
4. For each request not already in the ledger: ask the model (an error becomes a hold),
   apply the frozen threshold and the rails (which only make a verdict more cautious),
   then route it. With live mode off or the switch off, everything is held for a person.
   A skip closes a request only once skips have earned it; until then it is an
   ``advisory_skip``. An approval is sent only under the cap, and its ledger row is
   committed before the sender is called.

Live mode needs a sender and a lock whose held-out re-test and shadow check both passed
for this exact candidate. Without a sender (live off, the default) nothing is ever sent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .killswitch import KillSwitch
from .ledger import Ledger, LedgerError
from .lock import GateLock
from .policy import DEFAULT_RAILS, HOLD, SKIP, Agent, Rail, Verdict, decide, final_verdict
from .records import Record, iso


class NotEarned(Exception):
    """The candidate hasn't earned the right to act."""


class Sender(Protocol):
    def send(self, request_id: str, payload: Mapping[str, Any]) -> str | None:
        """Act once. Use ``request_id`` as the idempotency key. Return a receipt id."""

    def lookup(self, request_id: str) -> str | None:
        """Return the receipt for ``request_id`` if the action happened, else None."""


def default_payload(record: Record, verdict: Verdict) -> dict[str, Any]:
    return {"item_id": record.id, "reason": verdict.reason}


@dataclass
class CycleReport:
    switch_on: bool
    outcomes: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, item_id: str, state: str, why: str) -> None:
        self.outcomes.append((item_id, state, why))

    def count(self, state: str) -> int:
        return sum(1 for _, s, _ in self.outcomes if s == state)


class Guard:
    def __init__(
        self,
        agent: Agent,
        ledger: Ledger,
        switch: KillSwitch,
        lock: GateLock,
        *,
        fingerprint: str,
        sender: Sender | None = None,
        max_sends_per_cycle: int | None = None,
        rails: Sequence[Rail] = DEFAULT_RAILS,
        make_payload: Callable[[Record, Verdict], Mapping[str, Any]] = default_payload,
        timeout_s: float | None = None,
    ) -> None:
        if sender is not None:
            if not lock.retest_passed:
                raise NotEarned("live mode needs a lock whose held-out re-test passed")
            if not lock.shadow_passed:
                raise NotEarned("live mode needs a lock whose latest shadow check is GREEN")
            if fingerprint != lock.fingerprint:
                raise NotEarned("this candidate's fingerprint differs from the one that was tested")
        if max_sends_per_cycle is not None and max_sends_per_cycle < 0:
            raise ValueError("max_sends_per_cycle must be >= 0")
        self._agent = agent
        self._ledger = ledger
        self._switch = switch
        self._lock = lock
        self._fingerprint = fingerprint
        self._sender = sender
        self._cap = max_sends_per_cycle
        self._rails = tuple(rails)
        self._make_payload = make_payload
        self._timeout = timeout_s

    @property
    def live(self) -> bool:
        return self._sender is not None

    def _switch_on(self) -> tuple[bool, str]:
        try:
            state = self._switch.state()
        except Exception as exc:
            return False, f"switch unreadable: {type(exc).__name__}"
        return bool(state.running), state.detail

    def recover(self, report: CycleReport | None = None) -> None:
        """Settle every ``sending`` row by lookup. Nothing is ever resent."""
        for row in self._ledger.in_state("sending"):
            state, why = self._settle(
                row.item_id, row.request_id or "", "found in sending at start"
            )
            if report is not None:
                report.add(row.item_id, state, why)

    def _settle(self, item_id: str, request_id: str, why: str) -> tuple[str, str]:
        receipt = None
        if self._sender is not None:
            try:
                receipt = self._sender.lookup(request_id)
            except Exception as exc:
                why = f"{why}; lookup failed: {type(exc).__name__}"
        if receipt:
            self._ledger.move(
                item_id, "sending", "sent", f"confirmed_by_lookup ({why})", receipt=str(receipt)
            )
            return "sent", "confirmed_by_lookup"
        self._ledger.move(item_id, "sending", "unconfirmed_never_resubmitted", why)
        return "unconfirmed_never_resubmitted", why

    def _send(self, item_id: str, request_id: str, payload: Mapping[str, Any]) -> tuple[str, str]:
        if self._sender is None:
            raise NotEarned("no sender: live mode is off")
        try:
            receipt = self._sender.send(request_id, payload)
        except Exception as exc:
            return self._settle(item_id, request_id, f"send raised {type(exc).__name__}")
        if not receipt:
            return self._settle(item_id, request_id, "send returned no receipt")
        self._ledger.move(item_id, "sending", "sent", "receipt", receipt=str(receipt))
        return "sent", "receipt"

    def _write(
        self,
        record: Record,
        verdict: Verdict,
        state: str,
        why: str,
        report: CycleReport,
        payload: Mapping[str, Any] | None = None,
    ) -> str | None:
        request_id = self._ledger.record(
            record.id,
            decided_at=iso(record.decided_at),
            action=verdict.action,
            reason=verdict.reason,
            confidence=verdict.confidence,
            state=state,
            fingerprint=self._fingerprint,
            payload=payload,
            detail=why,
        )
        if state != "sending":
            report.add(record.id, state, why)
        return request_id

    def _route(self, record: Record, report: CycleReport, budget: int | None) -> int | None:
        """Decide one new request and act on it if everything allows. Returns the budget left."""
        request = record.agent_request(clock=True)
        call = decide(self._agent, request, timeout_s=self._timeout)
        verdict = final_verdict(request, call, self._lock.threshold, self._rails)
        if verdict.action == HOLD:
            self._write(record, verdict, "held", verdict.reason, report)
            return budget
        if not self.live:
            self._write(record, verdict, "held", "live_off", report)
            return budget
        on, detail = self._switch_on()
        if not on:
            self._write(record, verdict, "held", f"kill_switch: {detail}", report)
            return budget
        if verdict.action == SKIP:
            state = "skipped" if self._lock.skips_earned else "advisory_skip"
            self._write(record, verdict, state, verdict.reason, report)
            return budget
        try:
            payload = dict(self._make_payload(record, verdict))
        except Exception as exc:
            self._write(record, verdict, "held", f"payload_error: {type(exc).__name__}", report)
            return budget
        if budget is not None and budget <= 0:
            self._write(record, verdict, "would_approve", "cap_reached", report, payload)
            return budget
        request_id = self._write(record, verdict, "sending", "intent", report, payload)
        if request_id is None:
            raise LedgerError(f"{record.id}: no request id for a send")
        report.add(record.id, *self._send(record.id, request_id, payload))
        return None if budget is None else budget - 1

    def cycle(self, records: Sequence[Record]) -> CycleReport:
        """Run one cycle: settle, read the switch, send the backlog, then route new requests."""
        settled = CycleReport(switch_on=False)
        self.recover(settled)
        on, detail = self._switch_on()
        report = CycleReport(switch_on=on, outcomes=settled.outcomes)
        budget = self._cap

        for row in self._ledger.in_state("would_approve"):
            if not self.live or not on:
                why = "live_off" if not self.live else f"kill_switch: {detail}"
                self._ledger.move(row.item_id, "would_approve", "held", why)
                report.add(row.item_id, "held", why)
            elif budget is not None and budget <= 0:
                report.add(row.item_id, "would_approve", "cap_reached")
            else:
                on, detail = self._switch_on()
                if not on:
                    why = f"kill_switch: {detail}"
                    self._ledger.move(row.item_id, "would_approve", "held", why)
                    report.add(row.item_id, "held", why)
                    continue
                request_id = self._ledger.begin_send(row.item_id)
                report.add(row.item_id, *self._send(row.item_id, request_id, row.payload or {}))
                budget = None if budget is None else budget - 1

        for record in records:
            if not self._ledger.has(record.id):
                budget = self._route(record, report, budget)
        return report
