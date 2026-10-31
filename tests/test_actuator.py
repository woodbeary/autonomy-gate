import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from autonomy_gate.actuator import Guard, NotEarned
from autonomy_gate.killswitch import FileKillSwitch, StaticKillSwitch
from autonomy_gate.ledger import Ledger, LedgerError
from autonomy_gate.lock import GateLock
from autonomy_gate.replay import replay, run_agent
from autonomy_gate.synthetic import FileOutbox, candidate
from autonomy_gate.threshold import Bars

SRC = Path(__file__).resolve().parent.parent / "src"


def approve_all(request):
    return {"action": "approve", "reason": "ok", "confidence": 0.99}


def make_lock(sets, *, retest=True, shadow=True, skips=False):
    history = sets["history"]
    lock = GateLock.from_replay(
        history, replay(history, run_agent(candidate, history), Bars()), fingerprint="fp"
    )
    if retest:
        lock.retest = {"passed": True}
    if shadow:
        lock.golive = {"passed": True, "skips_earned": skips}
    return lock


class Recorder:
    def __init__(self, fail=None):
        self.sent, self.lookups, self.fail = [], [], fail

    def send(self, request_id, payload):
        self.sent.append(request_id)
        if self.fail:
            raise self.fail
        return f"rcpt-{len(self.sent)}"

    def lookup(self, request_id):
        self.lookups.append(request_id)
        return None


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as db:
        yield db


def test_live_mode_needs_an_earned_lock_for_this_candidate(sets, ledger):
    on = StaticKillSwitch(True)
    for kwargs, message in (({"retest": False}, "re-test"), ({"shadow": False}, "shadow")):
        with pytest.raises(NotEarned, match=message):
            Guard(
                candidate,
                ledger,
                on,
                make_lock(sets, **kwargs),
                fingerprint="fp",
                sender=Recorder(),
            )
    with pytest.raises(NotEarned, match="fingerprint"):
        Guard(candidate, ledger, on, make_lock(sets), fingerprint="other", sender=Recorder())
    Guard(candidate, ledger, on, make_lock(sets, retest=False), fingerprint="other")


def test_live_off_holds_everything(sets, ledger):
    guard = Guard(
        approve_all, ledger, StaticKillSwitch(True), make_lock(sets, skips=True), fingerprint="fp"
    )
    report = guard.cycle(sets["live"][:5])
    assert {s for _, s, _ in report.outcomes} == {"held"}
    assert all(why == "live_off" for _, _, why in report.outcomes)


def test_kill_switch_off_or_unreadable_holds_everything(sets, ledger, tmp_path):
    sender = Recorder()
    switch = FileKillSwitch(tmp_path / "switch")
    guard = Guard(
        candidate, ledger, switch, make_lock(sets, skips=True), fingerprint="fp", sender=sender
    )
    report = guard.cycle(sets["live"][:10])
    assert report.count("held") == 10 and not sender.sent
    switch.set(False)
    assert guard.cycle(sets["live"][10:20]).count("held") == 10
    (tmp_path / "switch").write_text("yes please", encoding="utf-8")
    assert guard.cycle(sets["live"][20:25]).count("held") == 5
    assert not sender.sent


def test_switch_read_right_before_each_send(sets, ledger):
    class FlipsAfterFirstRead:
        def __init__(self):
            self.reads = 0

        def state(self):
            self.reads += 1
            return StaticKillSwitch(self.reads == 1).state()

    sender = Recorder()
    guard = Guard(
        approve_all, ledger, FlipsAfterFirstRead(), make_lock(sets), fingerprint="fp", sender=sender
    )
    report = guard.cycle(sets["live"][:3])
    assert not sender.sent and report.count("held") == 3


def test_model_error_becomes_a_hold_and_skips_are_advisory(sets, ledger):
    def broken(request):
        if request["id"] == sets["live"][0].id:
            raise RuntimeError("model down")
        return {"action": "skip", "reason": "not_a_job", "confidence": 0.9}

    sender = Recorder()
    guard = Guard(
        broken, ledger, StaticKillSwitch(True), make_lock(sets), fingerprint="fp", sender=sender
    )
    report = guard.cycle(sets["live"][:3])
    assert report.outcomes[0][1:] == ("held", "model_error")
    assert [s for _, s, _ in report.outcomes[1:]] == ["advisory_skip", "advisory_skip"]
    earned = Guard(
        broken,
        ledger,
        StaticKillSwitch(True),
        make_lock(sets, skips=True),
        fingerprint="fp",
        sender=sender,
    )
    assert earned.cycle(sets["live"][3:5]).count("skipped") == 2
    assert not sender.sent


def test_cap_defers_approvals_and_retries_them_without_asking_the_model_again(sets, ledger):
    asked = []

    def counting(request):
        asked.append(request["id"])
        return approve_all(request)

    sender = Recorder()
    guard = Guard(
        counting,
        ledger,
        StaticKillSwitch(True),
        make_lock(sets),
        fingerprint="fp",
        sender=sender,
        max_sends_per_cycle=2,
    )
    first = guard.cycle(sets["live"][:5])
    assert first.count("sent") == 2 and first.count("would_approve") == 3
    second = guard.cycle([])
    assert second.count("sent") == 2 and second.count("would_approve") == 1
    assert len(asked) == 5
    assert len(sender.sent) == len(set(sender.sent)) == 4


def test_a_send_that_raises_is_settled_by_lookup_never_resent(sets, ledger):
    sender = Recorder(fail=TimeoutError("no answer"))
    guard = Guard(
        approve_all,
        ledger,
        StaticKillSwitch(True),
        make_lock(sets),
        fingerprint="fp",
        sender=sender,
    )
    record = sets["live"][0]
    report = guard.cycle([record])
    assert report.outcomes == [
        (record.id, "unconfirmed_never_resubmitted", "send raised TimeoutError")
    ]
    assert sender.lookups == sender.sent
    guard.cycle([record])
    assert len(sender.sent) == 1


def test_a_send_with_no_receipt_is_a_claim_not_a_fact(sets, ledger):
    class Silent(Recorder):
        def send(self, request_id, payload):
            self.sent.append(request_id)
            return None

        def lookup(self, request_id):
            return "rcpt-found"

    guard = Guard(
        approve_all,
        ledger,
        StaticKillSwitch(True),
        make_lock(sets),
        fingerprint="fp",
        sender=Silent(),
    )
    report = guard.cycle(sets["live"][:1])
    assert report.outcomes[0][1:] == ("sent", "confirmed_by_lookup")


def test_ledger_refuses_a_second_send(ledger):
    ledger.record(
        "x",
        decided_at="2026-03-01T00:00:00Z",
        action="approve",
        reason="r",
        confidence=1.0,
        state="sending",
        fingerprint="fp",
    )
    ledger.move("x", "sending", "unconfirmed_never_resubmitted", "lost")
    with pytest.raises(LedgerError):
        ledger.record(
            "x",
            decided_at="2026-03-01T00:00:00Z",
            action="approve",
            reason="r",
            confidence=1.0,
            state="sending",
            fingerprint="fp",
        )
    with pytest.raises(LedgerError):
        ledger.move("x", "unconfirmed_never_resubmitted", "sending")
    with pytest.raises(LedgerError):
        ledger.begin_send("x")


CRASHING_RUN = textwrap.dedent(
    """
    import sys
    from autonomy_gate.actuator import Guard
    from autonomy_gate.killswitch import StaticKillSwitch
    from autonomy_gate.ledger import Ledger
    from autonomy_gate.lock import GateLock
    from autonomy_gate.records import load_records
    from autonomy_gate.synthetic import FileOutbox

    db, lock_path, outbox, crash, requests = sys.argv[1:6]
    records = load_records(requests)
    guard = Guard(lambda r: {"action": "approve", "reason": "ok", "confidence": 0.99},
                  Ledger(db), StaticKillSwitch(True), GateLock.load(lock_path),
                  fingerprint="fp", sender=FileOutbox(outbox, crash=crash))
    guard.cycle(records)
    """
)


@pytest.mark.parametrize(
    ("crash", "expected"),
    [("before_write", "unconfirmed_never_resubmitted"), ("after_write", "sent")],
)
def test_hard_crash_mid_send_is_never_sent_twice(sets, tmp_path, crash, expected):
    lock = make_lock(sets)
    lock.save(tmp_path / "gate.lock")
    requests = tmp_path / "requests.jsonl"
    record = sets["live"][0]
    requests.write_text(__import__("json").dumps(record.to_dict()) + "\n", encoding="utf-8")
    script = tmp_path / "crash.py"
    script.write_text(CRASHING_RUN, encoding="utf-8")
    outbox = tmp_path / "outbox.jsonl"
    db = tmp_path / "ledger.db"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(db),
            str(tmp_path / "gate.lock"),
            str(outbox),
            crash,
            str(requests),
        ],
        env={"PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 137, proc.stderr
    with Ledger(db) as ledger:
        assert ledger.get(record.id).state == "sending"
        guard = Guard(
            approve_all,
            ledger,
            StaticKillSwitch(True),
            lock,
            fingerprint="fp",
            sender=FileOutbox(outbox),
        )
        report = guard.cycle([record])
        row = ledger.get(record.id)
        assert row.state == expected
        assert report.outcomes[0][1] == expected
        guard.cycle([record])
    lines = outbox.read_text(encoding="utf-8").splitlines() if outbox.exists() else []
    assert len(lines) == (1 if crash == "after_write" else 0)
