"""A sender must act at most once per request id: a repeated send replays, never re-applies."""

import pytest

from autonomy_gate.actuator import SenderContractError, check_sender
from autonomy_gate.synthetic import FileOutbox


def lines(path):
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def test_the_demo_outbox_keeps_the_contract(tmp_path):
    outbox = tmp_path / "outbox.jsonl"
    receipt = check_sender(FileOutbox(outbox), actions=lambda: lines(outbox))
    assert receipt and lines(outbox) == 1


class Appender:
    def __init__(self):
        self.done = {}
        self.count = 0

    def send(self, request_id, payload):
        self.count += 1
        self.done[request_id] = f"rcpt-{self.count}"
        return self.done[request_id]

    def lookup(self, request_id):
        return self.done.get(request_id)


class SameReceiptActsTwice(Appender):
    def send(self, request_id, payload):
        self.count += 1
        self.done.setdefault(request_id, "rcpt-1")
        return self.done[request_id]


class ForgetfulLookup(Appender):
    def lookup(self, request_id):
        return None


class SeesEverything(Appender):
    def lookup(self, request_id):
        return "rcpt-0"


@pytest.mark.parametrize(
    ("sender", "step"),
    [
        (Appender(), "a second send with the same request id returned"),
        (SameReceiptActsTwice(), "acted 2 times"),
        (ForgetfulLookup(), "lookup after the send returned"),
        (SeesEverything(), "lookup before any send returned"),
    ],
)
def test_a_sender_that_breaks_the_contract_is_named(sender, step):
    with pytest.raises(SenderContractError, match=step):
        check_sender(sender, actions=lambda: sender.count)
