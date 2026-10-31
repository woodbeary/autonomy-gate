import time

import pytest

from autonomy_gate.policy import (
    APPROVE,
    HOLD,
    SKIP,
    Verdict,
    VerdictError,
    apply_rails,
    apply_threshold,
    decide,
    final_verdict,
    require_clock,
)

REQ = {"id": "a", "decidedAt": "2026-03-01T10:00:00Z", "text": "x"}


def test_model_errors_become_holds():
    def boom(_):
        raise RuntimeError("upstream 500")

    for agent in (
        boom,
        lambda _: "approve",
        lambda _: {"action": "send"},
        lambda _: {"action": "approve", "reason": "x", "confidence": 1.7},
        lambda _: {"action": "approve", "reason": ""},
    ):
        call = decide(agent, REQ)
        assert call.verdict == Verdict(HOLD, "model_error")
        assert call.failed


def test_timeout_becomes_a_hold():
    def slow(_):
        time.sleep(0.5)
        return {"action": "approve", "reason": "x", "confidence": 1}

    call = decide(slow, REQ, timeout_s=0.05)
    assert call.verdict.action == HOLD and "timeout" in call.error


def test_agent_gets_a_copy_it_cannot_change():
    request = {"id": "a", "features": {"n": 1}}

    def meddle(r):
        r["features"]["n"] = 99
        return {"action": "hold", "reason": "x"}

    decide(meddle, request)
    assert request["features"]["n"] == 1


def test_threshold_turns_weak_approvals_into_holds():
    assert apply_threshold(Verdict(APPROVE, "x", 0.5), 0.6).action == HOLD
    assert apply_threshold(Verdict(APPROVE, "x", 0.6), 0.6).action == APPROVE
    assert apply_threshold(Verdict(SKIP, "x", 0.1), 0.6).action == SKIP
    assert apply_threshold(Verdict(APPROVE, "x"), 0.99).action == APPROVE


def test_rails_can_only_make_a_verdict_more_cautious():
    def upgrade(_req, _v):
        return Verdict(APPROVE, "rail says go", 1.0)

    def downgrade(_req, _v):
        return Verdict(SKIP, "out_of_area")

    def broken(_req, _v):
        raise KeyError("oops")

    assert apply_rails(REQ, Verdict(HOLD, "unsure"), [upgrade]).action == HOLD
    assert apply_rails(REQ, Verdict(SKIP, "x"), [upgrade]).action == SKIP
    assert apply_rails(REQ, Verdict(APPROVE, "x", 0.9), [downgrade, upgrade]).action == SKIP
    out = apply_rails(REQ, Verdict(APPROVE, "x", 0.9), [broken])
    assert out.action == HOLD and out.reason == "rail_error:broken"


def test_no_clock_no_approval():
    no_clock = {"id": "a", "text": "x"}
    assert require_clock(no_clock, Verdict(APPROVE, "x", 1.0)).reason == "no_clock"
    assert require_clock(REQ, Verdict(APPROVE, "x", 1.0)) is None
    call = decide(lambda _: {"action": "approve", "reason": "x", "confidence": 1}, no_clock)
    assert final_verdict(no_clock, call, 0.5).action == HOLD


def test_verdict_validation():
    with pytest.raises(VerdictError):
        Verdict("send", "x")
    with pytest.raises(VerdictError):
        Verdict(APPROVE, "x", float("nan"))
