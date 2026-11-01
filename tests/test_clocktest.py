"""The clock harness: the same decisions asked with and without decidedAt, paired.

Every number here comes from the synthetic stand-in, whose formula shares the person's, so
the size of the drop is built in. The tests check the harness, not any model.
"""

import json
import textwrap

import pytest

from autonomy_gate.adapters.common import DEFAULT_INSTRUCTIONS
from autonomy_gate.cli import main
from autonomy_gate.clocktest import measure_clock
from autonomy_gate.lock import GateLock, agent_fingerprint
from autonomy_gate.synthetic import candidate
from autonomy_gate.usage import UsageMeter

C = "autonomy_gate.synthetic:candidate"


def test_the_two_arms_differ_only_in_the_clock(sets):
    seen = {True: [], False: []}

    def recording(request):
        seen["decidedAt" in request].append(request["id"])
        return candidate(request)

    records = sets["history"][:40]
    measure_clock(recording, records)
    assert seen[True] == seen[False] == [r.id for r in records]


def test_the_stand_in_loses_same_day_approvals_without_the_clock(sets):
    report = measure_clock(candidate, sets["history"])
    with_clock, without = report.with_clock, report.without_clock
    assert with_clock.confusion.tp == 334 - with_clock.confusion.fn
    assert with_clock.confusion.recall > without.confusion.recall
    assert report.only_with > 0 and report.p_value < 0.001
    assert report.only_with - report.only_without == with_clock.confusion.tp - without.confusion.tp
    relative_drop = with_clock.relative.recall - without.relative.recall
    other_drop = with_clock.other.recall - without.other.recall
    assert relative_drop > 0.3 > other_drop
    assert report.relative_n + report.other_n == len(sets["history"])
    assert with_clock.usage is None and report.cost_usd is None


class Metered:
    def __init__(self):
        self.usage = UsageMeter()
        self.model = "fake-model"

    def describe(self):
        return {"provider": "fake", "model": self.model, "effort": "low"}

    def __call__(self, request):
        clocked = "decidedAt" in request
        self.usage.add(input_tokens=100 if clocked else 90, output_tokens=10)
        return candidate(request)


def test_tokens_and_cost_are_counted_per_arm(sets):
    agent = Metered()
    records = sets["history"][:50]
    report = measure_clock(agent, records, usd_per_mtok=(2.0, 10.0))
    assert report.with_clock.usage["input_tokens"] == 5000
    assert report.without_clock.usage["input_tokens"] == 4500
    assert report.with_clock.usage["calls"] == report.without_clock.usage["calls"] == 50
    expected = (9500 * 2.0 + 1000 * 10.0) / 1e6
    assert abs(report.cost_usd["total"] - expected) < 1e-12
    assert report.meta["agent"] == {"provider": "fake", "model": "fake-model", "effort": "low"}


def test_the_command_writes_a_report_with_model_date_n_effort_and_cost(tmp_path, capsys):
    main(["synth", str(tmp_path / "ex")])
    agent_file = tmp_path / "fake_agent.py"
    agent_file.write_text(
        textwrap.dedent(
            """
            from autonomy_gate.synthetic import candidate
            from autonomy_gate.usage import UsageMeter

            class Agent:
                fingerprint = "fake v1"

                def __init__(self):
                    self.usage = UsageMeter()

                def describe(self):
                    return {"provider": "fake", "model": "fake-model", "effort": "low"}

                def __call__(self, request):
                    self.usage.add(input_tokens=120, output_tokens=8)
                    return candidate(request)
            """
        ),
        encoding="utf-8",
    )
    out = tmp_path / "clock-report.json"
    code = main(
        [
            "clock-test",
            str(tmp_path / "ex" / "leads_history.jsonl"),
            "--candidate",
            f"{agent_file}:Agent",
            "--sample",
            "100",
            "--usd-per-mtok-in",
            "4",
            "--usd-per-mtok-out",
            "20",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "with clock" in text and "without clock" in text and "McNemar" in text
    report = json.loads(out.read_text(encoding="utf-8"))
    meta = report["meta"]
    assert meta["agent"]["model"] == "fake-model" and meta["agent"]["effort"] == "low"
    assert meta["n"] == 100 and meta["sample"] == 100 and meta["ran_at"].endswith("Z")
    assert meta["labels_sha256"] and meta["fingerprint"].startswith("sha256:")
    assert report["arms"]["with_clock"]["usage"]["calls"] == 100
    assert report["cost_usd"]["total"] == pytest.approx((200 * 120 * 4 + 200 * 8 * 20) / 1e6)
    assert set(report["arms"]) == {"with_clock", "without_clock"}
    assert "only_with_clock" in report["paired"]


def test_without_prices_there_is_no_cost(tmp_path, capsys):
    main(["synth", str(tmp_path)])
    capsys.readouterr()
    code = main(["clock-test", str(tmp_path / "leads_history.jsonl"), "--candidate", C])
    assert code == 0
    text = capsys.readouterr().out
    assert "no usage reported" in text and "--usd-per-mtok-in" in text


def test_the_report_carries_the_fingerprint_the_lock_records(tmp_path, capsys):
    main(["synth", str(tmp_path)])
    agent_file = tmp_path / "plain_agent.py"
    agent_file.write_text(
        "from autonomy_gate.synthetic import candidate\n\n\n"
        "def plain(request):\n"
        "    return candidate(request)\n",
        encoding="utf-8",
    )
    spec = f"{agent_file}:plain"
    labels = str(tmp_path / "leads_history.jsonl")
    lock, out = tmp_path / "gate.lock", tmp_path / "clock.json"
    main(["replay", labels, "--candidate", spec, "--lock", str(lock)])
    main(["clock-test", labels, "--candidate", spec, "--sample", "40", "--out", str(out)])
    capsys.readouterr()
    recorded = json.loads(out.read_text(encoding="utf-8"))["meta"]["fingerprint"]
    assert recorded == GateLock.load(lock).fingerprint


def test_a_plain_callable_is_fingerprinted_by_its_spec_or_not_at_all(sets):
    records = sets["history"][:10]

    def plain(request):
        return candidate(request)

    assert measure_clock(plain, records).meta["fingerprint"] is None
    named = measure_clock(plain, records, candidate="mine:plain").meta["fingerprint"]
    other = measure_clock(plain, records, candidate="mine:other").meta["fingerprint"]
    assert named == agent_fingerprint(plain, "mine:plain", True) != other
    own = measure_clock(candidate, records).meta["fingerprint"]
    assert own == agent_fingerprint(candidate, C, True)


def test_the_default_instructions_do_not_tell_the_bare_arm_it_has_a_clock():
    text = " ".join(DEFAULT_INSTRUCTIONS.split())
    assert "If the request includes decidedAt" in text
    assert "The request includes decidedAt" not in text
