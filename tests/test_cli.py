"""The README demo, end to end, through the command line."""

import filecmp
import json
from pathlib import Path

import pytest

from autonomy_gate.cli import main

ROOT = Path(__file__).resolve().parent.parent
C = "autonomy_gate.synthetic:candidate"


def test_committed_examples_are_exactly_what_synth_writes(tmp_path):
    assert main(["synth", str(tmp_path)]) == 0
    names = sorted(p.name for p in (ROOT / "examples").iterdir())
    assert names == sorted(p.name for p in tmp_path.iterdir())
    for name in names:
        assert filecmp.cmp(ROOT / "examples" / name, tmp_path / name, shallow=False), name


def test_demo(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["synth", "ex"]) == 0
    assert (
        main(
            [
                "lint",
                "ex/leads_history.jsonl",
                "ex/leads_heldout.jsonl",
                "ex/leads_live_labels.jsonl",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "replay",
                "ex/leads_history.jsonl",
                "--candidate",
                C,
                "--bars",
                "ex/gate.yaml",
                "--strip-clock",
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "CLOCK STRIPPED" in out and "misses: recall 0.647 < 0.70" in out

    assert (
        main(
            [
                "replay",
                "ex/leads_history.jsonl",
                "--candidate",
                C,
                "--bars",
                "ex/gate.yaml",
                "--baseline",
                "--lock",
                "gate.lock",
            ]
        )
        == 0
    )
    assert "Locked threshold" in capsys.readouterr().out

    assert main(["retest", "ex/leads_history.jsonl", "--candidate", C, "--lock", "gate.lock"]) == 2
    assert "refusing to re-test" in capsys.readouterr().err
    assert main(["retest", "ex/leads_heldout.jsonl", "--candidate", C, "--lock", "gate.lock"]) == 0
    assert "PASSED" in capsys.readouterr().out

    run = [
        "run",
        "ex/leads_live.jsonl",
        "--candidate",
        C,
        "--lock",
        "gate.lock",
        "--ledger",
        "ledger.db",
        "--switch",
        "switch.txt",
        "--live",
        "--outbox",
        "outbox.jsonl",
    ]
    assert main([*run, "--batch", "1"]) == 2
    assert "shadow check is GREEN" in capsys.readouterr().err

    assert (
        main(
            [
                "shadow",
                "ex/leads_live.jsonl",
                "--candidate",
                C,
                "--lock",
                "gate.lock",
                "--log",
                "shadow.jsonl",
                "--until",
                "2026-03-01T12:00:00Z",
            ]
        )
        == 0
    )
    check = [
        "check",
        "--shadow",
        "shadow.jsonl",
        "--labels",
        "ex/leads_live_labels.jsonl",
        "--lock",
        "gate.lock",
    ]
    capsys.readouterr()
    assert main(check) == 1
    out = capsys.readouterr().out
    assert out.startswith("RED") and "FAIL  overlaps with a person" in out
    assert (
        main(
            [
                "shadow",
                "ex/leads_live.jsonl",
                "--candidate",
                C,
                "--lock",
                "gate.lock",
                "--log",
                "shadow.jsonl",
            ]
        )
        == 0
    )
    assert main(check) == 0
    assert "\nGREEN: every bar passes." in capsys.readouterr().out

    assert main([*run, "--batch", "5"]) == 0
    assert "kill switch off" in capsys.readouterr().out
    assert main(["switch", "switch.txt", "on"]) == 0
    assert main([*run, "--batch", "10", "--cap", "2"]) == 0
    out = capsys.readouterr().out
    assert "would_approve" in out and " sent " in out
    sent = [json.loads(line) for line in Path("outbox.jsonl").read_text().splitlines()]
    assert len(sent) == len({row["request_id"] for row in sent}) == 2
    assert main(["ledger", "ledger.db"]) == 0


def test_lock_refuses_a_clockless_replay(tmp_path, capsys):
    main(["synth", str(tmp_path)])
    code = main(
        [
            "replay",
            str(tmp_path / "leads_history.jsonl"),
            "--candidate",
            C,
            "--strip-clock",
            "--lock",
            str(tmp_path / "x.lock"),
        ]
    )
    assert code == 2 and "clock stripped" in capsys.readouterr().err


@pytest.mark.parametrize("spec", ["nope", "autonomy_gate.synthetic:missing"])
def test_bad_candidate_spec_is_a_usage_error(tmp_path, spec, capsys):
    main(["synth", str(tmp_path)])
    assert main(["replay", str(tmp_path / "leads_history.jsonl"), "--candidate", spec]) == 2
