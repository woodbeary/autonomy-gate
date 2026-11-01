"""The agent's own decisions must never become its labels or count as agreement."""

import json
from pathlib import Path

import pytest

from autonomy_gate.golive import check
from autonomy_gate.lint import lint_file
from autonomy_gate.records import (
    AGENT,
    ProvenanceError,
    load_labels,
    require_human_labels,
    split_provenance,
)
from autonomy_gate.replay import replay, run_agent
from autonomy_gate.synthetic import candidate
from autonomy_gate.threshold import Bars

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LABEL_FILES = ["leads_history.jsonl", "leads_heldout.jsonl", "leads_live_labels.jsonl"]


@pytest.mark.parametrize("name", LABEL_FILES)
def test_committed_label_files_hold_only_human_decisions(name):
    result = lint_file(EXAMPLES / name)
    assert result.ok, "\n".join(result.errors)
    assert all(r.decision.decided_by == "human" for r in load_labels(EXAMPLES / name))


def test_an_agent_approval_written_into_a_label_file_is_refused(tmp_path):
    lines = (EXAMPLES / "leads_history.jsonl").read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[0])
    forged["decision"] = {"action": "approve", "decidedBy": AGENT, "basis": "content"}
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join([json.dumps(forged), *lines[1:]]) + "\n", encoding="utf-8")
    assert not lint_file(path).ok
    with pytest.raises(ProvenanceError):
        load_labels(path)


def test_replay_refuses_agent_decisions(record):
    records = [record(i, label=True) for i in range(5)] + [record(9, label=True, by=AGENT)]
    with pytest.raises(ProvenanceError):
        replay(records, run_agent(candidate, records), Bars())


def test_agreement_report_refuses_agent_decisions(record):
    labels = [record(1, label=True), record(2, label=True, by=AGENT)]
    entry = {
        "id": "r-0002",
        "decidedAt": "2026-03-01T09:10:00Z",
        "action": "approve",
        "latency_s": 0.1,
        "error": None,
        "fingerprint": "f",
    }
    with pytest.raises(ProvenanceError):
        check([entry], labels, Bars())


def test_split_keeps_agent_decisions_out_of_labels(record):
    records = [record(1, label=True), record(2, label=False, by=AGENT), record(3)]
    split = split_provenance(records)
    assert [r.id for r in split.human] == ["r-0001"]
    assert [r.id for r in split.agent] == ["r-0002"]
    assert [r.id for r in split.undecided] == ["r-0003"]
    assert split.assisted == []
    assert records[1].label is None
    require_human_labels(split.human)
