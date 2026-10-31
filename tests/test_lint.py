import json

from autonomy_gate.lint import lint_file


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def row(i, **over):
    base = {
        "id": f"r{i}",
        "decidedAt": "2026-03-01T10:00:00Z",
        "capturedAt": "2026-03-01T10:00:00Z",
        "request": {"text": "Moving out today, need a couch gone before 5pm."},
        "decision": {"action": "approve", "decidedBy": "human", "basis": "content"},
    }
    base.update(over)
    return base


def test_clean_file_passes_and_counts_relative_time(tmp_path):
    result = lint_file(write(tmp_path / "ok.jsonl", [row(1), row(2)]))
    assert result.ok, result.errors
    assert "2 of 2 requests use relative time" in result.notes[0]


def test_missing_clock_fails(tmp_path):
    bad = row(1)
    del bad["decidedAt"]
    result = lint_file(write(tmp_path / "f.jsonl", [bad, row(2, decidedAt="2026-03-01T10:00:00")]))
    assert not result.ok
    assert "no clock: decidedAt is missing" in result.errors[0]
    assert "no timezone" in result.errors[1]


def test_snapshot_taken_after_the_decision_is_leakage(tmp_path):
    result = lint_file(write(tmp_path / "f.jsonl", [row(1, capturedAt="2026-03-02T10:00:00Z")]))
    assert not result.ok
    assert "leakage: request captured 1 day, 0:00:00 after the decision" in result.errors[0]


def test_agent_decision_in_a_label_file_fails(tmp_path):
    agent = row(1, decision={"action": "approve", "decidedBy": "agent"})
    path = write(tmp_path / "f.jsonl", [agent, row(2)])
    result = lint_file(path)
    assert not result.ok and "provenance: decided by the agent" in result.errors[0]
    export = lint_file(path, labels=False)
    assert export.ok and "1 decisions were made by the agent" in export.notes[-1]


def test_decision_without_provenance_fails(tmp_path):
    result = lint_file(write(tmp_path / "f.jsonl", [row(1, decision={"action": "approve"})]))
    assert not result.ok and "provenance: decision has no decidedBy" in result.errors[0]


def test_bad_json_and_duplicates(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps(row(1)) + "\n" + json.dumps(row(1)) + "\n", encoding="utf-8")
    assert "duplicate id 'r1'" in lint_file(path).errors[0]
    path.write_text("{not json\n", encoding="utf-8")
    assert "not valid JSON" in lint_file(path).errors[0]
