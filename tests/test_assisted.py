"""A person who saw the agent's suggestion is not an independent label.

Such decisions say ``decidedBy: "human_assisted"``. Every advisory skip produces one: a
person decides with the agent's suggestion attached. They are reported apart, never choose
the threshold, never count toward a go-live bar, and are never a held-out re-test.
"""

import json

import pytest

from autonomy_gate.cli import main
from autonomy_gate.golive import check, format_check
from autonomy_gate.lint import lint_file
from autonomy_gate.lock import GateLock
from autonomy_gate.records import (
    HUMAN_ASSISTED,
    DataError,
    ProvenanceError,
    load_assisted,
    load_labels,
    require_human_labels,
    split_provenance,
    write_jsonl,
)
from autonomy_gate.replay import replay, run_agent
from autonomy_gate.synthetic import candidate
from autonomy_gate.threshold import Bars

C = "autonomy_gate.synthetic:candidate"


def assisted(record):
    return record(9, label=True, by=HUMAN_ASSISTED)


def test_an_assisted_decision_is_neither_a_label_nor_an_agent_decision(record):
    rows = [
        record(1, label=True),
        record(2, label=False, by=HUMAN_ASSISTED),
        record(3, label=True, by="agent"),
        record(4),
    ]
    split = split_provenance(rows)
    assert [[r.id for r in group] for group in split] == [
        ["r-0001"],
        ["r-0002"],
        ["r-0003"],
        ["r-0004"],
    ]
    assert split.assisted == [rows[1]]
    assert rows[1].label is None and rows[1].assisted_label is False
    assert rows[0].assisted_label is None


def test_every_label_path_refuses_an_assisted_decision(sets, record, tmp_path):
    bad = assisted(record)
    with pytest.raises(ProvenanceError, match="saw the agent's suggestion"):
        require_human_labels([bad])
    path = tmp_path / "labels.jsonl"
    write_jsonl(path, [bad.to_dict()])
    with pytest.raises(ProvenanceError):
        load_labels(path)
    with pytest.raises(ProvenanceError):
        replay([bad], run_agent(candidate, [bad]), Bars())
    entry = {
        "id": bad.id,
        "decidedAt": "2026-03-01T09:10:00Z",
        "action": "approve",
        "latency_s": 0.1,
        "error": None,
        "fingerprint": "f",
    }
    with pytest.raises(ProvenanceError):
        check([entry], [bad], Bars())


def test_an_assisted_decision_is_never_a_held_out_retest(sets):
    history = sets["history"]
    lock = GateLock.from_replay(
        history, replay(history, run_agent(candidate, history), Bars()), fingerprint="fp"
    )
    heldout = list(sets["heldout"])
    original = heldout[0]
    heldout[0] = type(original)(
        original.id,
        original.decided_at,
        original.request,
        original.captured_at,
        type(original.decision)("approve", HUMAN_ASSISTED, "content"),
    )
    with pytest.raises(ProvenanceError, match="saw the agent's suggestion"):
        lock.run_retest(heldout, run_agent(candidate, heldout))
    assert lock.retest is None


def test_lint_fails_a_label_file_with_one_and_notes_it_in_an_export(record, tmp_path):
    path = tmp_path / "labels.jsonl"
    write_jsonl(path, [record(1, label=True).to_dict(), assisted(record).to_dict()])
    as_labels = lint_file(path)
    assert not as_labels.ok
    assert any("saw the agent's suggestion" in e for e in as_labels.errors)
    as_export = lint_file(path, labels=False)
    assert as_export.ok
    assert any("1 decisions were made by a person who saw" in n for n in as_export.notes)


def test_labels_writes_assisted_decisions_to_their_own_file(record, tmp_path, capsys):
    export = tmp_path / "export.jsonl"
    rows = [
        record(1, label=True),
        record(2, label=False, by=HUMAN_ASSISTED),
        record(3, label=True, by="agent"),
        record(4),
    ]
    write_jsonl(export, [r.to_dict() for r in rows])
    labels, kept = tmp_path / "labels.jsonl", tmp_path / "assisted.jsonl"
    assert main(["labels", str(export), "-o", str(labels), "--assisted", str(kept)]) == 0
    out = capsys.readouterr().out
    assert "wrote 1 human decisions" in out and "1 made by a person who saw" in out
    assert [r.id for r in load_labels(labels)] == ["r-0001"]
    assert [r.id for r in load_assisted(kept)] == ["r-0002"]
    assert main(["labels", str(export), "-o", str(labels)]) == 0
    assert "dropped 1 made by a person who saw" in capsys.readouterr().out


def test_load_assisted_takes_only_assisted_decisions(record, tmp_path):
    path = tmp_path / "assisted.jsonl"
    write_jsonl(path, [record(1, label=True).to_dict()])
    with pytest.raises(ProvenanceError, match="human_assisted"):
        load_assisted(path)


def flipped(records):
    out = []
    for r in records:
        row = r.to_dict()
        row["id"] = "assisted-" + r.id
        row["decision"] = {
            "action": "skip" if r.label else "approve",
            "decidedBy": HUMAN_ASSISTED,
            "basis": "content",
        }
        out.append(row)
    return out


def test_replay_reports_assisted_decisions_apart_and_they_never_move_the_threshold(
    sets, tmp_path, capsys
):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [r.to_dict() for r in sets["history"]])
    path = tmp_path / "assisted.jsonl"
    write_jsonl(path, flipped(sets["history"][:60]))

    main(["replay", str(labels), "--candidate", C])
    alone = capsys.readouterr().out
    lock = tmp_path / "gate.lock"
    main(["replay", str(labels), "--candidate", C, "--assisted", str(path), "--lock", str(lock)])
    both = capsys.readouterr().out

    def threshold_line(text):
        return next(line for line in text.splitlines() if line.startswith("candidate threshold"))

    assert threshold_line(alone) == threshold_line(both)
    assert "Assisted decisions" in both and "Assisted decisions" not in alone
    assert "60 decisions" in both
    saved = GateLock.load(lock)
    with pytest.raises(Exception, match="already used"):
        saved.check_fresh(load_assisted(path), allow_older=True)


def test_an_id_in_both_files_is_refused(sets, tmp_path, capsys):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [r.to_dict() for r in sets["history"][:50]])
    row = sets["history"][0].to_dict()
    row["decision"]["decidedBy"] = HUMAN_ASSISTED
    path = tmp_path / "assisted.jsonl"
    write_jsonl(path, [row])
    assert main(["replay", str(labels), "--candidate", C, "--assisted", str(path)]) == 2
    assert "both" in capsys.readouterr().err


def test_check_counts_assisted_overlaps_toward_no_bar(record):
    people = [record(i, label=True) for i in range(1, 4)]
    helped = [record(i, label=i % 2 == 0, by=HUMAN_ASSISTED) for i in range(4, 10)]
    shadow = [
        {
            "id": r.id,
            "decidedAt": "2026-03-01T09:00:00Z",
            "action": "approve",
            "latency_s": 0.1,
            "error": None,
            "fingerprint": "f",
        }
        for r in people + helped
    ]
    report = check(shadow, people, Bars(), assisted=helped)
    assert report.confusion.n == 3
    assert report.assisted_overlaps == 6 and report.assisted_agree == 3
    text = format_check(report)
    assert "Assisted overlaps" in text and "count toward no bar" in text
    assert report.summary()["assisted_overlaps"] == 6
    with pytest.raises(DataError, match="both"):
        check(shadow, people, Bars(), assisted=[record(1, label=True, by=HUMAN_ASSISTED)])


def test_check_cli_takes_an_assisted_file(record, tmp_path, capsys):
    people = [record(i, label=True) for i in range(1, 4)]
    helped = [record(i, label=True, by=HUMAN_ASSISTED) for i in range(4, 6)]
    write_jsonl(tmp_path / "labels.jsonl", [r.to_dict() for r in people])
    write_jsonl(tmp_path / "assisted.jsonl", [r.to_dict() for r in helped])
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text(
        "".join(
            json.dumps(
                {
                    "mode": "shadow",
                    "id": r.id,
                    "decidedAt": "2026-03-01T09:00:00Z",
                    "action": "approve",
                    "latency_s": 0.1,
                    "error": None,
                    "fingerprint": "f",
                }
            )
            + "\n"
            for r in people + helped
        ),
        encoding="utf-8",
    )
    args = ["check", "--shadow", str(shadow), "--labels", str(tmp_path / "labels.jsonl")]
    main([*args, "--assisted", str(tmp_path / "assisted.jsonl")])
    assert "Assisted overlaps (a person who saw the suggestion): 2" in capsys.readouterr().out
