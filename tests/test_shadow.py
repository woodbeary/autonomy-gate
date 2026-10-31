import inspect
import socket

import pytest

from autonomy_gate.shadow import Shadow, load_shadow
from autonomy_gate.synthetic import candidate


def test_shadow_takes_no_sender():
    params = inspect.signature(Shadow).parameters
    assert not any(name in params for name in ("sender", "transport", "send", "outbox"))
    with pytest.raises(TypeError):
        Shadow(candidate, "log.jsonl", fingerprint="f", sender=object())
    shadow = Shadow(candidate, "log.jsonl", fingerprint="f")
    with pytest.raises(AttributeError):
        shadow.sender = object()


def test_shadow_run_opens_no_socket(sets, tmp_path, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("shadow mode opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    shadow = Shadow(candidate, tmp_path / "s.jsonl", fingerprint="f", threshold=0.6)
    entries = shadow.run(sets["live"][:40])
    assert len(entries) == 40


def test_the_agent_sees_only_a_copy_of_the_request(sets, tmp_path):
    seen = []

    def agent(request):
        seen.append(request)
        request["text"] = "tampered"
        return {"action": "approve", "reason": "x", "confidence": 1.0}

    record = sets["live"][0]
    Shadow(agent, tmp_path / "s.jsonl", fingerprint="f").observe(record)
    assert set(seen[0]) == {"id", "decidedAt", "text", "postedAt", "features"}
    assert record.text != "tampered"


def test_log_records_what_would_have_happened_and_is_resumable(sets, tmp_path):
    log = tmp_path / "s.jsonl"

    def failing(_):
        raise RuntimeError("model down")

    Shadow(failing, log, fingerprint="f").run(sets["live"][:3])
    Shadow(candidate, log, fingerprint="f").run(sets["live"][:5])
    entries = load_shadow(log)
    assert [e["id"] for e in entries] == [r.id for r in sets["live"][:5]]
    assert entries[0]["action"] == "hold" and entries[0]["error"].startswith("RuntimeError")
    assert entries[3]["error"] is None and entries[3]["fingerprint"] == "f"
