"""Each adapter through its real SDK, against a fake API on 127.0.0.1 with a fake key.

These run only where the SDKs are installed (``pip install -e ".[dev,openai,anthropic]"``;
CI has a job for it). They check what the SDK actually puts on the wire and how it hands the
answer back, which a fake client can't. Nothing leaves the machine.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from autonomy_gate.adapters.common import VERDICT_SCHEMA
from autonomy_gate.cli import main
from autonomy_gate.policy import APPROVE, HOLD, decide


def sdk(name):
    if os.environ.get("AUTONOMY_GATE_REQUIRE_SDKS") == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name)


REQUEST = {"id": "lead-1", "decidedAt": "2026-03-01T10:00:00Z", "text": "Moving out today."}
VERDICT = {"action": "Approve", "reason": "same_day_job", "confidence": 0.91}


class FakeAPI:
    def __init__(self, replies):
        self.replies = replies
        self.seen = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                headers = {k.lower(): v for k, v in self.headers.items()}
                api.seen.append((self.path, headers, body))
                reply = api.replies[self.path]
                status, reply = reply(body) if callable(reply) else reply
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def offline(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "OPENAI_")) or name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("AUTONOMY_GATE_ALLOW_MODEL_CALLS", "1")
    return monkeypatch


def claude_message(text, stop_reason="end_turn"):
    content = [{"type": "thinking", "thinking": "", "signature": "fake"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 150,
            "output_tokens": 40,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def test_claude_through_the_anthropic_sdk(offline):
    sdk("anthropic")
    from autonomy_gate.adapters.anthropic_messages import ClaudeMessagesAgent

    with FakeAPI({"/v1/messages": (200, claude_message(json.dumps(VERDICT)))}) as api:
        offline.setenv("ANTHROPIC_BASE_URL", api.url)
        offline.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        agent = ClaudeMessagesAgent(timeout_s=5.0)
        call = decide(agent, REQUEST)
        api.replies["/v1/messages"] = (200, claude_message(None, stop_reason="refusal"))
        refused = decide(agent, REQUEST)

    assert call.verdict.action == APPROVE and call.verdict.confidence == 0.91
    assert refused.verdict.action == HOLD and "refused" in refused.error
    path, headers, body = api.seen[0]
    assert path == "/v1/messages"
    assert "test-key-not-real" in headers.get("x-api-key", "") + headers.get("authorization", "")
    assert headers["anthropic-version"]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
    }
    assert json.loads(body["messages"][0]["content"])["decidedAt"] == REQUEST["decidedAt"]
    assert agent.usage.snapshot()["calls"] == 2
    assert agent.usage.snapshot()["input_tokens"] == 300


def openai_response(text):
    return {
        "id": "resp_fake",
        "object": "response",
        "created_at": 1790000000,
        "status": "completed",
        "model": "some-model",
        "output": [
            {
                "type": "message",
                "id": "msg_fake",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 200,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 60,
            "output_tokens_details": {"reasoning_tokens": 32},
            "total_tokens": 260,
        },
    }


def test_openai_through_the_openai_sdk(offline):
    sdk("openai")
    from autonomy_gate.adapters.openai_responses import OpenAIResponsesAgent

    verdict = json.dumps({**VERDICT, "action": "approve"})
    with FakeAPI({"/v1/responses": (200, openai_response(verdict))}) as api:
        offline.setenv("OPENAI_BASE_URL", api.url + "/v1")
        offline.setenv("OPENAI_API_KEY", "test-key-not-real")
        offline.setenv("AUTONOMY_GATE_OPENAI_MODEL", "some-model")
        offline.setenv("AUTONOMY_GATE_OPENAI_EFFORT", "low")
        agent = OpenAIResponsesAgent(timeout_s=5.0)
        call = decide(agent, REQUEST)

    assert call.verdict.action == APPROVE, call.error
    path, headers, body = api.seen[0]
    assert path == "/v1/responses"
    assert headers["authorization"] == "Bearer test-key-not-real"
    assert body["store"] is False and body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert agent.usage.snapshot()["reasoning_tokens"] == 32


def test_clock_test_runs_end_to_end_through_the_claude_sdk(offline, sets, tmp_path, capsys):
    sdk("anthropic")

    def answer(body):
        asked = json.loads(body["messages"][0]["content"])
        action = "approve" if "decidedAt" in asked else "skip"
        verdict = {"action": action, "reason": "fake", "confidence": 0.9}
        return 200, claude_message(json.dumps(verdict))

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps(r.to_dict()) + "\n" for r in sets["history"][:12]), encoding="utf-8"
    )
    out = tmp_path / "clock.json"
    with FakeAPI({"/v1/messages": answer}) as api:
        offline.setenv("ANTHROPIC_BASE_URL", api.url)
        offline.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        code = main(
            [
                "clock-test",
                str(labels),
                "--candidate",
                "autonomy_gate.adapters.anthropic_messages:ClaudeMessagesAgent",
                "--usd-per-mtok-in",
                "4",
                "--usd-per-mtok-out",
                "20",
                "--out",
                str(out),
            ]
        )
    assert code == 0, capsys.readouterr().err
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["meta"]["agent"]["model"] == "claude-opus-5-5"
    assert report["meta"]["agent"]["effort"] == "low"
    assert report["arms"]["with_clock"]["usage"]["calls"] == 12
    assert report["arms"]["without_clock"]["as_answered"]["tp"] == 0
    assert report["cost_usd"]["total"] == pytest.approx((24 * 150 * 4 + 24 * 40 * 20) / 1e6)
    assert len(api.seen) == 24
