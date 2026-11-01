"""The Claude adapter, driven by a fake client: no network, no API key, no anthropic package."""

import json
import sys
from types import SimpleNamespace

import pytest

from autonomy_gate.adapters.anthropic_messages import DEFAULT_MODEL, ClaudeMessagesAgent
from autonomy_gate.adapters.common import VERDICT_SCHEMA, AdapterError
from autonomy_gate.adapters.openai_responses import OpenAIResponsesAgent
from autonomy_gate.policy import APPROVE, HOLD, decide

REQUEST = {"id": "lead-1", "decidedAt": "2026-03-01T10:00:00Z", "text": "Moving out today."}


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.messages = SimpleNamespace(create=self.create)
        self.response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def message(text=None, stop_reason="end_turn", usage=(120, 30), stop_details=None):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(
            input_tokens=usage[0],
            output_tokens=usage[1],
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


VERDICT = '{"action": "approve", "reason": "same_day_job", "confidence": 0.93}'


def test_request_uses_output_config_with_the_schema_and_an_explicit_effort(monkeypatch):
    monkeypatch.delenv("AUTONOMY_GATE_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("AUTONOMY_GATE_ANTHROPIC_EFFORT", raising=False)
    client = FakeClient(message(VERDICT))
    agent = ClaudeMessagesAgent(client=client)
    assert agent(REQUEST) == {"action": "approve", "reason": "same_day_job", "confidence": 0.93}
    kwargs = client.calls[0]
    assert kwargs["model"] == DEFAULT_MODEL == "claude-opus-5-5"
    assert kwargs["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
    }
    assert "decidedAt" in kwargs["system"]
    [turn] = kwargs["messages"]
    assert turn["role"] == "user" and json.loads(turn["content"]) == REQUEST
    assert kwargs["max_tokens"] >= 1024
    assert not {"output_format", "temperature", "top_p", "top_k", "fallbacks", "thinking"} & set(
        kwargs
    )


def test_the_schema_has_no_numeric_constraints_and_bounds_are_checked_locally():
    def walk(node):
        if isinstance(node, dict):
            assert not {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"} & set(node)
            for value in node.values():
                walk(value)

    walk(VERDICT_SCHEMA)
    agent = ClaudeMessagesAgent(
        client=FakeClient(message('{"action": "approve", "reason": "x", "confidence": 1.7}'))
    )
    call = decide(agent, REQUEST)
    assert call.verdict.action == HOLD and "outside [0, 1]" in call.error


def test_the_action_is_lowercased_before_the_verdict_is_checked():
    agent = ClaudeMessagesAgent(
        client=FakeClient(message('{"action": "Approve", "reason": "x", "confidence": 0.9}'))
    )
    assert decide(agent, REQUEST).verdict.action == APPROVE


@pytest.mark.parametrize(
    "response",
    [
        message(
            None,
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber", explanation=""),
        ),
        message('{"action": "appr', stop_reason="max_tokens"),
        message(VERDICT, stop_reason="tool_use"),
        message(None),
        message("not json"),
        message("[1, 2]"),
    ],
)
def test_refused_cut_off_or_garbled_answers_become_holds(response):
    agent = ClaudeMessagesAgent(client=FakeClient(response))
    with pytest.raises(AdapterError):
        agent(REQUEST)
    call = decide(agent, REQUEST)
    assert call.verdict.action == HOLD and call.error.startswith("AdapterError")


def test_every_response_counts_its_tokens_even_a_refusal():
    client = FakeClient(message(VERDICT, usage=(100, 40)))
    agent = ClaudeMessagesAgent(client=client)
    agent(REQUEST)
    client.response = message(None, stop_reason="refusal", usage=(90, 2))
    decide(agent, REQUEST)
    assert agent.usage.snapshot() == {
        "calls": 2,
        "input_tokens": 190,
        "output_tokens": 42,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def test_no_real_client_is_built_without_the_opt_in(monkeypatch):
    monkeypatch.delenv("AUTONOMY_GATE_ALLOW_MODEL_CALLS", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(PermissionError, match="AUTONOMY_GATE_ALLOW_MODEL_CALLS=1"):
        ClaudeMessagesAgent()


def test_the_opt_in_builds_the_sdk_client(monkeypatch):
    built = []

    class Anthropic:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setenv("AUTONOMY_GATE_ALLOW_MODEL_CALLS", "1")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Anthropic))
    agent = ClaudeMessagesAgent(timeout_s=30.0)
    assert isinstance(agent.client, Anthropic) and built == [{"timeout": 30.0}]


def test_model_and_effort_come_from_the_environment_and_change_the_fingerprint(monkeypatch):
    monkeypatch.setenv("AUTONOMY_GATE_ANTHROPIC_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("AUTONOMY_GATE_ANTHROPIC_EFFORT", "medium")
    agent = ClaudeMessagesAgent(client=FakeClient(None))
    assert agent.describe() == {
        "provider": "anthropic-messages",
        "model": "claude-sonnet-5-5",
        "effort": "medium",
        "max_tokens": agent.max_tokens,
    }
    others = {
        ClaudeMessagesAgent("claude-opus-5-5", client=FakeClient(None)).fingerprint,
        ClaudeMessagesAgent(client=FakeClient(None), effort="high").fingerprint,
        ClaudeMessagesAgent(client=FakeClient(None), instructions="other").fingerprint,
        OpenAIResponsesAgent("claude-sonnet-5-5", client=FakeClient(None)).fingerprint,
    }
    assert agent.fingerprint not in others and len(others) == 4
