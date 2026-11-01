"""The OpenAI adapter, driven by a fake client: no network, no API key, no openai package."""

import json
import sys
from types import SimpleNamespace

import pytest

from autonomy_gate.adapters.openai_responses import (
    VERDICT_SCHEMA,
    AdapterError,
    OpenAIResponsesAgent,
)
from autonomy_gate.policy import HOLD, decide

REQUEST = {"id": "lead-1", "decidedAt": "2026-03-01T10:00:00Z", "text": "Moving out today."}


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.responses = SimpleNamespace(create=self.create)
        self._response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def completed(text):
    return SimpleNamespace(
        status="completed",
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text")])],
        output_text=text,
    )


def test_request_uses_strict_structured_outputs_and_carries_the_clock():
    client = FakeClient(
        completed('{"action": "approve", "reason": "same_day_job", "confidence": 0.93}')
    )
    agent = OpenAIResponsesAgent("some-model", client=client, reasoning_effort="low")
    assert agent(REQUEST) == {"action": "approve", "reason": "same_day_job", "confidence": 0.93}
    kwargs = client.calls[0]
    fmt = kwargs["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "verdict"
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["required"]) == set(fmt["schema"]["properties"])
    assert json.loads(kwargs["input"])["decidedAt"] == REQUEST["decidedAt"]
    assert "decidedAt" in kwargs["instructions"]
    assert kwargs["store"] is False and kwargs["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
            output=[],
            output_text='{"action": "appr',
        ),
        SimpleNamespace(
            status="completed",
            output_text="",
            output=[
                SimpleNamespace(
                    content=[SimpleNamespace(type="refusal", refusal="I can't help with that.")]
                )
            ],
        ),
        completed("not json"),
    ],
)
def test_incomplete_refused_or_garbled_answers_become_holds(response):
    agent = OpenAIResponsesAgent("some-model", client=FakeClient(response))
    with pytest.raises(AdapterError):
        agent(REQUEST)
    call = decide(agent, REQUEST)
    assert call.verdict.action == HOLD and call.error.startswith("AdapterError")


def test_fingerprint_changes_with_the_prompt_or_model():
    a = OpenAIResponsesAgent("m1", client=FakeClient(None))
    b = OpenAIResponsesAgent("m1", "other instructions", client=FakeClient(None))
    c = OpenAIResponsesAgent("m2", client=FakeClient(None))
    assert len({a.fingerprint, b.fingerprint, c.fingerprint}) == 3


def test_model_is_required(monkeypatch):
    monkeypatch.delenv("AUTONOMY_GATE_OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError):
        OpenAIResponsesAgent(client=FakeClient(None))
    assert VERDICT_SCHEMA["properties"]["action"]["enum"] == ["approve", "hold", "skip"]


def test_no_real_client_is_built_without_the_opt_in(monkeypatch):
    monkeypatch.delenv("AUTONOMY_GATE_ALLOW_MODEL_CALLS", raising=False)
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(PermissionError, match="AUTONOMY_GATE_ALLOW_MODEL_CALLS=1"):
        OpenAIResponsesAgent("some-model")
    monkeypatch.setenv("AUTONOMY_GATE_ALLOW_MODEL_CALLS", "yes")
    with pytest.raises(PermissionError):
        OpenAIResponsesAgent("some-model")


def test_the_opt_in_builds_the_sdk_client(monkeypatch):
    built = []

    class OpenAI:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setenv("AUTONOMY_GATE_ALLOW_MODEL_CALLS", "1")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    agent = OpenAIResponsesAgent("some-model", timeout_s=12.0)
    assert isinstance(agent.client, OpenAI) and built == [{"timeout": 12.0}]


def with_usage(response, inp, out, reasoning):
    response.usage = SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    return response


def test_every_response_counts_its_tokens_even_a_refusal():
    ok = with_usage(completed('{"action": "skip", "reason": "x", "confidence": 0.5}'), 100, 20, 8)
    refused = with_usage(
        SimpleNamespace(
            status="completed",
            output_text="",
            output=[SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="no")])],
        ),
        90,
        5,
        0,
    )
    agent = OpenAIResponsesAgent("some-model", client=FakeClient(ok))
    agent(REQUEST)
    agent.client._response = refused
    with pytest.raises(AdapterError):
        agent(REQUEST)
    assert agent.usage.snapshot() == {
        "calls": 2,
        "input_tokens": 190,
        "output_tokens": 25,
        "reasoning_tokens": 8,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def test_effort_and_model_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AUTONOMY_GATE_OPENAI_MODEL", "env-model")
    monkeypatch.setenv("AUTONOMY_GATE_OPENAI_EFFORT", "low")
    agent = OpenAIResponsesAgent(client=FakeClient(None))
    assert agent.request_kwargs(REQUEST)["reasoning"] == {"effort": "low"}
    assert agent.describe() == {
        "provider": "openai-responses",
        "model": "env-model",
        "effort": "low",
        "max_output_tokens": 2000,
    }
    assert (
        agent.fingerprint
        != OpenAIResponsesAgent(
            "env-model", client=FakeClient(None), reasoning_effort="high"
        ).fingerprint
    )
