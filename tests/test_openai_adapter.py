"""The OpenAI adapter, driven by a fake client: no network, no API key, no openai package."""

import json
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
