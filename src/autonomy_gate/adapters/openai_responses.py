"""An agent backed by the OpenAI Responses API with Structured Outputs. Optional.

Install with ``pip install "autonomy-gate[openai]"``. Nothing else in the package imports
this module, and the tests drive it with a fake client, so neither the ``openai`` package
nor an API key is needed to run them.

The request follows OpenAI's docs as of October 2026:

* Structured Outputs on the Responses API: the schema goes in ``text.format`` with
  ``type: "json_schema"`` and ``strict: true``; every property is listed in ``required``
  and objects set ``additionalProperties: false``. A refusal comes back as a ``refusal``
  content item instead of output text, and a response cut off by ``max_output_tokens``
  has ``status: "incomplete"``. Both are raised here, so the gate turns them into holds.
  https://developers.openai.com/api/docs/guides/structured-outputs
* Create a model response (``POST /v1/responses``):
  https://developers.openai.com/api/reference/resources/responses/methods/create

Use it from the command line with
``--candidate autonomy_gate.adapters.openai_responses:OpenAIResponsesAgent`` and the
model in ``AUTONOMY_GATE_OPENAI_MODEL`` (and, optionally, your own instructions in the
file named by ``AUTONOMY_GATE_INSTRUCTIONS_FILE``). It sends ``store=False``, so the
response is not stored for later retrieval through the API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..lock import fingerprint

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["approve", "hold", "skip"],
            "description": "approve: act now; hold: leave it for a person; skip: close it",
        },
        "reason": {
            "type": "string",
            "description": "a short snake_case reason code, such as same_day_job or expired",
        },
        "confidence": {
            "type": "number",
            "description": "0 to 1: how likely it is that a person would make the same call",
        },
    },
    "required": ["action", "reason", "confidence"],
    "additionalProperties": False,
}

DEFAULT_INSTRUCTIONS = """\
You decide what happens to one incoming request. Answer with an action:
- approve: it is worth acting on now;
- hold: you are unsure, or it needs a person;
- skip: it is not worth acting on.
The request includes decidedAt, the moment of this decision. Judge every time-dependent
phrase ("today", "before 5pm", "this weekend") against decidedAt, never against your own
idea of the date. Give a short reason code and your confidence that a person would agree.
"""


class AdapterError(RuntimeError):
    """The API answered with something that isn't a verdict."""


class OpenAIResponsesAgent:
    """Callable agent: ``agent(request) -> {"action", "reason", "confidence"}``."""

    def __init__(
        self,
        model: str | None = None,
        instructions: str | None = None,
        *,
        client: Any = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 2000,
        timeout_s: float = 60.0,
    ) -> None:
        model = model or os.environ.get("AUTONOMY_GATE_OPENAI_MODEL")
        if not model:
            raise ValueError("pass model= or set AUTONOMY_GATE_OPENAI_MODEL")
        if instructions is None:
            path = os.environ.get("AUTONOMY_GATE_INSTRUCTIONS_FILE")
            instructions = Path(path).read_text(encoding="utf-8") if path else DEFAULT_INSTRUCTIONS
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError('install the extra: pip install "autonomy-gate[openai]"') from exc
            client = OpenAI(timeout=timeout_s)
        self.client = client
        self.model = model
        self.instructions = instructions
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.fingerprint = fingerprint(
            "openai-responses", model, instructions, VERDICT_SCHEMA, reasoning_effort
        )

    def request_kwargs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            "input": json.dumps(dict(request), sort_keys=True, ensure_ascii=False),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "verdict",
                    "strict": True,
                    "schema": VERDICT_SCHEMA,
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return kwargs

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self.client.responses.create(**self.request_kwargs(request))
        status = getattr(response, "status", None)
        if status is not None and status != "completed":
            details = getattr(response, "incomplete_details", None) or getattr(
                response, "error", None
            )
            raise AdapterError(f"response status {status}: {details}")
        for item in getattr(response, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise AdapterError(f"model refused: {getattr(part, 'refusal', '')}")
        text = getattr(response, "output_text", None)
        if not text:
            raise AdapterError("no output text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AdapterError("output is not JSON") from exc
        if not isinstance(data, dict):
            raise AdapterError("output is not a JSON object")
        return data
