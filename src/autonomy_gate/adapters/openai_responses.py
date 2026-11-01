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
``--candidate autonomy_gate.adapters.openai_responses:OpenAIResponsesAgent``, the model in
``AUTONOMY_GATE_OPENAI_MODEL``, optionally the reasoning effort in
``AUTONOMY_GATE_OPENAI_EFFORT`` and your own instructions in the file named by
``AUTONOMY_GATE_INSTRUCTIONS_FILE``. It is off by default: it builds a real client only when
``AUTONOMY_GATE_ALLOW_MODEL_CALLS=1``. It sends ``store=False``, so the response is not
stored for later retrieval through the API. Every response's token usage is added to
``agent.usage``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from ..lock import fingerprint
from ..usage import UsageMeter
from .common import (
    DEFAULT_INSTRUCTIONS,
    VERDICT_SCHEMA,
    AdapterError,
    instructions_from_env,
    require_model_calls,
    verdict_from_text,
)

__all__ = ["DEFAULT_INSTRUCTIONS", "VERDICT_SCHEMA", "AdapterError", "OpenAIResponsesAgent"]


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
        instructions = instructions_from_env(instructions)
        reasoning_effort = reasoning_effort or os.environ.get("AUTONOMY_GATE_OPENAI_EFFORT")
        if client is None:
            require_model_calls("OpenAI Responses API")
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
        self.usage = UsageMeter()
        self.fingerprint = fingerprint(
            "openai-responses", model, instructions, VERDICT_SCHEMA, reasoning_effort
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "openai-responses",
            "model": self.model,
            "effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
        }

    def _count(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        details = getattr(usage, "output_tokens_details", None)
        self.usage.add(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            reasoning_tokens=getattr(details, "reasoning_tokens", 0),
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
        self._count(response)
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
        return verdict_from_text(getattr(response, "output_text", None))
