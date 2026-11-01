"""An agent backed by Claude through the Anthropic Messages API, with structured outputs. Optional.

Install with ``pip install "autonomy-gate[anthropic]"``. Nothing else in the package imports
this module, and the tests drive it with a fake client, so neither the ``anthropic`` package
nor an API key is needed to run them. It is off by default: it builds a real client only
when ``AUTONOMY_GATE_ALLOW_MODEL_CALLS=1``.

The request follows Anthropic's structured outputs docs as of October 2026
(https://platform.claude.com/docs/en/build-with-claude/structured-outputs):

* The schema goes in ``output_config.format`` with ``type: "json_schema"``; the older
  ``output_format`` parameter is deprecated. Numeric constraints such as ``minimum`` aren't
  supported, so the schema has none and :class:`~autonomy_gate.policy.Verdict` checks that
  confidence is in [0, 1].
* A refusal comes back with HTTP 200 and ``stop_reason: "refusal"``, and an answer cut off
  by ``max_tokens`` has ``stop_reason: "max_tokens"``. Both raise here, as does any stop
  other than ``end_turn``, so the gate turns them into holds.
* Enum values aren't guaranteed to keep their capitalization, so ``action`` is lowercased
  before the verdict is checked.
* The effort level goes in ``output_config.effort`` and is always sent, because the default
  differs between models.

It sends no fallback models: a refusal stays a hold, because an answer from a model the lock
never tested must not act. Every response's token usage is added to ``agent.usage``.

Use it from the command line with
``--candidate autonomy_gate.adapters.anthropic_messages:ClaudeMessagesAgent``, the model in
``AUTONOMY_GATE_ANTHROPIC_MODEL`` (``claude-opus-5-5`` if unset), the effort in
``AUTONOMY_GATE_ANTHROPIC_EFFORT`` (``low`` if unset), and your own instructions in the file
named by ``AUTONOMY_GATE_INSTRUCTIONS_FILE``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from ..lock import fingerprint
from ..usage import UsageMeter
from .common import (
    VERDICT_SCHEMA,
    AdapterError,
    instructions_from_env,
    require_model_calls,
    verdict_from_text,
)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"


class ClaudeMessagesAgent:
    """Callable agent: ``agent(request) -> {"action", "reason", "confidence"}``."""

    def __init__(
        self,
        model: str | None = None,
        instructions: str | None = None,
        *,
        client: Any = None,
        effort: str | None = None,
        max_tokens: int = 16000,
        timeout_s: float = 60.0,
    ) -> None:
        model = model or os.environ.get("AUTONOMY_GATE_ANTHROPIC_MODEL") or DEFAULT_MODEL
        instructions = instructions_from_env(instructions)
        effort = effort or os.environ.get("AUTONOMY_GATE_ANTHROPIC_EFFORT") or DEFAULT_EFFORT
        if client is None:
            require_model_calls("Anthropic Messages API")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError(
                    'install the extra: pip install "autonomy-gate[anthropic]"'
                ) from exc
            client = Anthropic(timeout=timeout_s)
        self.client = client
        self.model = model
        self.instructions = instructions
        self.effort = effort
        self.max_tokens = max_tokens
        self.usage = UsageMeter()
        self.fingerprint = fingerprint(
            "anthropic-messages", model, instructions, VERDICT_SCHEMA, effort
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "anthropic-messages",
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
        }

    def request_kwargs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.instructions,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(dict(request), sort_keys=True, ensure_ascii=False),
                }
            ],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
        }

    def _count(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.usage.add(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
        )

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self.client.messages.create(**self.request_kwargs(request))
        self._count(response)
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            raise AdapterError(f"model refused: {getattr(details, 'category', None)}")
        if stop == "max_tokens":
            raise AdapterError(f"answer cut off at max_tokens={self.max_tokens}")
        if stop != "end_turn":
            raise AdapterError(f"stop_reason {stop}")
        text = next(
            (b.text for b in getattr(response, "content", None) or [] if b.type == "text"),
            None,
        )
        data = verdict_from_text(text)
        if isinstance(data.get("action"), str):
            data["action"] = data["action"].lower()
        return data
