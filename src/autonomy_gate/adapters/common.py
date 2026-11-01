"""What every model adapter shares: the verdict schema, the default instructions, and the
rule that no adapter calls a model API unless ``AUTONOMY_GATE_ALLOW_MODEL_CALLS=1``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ALLOW_MODEL_CALLS = "AUTONOMY_GATE_ALLOW_MODEL_CALLS"

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


class ModelCallsOff(PermissionError):
    """Model API calls are off until the person running the gate turns them on."""


def require_model_calls(provider: str) -> None:
    if os.environ.get(ALLOW_MODEL_CALLS) != "1":
        raise ModelCallsOff(
            f"{provider} calls are off by default. Set {ALLOW_MODEL_CALLS}=1 to let this "
            "adapter call the API with your key; every call is billed to it."
        )


def instructions_from_env(instructions: str | None) -> str:
    if instructions is not None:
        return instructions
    path = os.environ.get("AUTONOMY_GATE_INSTRUCTIONS_FILE")
    return Path(path).read_text(encoding="utf-8") if path else DEFAULT_INSTRUCTIONS


def verdict_from_text(text: str | None) -> dict[str, Any]:
    if not text:
        raise AdapterError("no output text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterError("output is not JSON") from exc
    if not isinstance(data, dict):
        raise AdapterError("output is not a JSON object")
    return data
