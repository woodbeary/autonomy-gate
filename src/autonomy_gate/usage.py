"""Token counts per model call, summed safely across worker threads.

The model adapters add one entry per response they get back, failed or not, because a
refusal or a cut-off answer is billed too. The clock harness reads the totals before and
after each arm to report what the measurement used.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

FIELDS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


class UsageMeter:
    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._totals = dict.fromkeys(FIELDS, 0)

    def add(self, **tokens: int | None) -> None:
        unknown = set(tokens) - set(FIELDS[1:])
        if unknown:
            raise KeyError(f"unknown usage fields: {sorted(unknown)}")
        with self._mutex:
            self._totals["calls"] += 1
            for name, count in tokens.items():
                self._totals[name] += int(count or 0)

    def snapshot(self) -> dict[str, int]:
        with self._mutex:
            return dict(self._totals)


def usage_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {name: after.get(name, 0) - before.get(name, 0) for name in FIELDS}
