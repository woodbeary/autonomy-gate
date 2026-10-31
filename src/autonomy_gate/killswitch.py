"""Kill switches. Each one fails closed: if its state can't be read, the agent stops.

A switch says "run" only when it positively reads "run". A missing file, an unreadable
file or any other content means "stop". The runtime reads it at the start of every cycle
and again right before every send, so turning it off stops all actions on the next check,
with nothing to unwind.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

RUN = "run"
STOP = "stop"


@dataclass(frozen=True)
class SwitchState:
    running: bool
    detail: str


class KillSwitch(Protocol):
    def state(self) -> SwitchState: ...


class StaticKillSwitch:
    """An in-memory switch, for tests and for wiring your own source of truth."""

    def __init__(self, running: bool = False) -> None:
        self._running = running
        self._lock = threading.Lock()

    def state(self) -> SwitchState:
        with self._lock:
            return SwitchState(self._running, RUN if self._running else STOP)

    def set(self, running: bool) -> None:
        with self._lock:
            self._running = running


class FileKillSwitch:
    """A file that must contain ``run`` for the agent to act. Off by default.

    Write ``stop`` to it, or delete it, to stop the agent. :meth:`set` writes atomically.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def state(self) -> SwitchState:
        try:
            value = self.path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return SwitchState(False, f"{self.path} is missing")
        except (OSError, UnicodeDecodeError) as exc:
            return SwitchState(False, f"can't read {self.path}: {type(exc).__name__}")
        if value == RUN:
            return SwitchState(True, RUN)
        return SwitchState(False, f"{self.path} says {value[:40]!r}")

    def set(self, running: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".switch-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write((RUN if running else STOP) + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
