"""The write-ahead ledger: one row per request, written before anything is sent.

States:

* ``held``: left for a person (model error, below threshold, a rail, the kill switch, or
  live mode off).
* ``advisory_skip``: the agent would close it, but skips haven't earned that yet, so a
  person decides, with the suggestion attached.
* ``skipped``: closed without acting, once skips have earned it.
* ``would_approve``: approved but over the send cap. Sent on a later cycle from the stored
  verdict, without asking the model again.
* ``sending``: the intent row, committed before the sender is called, with the request id
  the sender uses as its idempotency key.
* ``sent``: the sender returned a receipt, or a lookup by request id found one.
* ``unconfirmed_never_resubmitted``: the answer was lost (crash, timeout, error) and a
  lookup couldn't confirm it. It is never sent again; a person checks it.

A request id is derived from the request's own id and is unique in the table, so each
request can reach ``sending`` at most once, ever.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATES = (
    "held",
    "advisory_skip",
    "skipped",
    "would_approve",
    "sending",
    "sent",
    "unconfirmed_never_resubmitted",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  item_id      TEXT PRIMARY KEY,
  decided_at   TEXT NOT NULL,
  decided_by   TEXT NOT NULL DEFAULT 'agent',
  action       TEXT NOT NULL,
  reason       TEXT NOT NULL,
  confidence   REAL,
  state        TEXT NOT NULL,
  request_id   TEXT UNIQUE,
  receipt      TEXT,
  detail       TEXT,
  payload      TEXT,
  fingerprint  TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  seq      INTEGER PRIMARY KEY AUTOINCREMENT,
  at       TEXT NOT NULL,
  item_id  TEXT NOT NULL,
  state    TEXT NOT NULL,
  detail   TEXT
);
"""


class LedgerError(Exception):
    """A transition the ledger refuses, such as sending a request twice."""


def request_id_for(item_id: str) -> str:
    return "ag_" + hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:24]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Row:
    item_id: str
    decided_at: str
    decided_by: str
    action: str
    reason: str
    confidence: float | None
    state: str
    request_id: str | None
    receipt: str | None
    detail: str | None
    payload: Any
    fingerprint: str
    created_at: str
    updated_at: str


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._db = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _event(self, item_id: str, state: str, detail: str | None) -> None:
        self._db.execute(
            "INSERT INTO events (at, item_id, state, detail) VALUES (?, ?, ?, ?)",
            (_now(), item_id, state, detail),
        )

    def get(self, item_id: str) -> Row | None:
        cur = self._db.execute("SELECT * FROM items WHERE item_id = ?", (item_id,))
        row = cur.fetchone()
        return None if row is None else self._row(cur, row)

    def has(self, item_id: str) -> bool:
        return self.get(item_id) is not None

    def in_state(self, state: str) -> list[Row]:
        cur = self._db.execute(
            "SELECT * FROM items WHERE state = ? ORDER BY decided_at, item_id", (state,)
        )
        return [self._row(cur, r) for r in cur.fetchall()]

    def rows(self) -> list[Row]:
        cur = self._db.execute("SELECT * FROM items ORDER BY decided_at, item_id")
        return [self._row(cur, r) for r in cur.fetchall()]

    def counts(self) -> dict[str, int]:
        cur = self._db.execute("SELECT state, COUNT(*) FROM items GROUP BY state")
        return dict(cur.fetchall())

    def events(self, item_id: str) -> list[tuple[str, str, str | None]]:
        cur = self._db.execute(
            "SELECT at, state, detail FROM events WHERE item_id = ? ORDER BY seq", (item_id,)
        )
        return list(cur.fetchall())

    @staticmethod
    def _row(cur: sqlite3.Cursor, values: tuple) -> Row:
        data = dict(zip([d[0] for d in cur.description], values, strict=True))
        data["payload"] = None if data["payload"] is None else json.loads(data["payload"])
        return Row(**data)

    def record(
        self,
        item_id: str,
        *,
        decided_at: str,
        action: str,
        reason: str,
        confidence: float | None,
        state: str,
        fingerprint: str,
        payload: Mapping[str, Any] | None = None,
        detail: str | None = None,
    ) -> str | None:
        """Insert a new request in ``state``. ``sending`` also assigns its request id.

        Commits before returning, so a ``sending`` row is durable before the sender runs.
        """
        if state not in STATES:
            raise LedgerError(f"unknown state {state!r}")
        request_id = request_id_for(item_id) if state == "sending" else None
        now = _now()
        try:
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute(
                "INSERT INTO items (item_id, decided_at, action, reason, confidence, state, "
                "request_id, detail, payload, fingerprint, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    decided_at,
                    action,
                    reason,
                    confidence,
                    state,
                    request_id,
                    detail,
                    None if payload is None else json.dumps(payload, sort_keys=True),
                    fingerprint,
                    now,
                    now,
                ),
            )
            self._event(item_id, state, detail)
            self._db.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            self._db.execute("ROLLBACK")
            raise LedgerError(f"{item_id} is already in the ledger") from exc
        return request_id

    def begin_send(self, item_id: str) -> str:
        """Move a deferred approval to ``sending`` and return its request id."""
        request_id = request_id_for(item_id)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            cur = self._db.execute(
                "UPDATE items SET state = 'sending', request_id = ?, updated_at = ? "
                "WHERE item_id = ? AND state = 'would_approve' AND request_id IS NULL",
                (request_id, _now(), item_id),
            )
            if cur.rowcount != 1:
                raise LedgerError(f"{item_id} is not a deferred approval that was never sent")
            self._event(item_id, "sending", None)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return request_id

    def move(
        self,
        item_id: str,
        from_state: str,
        to_state: str,
        detail: str | None = None,
        *,
        receipt: str | None = None,
    ) -> None:
        """Change state if the row is still in ``from_state``. Leaving ``sending`` or a
        settled state for anything that could send again is refused."""
        if to_state not in STATES:
            raise LedgerError(f"unknown state {to_state!r}")
        allowed = {
            "would_approve": {"held", "sending"},
            "sending": {"sent", "unconfirmed_never_resubmitted"},
        }
        if to_state not in allowed.get(from_state, set()):
            raise LedgerError(f"{from_state} -> {to_state} is not allowed")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            cur = self._db.execute(
                "UPDATE items SET state = ?, detail = ?, receipt = COALESCE(?, receipt), "
                "updated_at = ? WHERE item_id = ? AND state = ?",
                (to_state, detail, receipt, _now(), item_id, from_state),
            )
            if cur.rowcount != 1:
                raise LedgerError(f"{item_id} is not in state {from_state}")
            self._event(item_id, to_state, detail)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
