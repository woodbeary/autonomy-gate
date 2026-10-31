from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from autonomy_gate.records import HUMAN, Decision, Record
from autonomy_gate.synthetic import example_sets


@pytest.fixture(scope="session")
def sets():
    return example_sets()


def make_record(
    i,
    *,
    label=None,
    by=HUMAN,
    text="Need a couch picked up, can pay.",
    at=None,
    basis="content",
    features=None,
):
    at = at or datetime(2026, 3, 1, 9, tzinfo=UTC) + timedelta(minutes=10 * i)
    decision = None
    if label is not None:
        decision = Decision("approve" if label else "skip", by, basis)
    return Record(
        id=f"r-{i:04d}",
        decided_at=at,
        captured_at=at,
        request={"text": text, "features": features or {"x": float(i % 7)}},
        decision=decision,
    )


@pytest.fixture
def record():
    return make_record
