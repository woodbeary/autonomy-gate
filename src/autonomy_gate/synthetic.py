"""Synthetic pickup requests, a stand-in model and a file outbox, so everything runs offline.

Nothing here is real data or a real model. Requests come from templates and a seeded RNG,
and the "person" deciding them is a formula. :func:`candidate` stands in for an LLM: it
reads the text with keyword rules, misreads a few, and adds noise. Its numbers say nothing
about any real model; they exist so the demo and the tests run without an API key.

The task: a hauling business gets pickup requests. Should the agent reply with a quote now
(approve), leave it for a person (hold), or close it (skip)? About a quarter of requests
are same-day ("moving out today, need it gone before 5pm"). Whether one is still worth
answering depends on when it is being decided, so a model that isn't told ``decidedAt``
has to guess what today is. The stand-in guesses the way a model with no date in its
prompt does: from its own fixed idea of today. It then calls fresh same-day requests
expired, and recall collapses on exactly those.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .records import HUMAN, Decision, Record, iso, parse_ts, write_jsonl

SEED = 3
START = datetime(2026, 2, 2, 7, 0, tzinfo=UTC)
ASSUMED_TODAY = datetime(2026, 6, 30, 12, 0, tzinfo=UTC)
SIZES = {"history": 500, "heldout": 300, "live": 300}
LIVE_LABEL_SHARE = 0.6

ITEMS = [
    "a couch",
    "two dressers",
    "a mattress and box spring",
    "an old fridge",
    "a broken treadmill",
    "a sectional",
    "a washer and dryer",
    "a pile of yard waste",
    "a desk and an office chair",
    "a hot tub",
]

CATEGORIES: dict[str, tuple[float, float, list[str]]] = {
    "same_day": (
        0.24,
        2.6,
        [
            "Moving out today, need {item} gone before {hour}pm.",
            "Garage sale ends at {hour} today. Want whatever's left hauled off after, "
            "mostly {item}.",
            "Landlord walkthrough at {hour}pm today. Need {item} out before then.",
        ],
    ),
    "scheduled": (
        0.20,
        1.8,
        [
            "Looking for someone to clear out a garage next week, about a truck load.",
            "Need a quote for a backyard cleanup, branches and old fencing, any day this week.",
            "Clearing out a storage unit later this month. Mostly boxes and {item}.",
        ],
    ),
    "single_item": (
        0.18,
        1.3,
        [
            "Need {item} picked up, can pay.",
            "Have {item} to get rid of. Who hauls?",
        ],
    ),
    "price_question": (
        0.12,
        1.0,
        [
            "How much to haul away {item}?",
            "Ballpark for hauling {item}?",
        ],
    ),
    "giveaway": (
        0.14,
        -3.5,
        [
            "Free {item} on the curb, first come first serve.",
            "Giving away {item}, you pick up.",
        ],
    ),
    "question": (
        0.12,
        -3.8,
        [
            "Anyone know where the dump is and if they take paint?",
            "What day is bulk trash pickup on our street?",
        ],
    ),
}
EXPIRED = -3.5
SHARPNESS = 2.5
NOISE = 0.8
OPERATIONAL_SHARE = 0.03
REASONS = {
    "same_day": "same_day_job",
    "scheduled": "scheduled_job",
    "single_item": "single_item",
    "price_question": "price_question",
    "giveaway": "not_a_job",
    "question": "not_a_job",
}
_HOUR = re.compile(r"\b(\d{1,2})(?:\s?pm\b|\s+today\b)", re.IGNORECASE)


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _deadline_hour(text: str) -> int | None:
    m = _HOUR.search(text)
    return int(m.group(1)) + 12 if m else None


def expired(text: str, posted_at: datetime | None, now: datetime) -> bool:
    """A same-day request is expired once its day or its deadline has passed."""
    hour = _deadline_hour(text)
    if posted_at is None or hour is None:
        return True
    return now.date() != posted_at.date() or now.hour >= hour


def _log_odds(category: str, is_expired: bool, f: Mapping[str, Any]) -> float:
    """How a person decides, on average. The generator and the stand-in share it."""
    z = EXPIRED if category == "same_day" and is_expired else CATEGORIES[category][1]
    z += 0.45 * bool(f["has_photo"]) - 0.10 * (f["distance_miles"] - 7)
    z += -0.30 * f["replies"] + 0.08 * (f["items"] - 3)
    return SHARPNESS * z


def generate(n: int, *, seed: int = SEED, start: datetime = START) -> list[Record]:
    """``n`` requests decided by a person, about 30 a day, sorted by ``decidedAt``."""
    rng = random.Random(seed)
    names = list(CATEGORIES)
    weights = [CATEGORIES[c][0] for c in names]
    posted = start
    out: list[Record] = []
    for i in range(n):
        posted = posted + timedelta(minutes=rng.randint(20, 75))
        category = rng.choices(names, weights)[0]
        if category == "same_day":
            posted_at = posted.replace(hour=rng.randint(7, 12), minute=rng.randint(0, 59))
            late = rng.random() < 0.15
            delay = (
                timedelta(hours=rng.uniform(8, 30))
                if late
                else timedelta(minutes=rng.randint(5, 150))
            )
        else:
            posted_at = posted
            delay = timedelta(minutes=rng.randint(5, 240))
        decided_at = posted_at + delay
        hours = delay.total_seconds() / 3600
        features = {
            "distance_miles": round(min(rng.lognormvariate(1.9, 0.6), 45.0), 1),
            "has_photo": rng.random() < 0.45,
            "items": rng.randint(1, 3)
            if category in ("giveaway", "question")
            else rng.randint(1, 8),
            "replies": min(9, int(rng.expovariate(1 / (0.4 + 0.6 * hours)))),
        }
        text = rng.choice(CATEGORIES[category][2]).format(
            item=rng.choice(ITEMS), hour=rng.randint(3, 7)
        )
        is_expired = category == "same_day" and expired(text, posted_at, decided_at)
        approved = rng.random() < _sigmoid(_log_odds(category, is_expired, features))
        basis = "content"
        if approved and rng.random() < OPERATIONAL_SHARE:
            approved, basis = False, "operational"
        reason = "take_it" if approved else ("schedule_full" if basis == "operational" else "pass")
        out.append(
            Record(
                id=f"lead-{i + 1:05d}",
                decided_at=decided_at,
                captured_at=decided_at,
                request={"text": text, "postedAt": iso(posted_at), "features": features},
                decision=Decision("approve" if approved else "skip", HUMAN, basis, reason),
            )
        )
    return sorted(out, key=lambda r: (r.decided_at, r.id))


_KEYWORDS = [
    ("same_day", ("today",)),
    ("giveaway", ("free ", "giving away")),
    ("question", ("anyone know", "what day")),
    ("price_question", ("how much", "ballpark")),
    ("scheduled", ("next week", "this week", "later this month")),
    ("single_item", ("picked up", "get rid of")),
]


def _read_category(text: str) -> str:
    lowered = text.lower()
    for category, words in _KEYWORDS:
        if any(w in lowered for w in words):
            return category
    return "single_item"


def candidate(request: Mapping[str, Any]) -> dict[str, Any]:
    """The stand-in model. Deterministic per request id, so a replay is repeatable."""
    digest = hashlib.sha256(str(request.get("id", "")).encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    text = str(request.get("text", ""))
    category = _read_category(text)
    if rng.random() < 0.04:
        category = rng.choice(list(CATEGORIES))
    now = parse_ts(request["decidedAt"]) if request.get("decidedAt") else ASSUMED_TODAY
    posted = parse_ts(request["postedAt"]) if request.get("postedAt") else None
    is_expired = category == "same_day" and expired(text, posted, now)
    p = _sigmoid(_log_odds(category, is_expired, request["features"]) + rng.gauss(0, NOISE))
    reason = "expired" if is_expired else REASONS[category]
    if p >= 0.55:
        return {"action": "approve", "reason": reason, "confidence": round(p, 4)}
    if p <= 0.25:
        return {"action": "skip", "reason": reason, "confidence": round(1 - p, 4)}
    return {"action": "hold", "reason": "unsure", "confidence": round(max(p, 1 - p), 4)}


candidate.fingerprint = "synthetic-stand-in/1"  # type: ignore[attr-defined]


class FileOutbox:
    """A sender that appends to a JSONL file and can look a request id up.

    ``crash="before_write"`` or ``"after_write"`` ends the process mid-send with
    ``os._exit``, the way a power cut or ``kill -9`` would, to show what the runtime does
    with an answer it never got.
    """

    def __init__(self, path: str | Path, *, crash: str | None = None) -> None:
        if crash not in (None, "before_write", "after_write"):
            raise ValueError("crash must be None, 'before_write' or 'after_write'")
        self.path = Path(path)
        self.crash = crash

    def lookup(self, request_id: str) -> str | None:
        if not self.path.exists():
            return None
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["request_id"] == request_id:
                    return row["receipt"]
        return None

    def send(self, request_id: str, payload: Mapping[str, Any]) -> str:
        if self.crash == "before_write":
            os._exit(137)
        existing = self.lookup(request_id)
        if existing:
            return existing
        receipt = "rcpt_" + hashlib.sha256(request_id.encode()).hexdigest()[:12]
        row = {"request_id": request_id, "receipt": receipt, "payload": dict(payload)}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if self.crash == "after_write":
            os._exit(137)
        return receipt


GATE_YAML = """\
# Go-live bars. `name: number` lines only.
min_shadow_days: 7
min_overlaps: 40
min_overlap_approvals: 25
min_precision: 0.90
min_precision_lower_bound: 0.85
# Precision alone can pass an agent that misses half the work. Pick a recall floor.
min_recall: 0.70
max_p95_latency_s: 60
max_unavailable: 0.05
skip_min_precision: 0.80
skip_min_overlaps: 50
confidence: 0.95
"""


def example_sets(seed: int = SEED) -> dict[str, list[Record]]:
    """Three consecutive periods, and the person's decisions on part of the last one."""
    stream = generate(sum(SIZES.values()), seed=seed)
    sets: dict[str, list[Record]] = {}
    first = 0
    for name, size in SIZES.items():
        last = min(first + size, len(stream))
        while 0 < last < len(stream) and stream[last].decided_at == stream[last - 1].decided_at:
            last += 1
        sets[name] = stream[first:last]
        first = last
    rng = random.Random(seed + 1)
    labeled = [r for r in sets["live"] if rng.random() < LIVE_LABEL_SHARE]
    sets["live_labels"] = labeled
    sets["live"] = [r.without_decision() for r in sets["live"]]
    return sets


def write_examples(directory: str | Path, seed: int = SEED) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, records in example_sets(seed).items():
        path = directory / f"leads_{name}.jsonl"
        write_jsonl(path, (r.to_dict() for r in records))
        written.append(path)
    gate = directory / "gate.yaml"
    gate.write_text(GATE_YAML, encoding="utf-8")
    written.append(gate)
    return written
