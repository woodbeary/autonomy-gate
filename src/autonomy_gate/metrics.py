"""Counts, precision and recall with Wilson intervals, and an exact McNemar test.

Everything is reported as whole counts first ("75 of 87") and rates second, because a rate
without its denominator can't be checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def __str__(self) -> str:
        return f"[{self.low:.3f}, {self.high:.3f}]"


def wilson(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Unlike the textbook p +/- z*sqrt(p(1-p)/n) interval it stays inside [0, 1] and keeps
    its coverage near 0 and 1, which is where a gate's precision lives (Brown, Cai and
    DasGupta, "Interval Estimation for a Binomial Proportion", Statistical Science, 2001).
    With n == 0 nothing is known, so the interval is [0, 1].
    """
    if not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes} of {n}")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n == 0:
        return Interval(0.0, 1.0)
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return Interval(low, high)


@dataclass(frozen=True)
class Confusion:
    """Outcomes at one threshold. "Flagged" means the gate would let the agent act."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def flagged(self) -> int:
        return self.tp + self.fp

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float | None:
        return self.tp / self.flagged if self.flagged else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.positives if self.positives else None

    def precision_interval(self, confidence: float = 0.95) -> Interval:
        return wilson(self.tp, self.flagged, confidence)

    def recall_interval(self, confidence: float = 0.95) -> Interval:
        return wilson(self.tp, self.positives, confidence)

    def to_dict(self, confidence: float = 0.95) -> dict:
        p_ci = self.precision_interval(confidence)
        r_ci = self.recall_interval(confidence)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "precision_ci": [p_ci.low, p_ci.high],
            "recall": self.recall,
            "recall_ci": [r_ci.low, r_ci.high],
            "confidence": confidence,
        }


def confusion_at(items: list[tuple[float | None, bool]], threshold: float | None) -> Confusion:
    """Count outcomes when the agent acts on ``score >= threshold``.

    A missing score (the agent didn't approve, or failed) never acts, and neither does
    anything when ``threshold`` is None (no threshold cleared the bar).
    """
    tp = fp = fn = tn = 0
    for score, label in items:
        acts = threshold is not None and score is not None and score >= threshold
        if acts and label:
            tp += 1
        elif acts:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    return Confusion(tp, fp, fn, tn)


def p95(values: list[float]) -> float | None:
    """Nearest-rank 95th percentile: the smallest value at or above 95% of the sample."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for paired outcomes.

    ``only_a`` and ``only_b`` count the cases one system got right and the other got wrong.
    Under the null hypothesis each disagreement is a fair coin flip.
    """
    if only_a < 0 or only_b < 0:
        raise ValueError("counts must be non-negative")
    m = only_a + only_b
    if m == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2**m
    return min(1.0, 2 * tail)
