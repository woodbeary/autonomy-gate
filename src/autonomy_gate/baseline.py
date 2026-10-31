"""A classical baseline to beat before an LLM gets to act.

If a logistic regression on the numbers you already log does as well as the model, the
model hasn't earned anything. Fit it on older decisions and compare on newer ones.

It is deliberately plain: numeric and boolean features, standardized, L2-regularized, fit
by full-batch gradient descent. It is deterministic, so a replay gives the same baseline
every time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .records import Record


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticBaseline:
    """L2-regularized logistic regression on a decision's numeric features.

    ``baseline(record) -> float`` in [0, 1]. It reads only ``request.features``: numeric and
    boolean values count, everything else is ignored, and a missing feature counts as the
    training mean. It fits on human labels only.
    """

    def __init__(self, l2: float = 1.0, learning_rate: float = 0.5, epochs: int = 400) -> None:
        if l2 < 0 or learning_rate <= 0 or epochs < 1:
            raise ValueError("need l2 >= 0, learning_rate > 0 and epochs >= 1")
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.keys: list[str] = []
        self.means: list[float] = []
        self.scales: list[float] = []
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, records: Sequence[Record]) -> LogisticBaseline:
        labeled = [d for d in records if d.label is not None]
        if len(labeled) < 2 or len({d.label for d in labeled}) < 2:
            raise ValueError("fit needs labeled decisions of both classes")
        keys = sorted(
            {k for d in labeled for k, v in d.features.items() if _numeric(v) is not None}
        )
        if not keys:
            raise ValueError("no numeric or boolean features to fit on")
        self.keys = keys
        raw = [[_numeric(d.features.get(k)) for k in keys] for d in labeled]
        self.means, self.scales = [], []
        for j in range(len(keys)):
            col = [row[j] for row in raw if row[j] is not None]
            mean = sum(col) / len(col)
            var = sum((x - mean) ** 2 for x in col) / len(col)
            self.means.append(mean)
            self.scales.append(math.sqrt(var) or 1.0)
        xs = [self._standardize(row) for row in raw]
        ys = [1.0 if d.label else 0.0 for d in labeled]
        n = len(xs)
        w = [0.0] * len(keys)
        b = math.log((sum(ys) + 0.5) / (n - sum(ys) + 0.5))
        for _ in range(self.epochs):
            gw = [0.0] * len(keys)
            gb = 0.0
            for x, y in zip(xs, ys, strict=True):
                err = _sigmoid(b + sum(wj * xj for wj, xj in zip(w, x, strict=True))) - y
                gb += err
                for j, xj in enumerate(x):
                    gw[j] += err * xj
            for j in range(len(w)):
                w[j] -= self.learning_rate * (gw[j] / n + self.l2 * w[j] / n)
            b -= self.learning_rate * gb / n
        self.weights, self.bias = w, b
        return self

    def _standardize(self, row: Sequence[float | None]) -> list[float]:
        return [
            0.0 if v is None else (v - m) / s
            for v, m, s in zip(row, self.means, self.scales, strict=True)
        ]

    def score_features(self, features: Mapping[str, Any]) -> float:
        if not self.keys:
            raise RuntimeError("call fit() first")
        x = self._standardize([_numeric(features.get(k)) for k in self.keys])
        return _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, x, strict=True)))

    def __call__(self, record: Record) -> float:
        return self.score_features(record.features)
