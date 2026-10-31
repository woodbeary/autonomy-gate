"""The bars an agent has to clear, and choosing a threshold from labeled examples.

The defaults are the go-live bars the author wrote for his own lead agent before it was
allowed to act: at least 7 days of shadow, at least 40 overlaps with a person including 25
approvals, precision of at least 0.90 whose Wilson 95% lower bound is at least 0.85, p95
decision time of at most 60 s, the model unavailable on at most 5% of calls, and skips
left advisory until they reach 0.80 precision over 50 overlaps.

A confidence is anything ordered. The gate never trusts it as a probability: it sorts the
labeled examples by confidence and picks the lowest threshold whose precision clears the
bar, which gives the most recall the bar allows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .metrics import Confusion


@dataclass(frozen=True)
class Bars:
    min_shadow_days: float = 7.0
    min_overlaps: int = 40
    min_overlap_approvals: int = 25
    min_precision: float = 0.90
    min_precision_lower_bound: float = 0.85
    min_recall: float = 0.0
    max_p95_latency_s: float = 60.0
    max_unavailable: float = 0.05
    skip_min_precision: float = 0.80
    skip_min_overlaps: int = 50
    confidence: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "min_precision",
            "min_precision_lower_bound",
            "min_recall",
            "max_unavailable",
            "skip_min_precision",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        for name in ("min_overlaps", "min_overlap_approvals", "skip_min_overlaps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.min_shadow_days < 0 or self.max_p95_latency_s <= 0:
            raise ValueError("min_shadow_days must be >= 0 and max_p95_latency_s > 0")

    def precision_ok(self, c: Confusion) -> bool:
        if c.flagged == 0 or c.precision is None:
            return False
        low = c.precision_interval(self.confidence).low
        return c.precision >= self.min_precision and low >= self.min_precision_lower_bound

    def failures(self, c: Confusion) -> list[str]:
        """Why ``c`` misses the bar on a labeled set. Empty means it clears it."""
        out: list[str] = []
        if c.n < self.min_overlaps:
            out.append(f"labeled decisions {c.n} < {self.min_overlaps}")
        if c.positives < self.min_overlap_approvals:
            out.append(f"human approvals {c.positives} < {self.min_overlap_approvals}")
        if c.precision is None:
            out.append("the agent approved nothing, so precision is unknown")
        else:
            if c.precision < self.min_precision:
                out.append(f"precision {c.precision:.3f} < {self.min_precision:.2f}")
            low = c.precision_interval(self.confidence).low
            if low < self.min_precision_lower_bound:
                out.append(
                    f"Wilson {self.confidence:.0%} lower bound {low:.3f} "
                    f"< {self.min_precision_lower_bound:.2f}"
                )
        recall = c.recall or 0.0
        if recall < self.min_recall:
            out.append(f"recall {recall:.3f} < {self.min_recall:.2f}")
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict) -> Bars:
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(obj) - set(known))
        if unknown:
            raise ValueError(f"unknown bars: {', '.join(unknown)}")
        out = {}
        for key, value in obj.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{key} must be a number")
            out[key] = int(value) if known[key] == "int" and float(value).is_integer() else value
        return cls(**out)

    @classmethod
    def load(cls, path: str | Path) -> Bars:
        return cls.from_dict(parse_flat_yaml(Path(path).read_text(encoding="utf-8")))


_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^#]*?)\s*(#.*)?$")


def parse_flat_yaml(text: str) -> dict[str, float]:
    """Parse the one YAML shape the bars file uses: ``name: number`` lines and comments.

    Anything else (nesting, lists, strings) is an error rather than a guess, so a typo
    can't silently drop a bar.
    """
    out: dict[str, float] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line == "---":
            continue
        m = _LINE.match(line)
        if not m or raw[:1].isspace():
            raise ValueError(f"line {lineno}: expected `name: number`, got {raw!r}")
        key, value = m.group(1), m.group(2)
        if key in out:
            raise ValueError(f"line {lineno}: {key} is set twice")
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {key} must be a number, got {value!r}") from exc
        out[key] = int(number) if re.fullmatch(r"[+-]?\d+", value) else number
    return out


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    confusion: Confusion


def sweep(items: Sequence[tuple[float | None, bool]]) -> list[OperatingPoint]:
    """Every distinct threshold, from the highest confidence down, with its counts.

    Items with equal confidence always land on the same side of a threshold. Items with
    no confidence (the agent didn't approve, or failed) never act.
    """
    scored = sorted(((s, y) for s, y in items if s is not None), key=lambda t: -t[0])
    positives = sum(1 for _, y in items if y)
    negatives = len(items) - positives
    points: list[OperatingPoint] = []
    tp = fp = 0
    i = 0
    while i < len(scored):
        threshold = scored[i][0]
        while i < len(scored) and scored[i][0] == threshold:
            if scored[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append(OperatingPoint(threshold, Confusion(tp, fp, positives - tp, negatives - fp)))
    return points


def choose_threshold(
    items: Sequence[tuple[float | None, bool]], bars: Bars
) -> OperatingPoint | None:
    """The lowest threshold whose precision and Wilson lower bound clear ``bars``.

    Lower thresholds only add recall, so this is the most recall the bars allow. None
    means no threshold clears them, and the gate stays closed.
    """
    best: OperatingPoint | None = None
    for point in sweep(items):
        if bars.precision_ok(point.confusion):
            best = point
    return best


def threshold_for_precision(
    items: Sequence[tuple[float | None, bool]], target: float
) -> OperatingPoint | None:
    """The lowest threshold whose observed precision is at least ``target``.

    Used to compare two scorers fairly: hold one at the other's precision and compare how
    much each catches.
    """
    best: OperatingPoint | None = None
    for point in sweep(items):
        p = point.confusion.precision
        if p is not None and p >= target - 1e-12:
            best = point
    return best
