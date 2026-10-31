"""Decide when an LLM agent has earned the right to act.

Replay logged decisions through a classical baseline and the agent, with the clock in
every request. Choose a confidence threshold from human labels. Re-test it once on newer
decisions the gate has never seen. Run it in shadow until it clears written go-live bars.
Then let it act behind fail-closed checks, a kill switch and a write-ahead ledger that
never sends twice.
"""

from .lint import lint_file
from .metrics import Confusion, Interval, mcnemar_exact, wilson
from .policy import Call, Verdict, apply_rails, decide, require_clock
from .records import (
    DataError,
    Decision,
    ProvenanceError,
    Record,
    load_labels,
    load_records,
    require_human_labels,
)
from .threshold import Bars, choose_threshold

__version__ = "0.1.0"

__all__ = [
    "Bars",
    "Call",
    "Confusion",
    "DataError",
    "Decision",
    "Interval",
    "ProvenanceError",
    "Record",
    "Verdict",
    "apply_rails",
    "choose_threshold",
    "decide",
    "lint_file",
    "load_labels",
    "load_records",
    "mcnemar_exact",
    "require_clock",
    "require_human_labels",
    "wilson",
]
