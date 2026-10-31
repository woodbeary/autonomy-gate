import pytest

from autonomy_gate.metrics import Confusion, confusion_at, mcnemar_exact, p95, wilson


def test_nine_of_ten_is_not_ninety_percent():
    ci = wilson(9, 10)
    assert ci.low == pytest.approx(0.5958, abs=1e-3)
    assert ci.high == pytest.approx(0.9821, abs=1e-3)


def test_wilson_known_value_and_edges():
    assert wilson(75, 87).low == pytest.approx(0.7745, abs=1e-3)
    assert wilson(0, 0).low == 0.0 and wilson(0, 0).high == 1.0
    assert wilson(0, 5).low == 0.0
    assert wilson(5, 5).high == 1.0
    with pytest.raises(ValueError):
        wilson(6, 5)


def test_lower_bound_grows_with_evidence():
    assert wilson(9, 10).low < wilson(90, 100).low < wilson(900, 1000).low


def test_confusion_at_never_acts_without_a_score():
    c = confusion_at([(0.9, True), (None, True), (0.2, False), (0.95, False)], 0.5)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    closed = confusion_at([(0.9, True), (0.8, False)], None)
    assert closed.flagged == 0 and closed.precision is None


def test_precision_and_recall_counts():
    c = Confusion(tp=75, fp=12, fn=63, tn=50)
    assert c.precision == pytest.approx(75 / 87)
    assert c.recall == pytest.approx(75 / 138)


def test_p95_nearest_rank():
    assert p95([]) is None
    assert p95([float(i) for i in range(1, 101)]) == 95.0
    assert p95([3.0]) == 3.0


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
