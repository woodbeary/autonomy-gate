import pytest

from autonomy_gate.threshold import Bars, choose_threshold, parse_flat_yaml, sweep


def test_defaults_are_the_written_bars():
    b = Bars()
    assert (b.min_shadow_days, b.min_overlaps, b.min_overlap_approvals) == (7, 40, 25)
    assert (b.min_precision, b.min_precision_lower_bound) == (0.90, 0.85)
    assert (b.max_p95_latency_s, b.max_unavailable) == (60, 0.05)
    assert (b.skip_min_precision, b.skip_min_overlaps) == (0.80, 50)


def test_picks_the_lowest_threshold_that_clears_both_precision_bars():
    items = [(0.99, True)] * 60 + [(0.7, True)] * 30 + [(0.7, False)] * 2 + [(0.4, True)] * 5
    items += [(0.4, False)] * 10
    point = choose_threshold(items, Bars())
    assert point is not None and point.threshold == 0.7
    assert point.confusion.tp == 90 and point.confusion.fp == 2


def test_small_perfect_sample_does_not_clear_the_lower_bound():
    items = [(0.9, True)] * 13 + [(0.1, False)] * 5
    assert choose_threshold(items, Bars()) is None
    c = sweep(items)[0].confusion
    assert c.precision == 1.0
    assert any("lower bound" in f for f in Bars().failures(c))


def test_ties_land_on_the_same_side():
    points = sweep([(0.5, True), (0.5, False), (0.9, True)])
    assert [p.threshold for p in points] == [0.9, 0.5]
    assert points[1].confusion.flagged == 3


def test_failures_name_each_bar():
    c = sweep([(0.9, True)] * 9 + [(0.9, False)])[0].confusion
    failures = Bars(min_recall=0.5).failures(c)
    assert "labeled decisions 10 < 40" in failures
    assert "human approvals 9 < 25" in failures
    assert any(f.startswith("Wilson 95% lower bound 0.596") for f in failures)


def test_flat_yaml_is_strict():
    assert parse_flat_yaml("# c\nmin_overlaps: 40  # forty\nmin_precision: 0.9\n") == {
        "min_overlaps": 40,
        "min_precision": 0.9,
    }
    for bad in ("bars:\n  min_overlaps: 40\n", "min_overlaps: forty\n", "a: 1\na: 2\n", "- 1\n"):
        with pytest.raises(ValueError):
            parse_flat_yaml(bad)
    with pytest.raises(ValueError, match="unknown bars"):
        Bars.from_dict({"min_overlap": 40})
