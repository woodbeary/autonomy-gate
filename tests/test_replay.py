from autonomy_gate.baseline import LogisticBaseline
from autonomy_gate.replay import format_report, replay, run_agent, stratified_sample, time_split
from autonomy_gate.synthetic import candidate
from autonomy_gate.threshold import Bars

BARS = Bars(min_recall=0.70)


def test_without_a_clock_recall_collapses_on_same_day_requests(sets):
    history = sets["history"]
    with_clock = replay(history, run_agent(candidate, history), BARS)
    without = replay(history, run_agent(candidate, history, clock=False), BARS, clock=False)
    assert with_clock.as_answered.recall > 0.9
    assert without.as_answered.recall < 0.7
    assert without.as_answered.precision > 0.9
    expired = next(r for r in without.by_reason if r.reason == "expired")
    assert expired.skip == expired.n and expired.human_approved > 100
    assert with_clock.candidate.meets_bars
    assert not without.candidate.meets_bars
    assert "recall 0.647 < 0.70" in without.candidate.failures
    assert "CLOCK STRIPPED" in format_report(without)


def test_baseline_is_fitted_on_older_decisions_and_compared_at_equal_precision(sets):
    train, test = time_split(sets["history"], 0.5)
    assert max(r.decided_at for r in train) <= min(r.decided_at for r in test)
    baseline = LogisticBaseline().fit(train)
    report = replay(test, run_agent(candidate, test), BARS, baseline=baseline)
    m = report.matched
    assert m is not None and m.baseline is not None
    assert m.baseline.confusion.recall < report.candidate.choice.confusion.recall
    assert m.candidate_only > m.baseline_only and m.p_value < 1e-6
    text = format_report(report)
    assert "Baseline held to the candidate's precision" in text
    assert "Worst false approvals" in text


def test_content_only_labels_are_reported_separately(sets):
    history = sets["history"]
    report = replay(history, run_agent(candidate, history), BARS)
    assert any(r.decision.basis == "operational" for r in history)
    assert report.content_only.precision > report.candidate.choice.confusion.precision


def test_model_errors_count_as_holds_and_never_approve(record):
    records = [record(i, label=True) for i in range(50)]

    def flaky(request):
        if request["id"].endswith("7"):
            raise TimeoutError("slow")
        return {"action": "approve", "reason": "ok", "confidence": 0.99}

    report = replay(records, run_agent(flaky, records), Bars())
    assert report.candidate.errors == 5
    assert report.matrix[("hold", "approve")] == 5
    assert report.candidate.choice.confusion.tp == 45


def test_stratified_sample_keeps_the_mix(sets):
    history = sets["history"]
    sample = stratified_sample(history, 200, seed=1)
    share = sum(r.label for r in history) / len(history)
    assert len(sample) == 200
    assert abs(sum(r.label for r in sample) / 200 - share) < 0.01
