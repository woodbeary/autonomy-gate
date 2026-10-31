from autonomy_gate.golive import check, format_check
from autonomy_gate.shadow import Shadow, load_shadow
from autonomy_gate.synthetic import candidate
from autonomy_gate.threshold import Bars


def shadow_log(sets, tmp_path, days=None, agent=candidate):
    live = sets["live"]
    if days is not None:
        start = live[0].decided_at
        live = [r for r in live if (r.decided_at - start).total_seconds() < days * 86400]
    log = tmp_path / "shadow.jsonl"
    Shadow(agent, log, fingerprint="f", threshold=0.6).run(live)
    return load_shadow(log)


def test_red_names_every_failing_bar_then_green_after_a_full_week(sets, tmp_path):
    early = check(shadow_log(sets, tmp_path, days=1), sets["live_labels"], Bars())
    assert not early.passed
    failing = {r.name for r in early.failing}
    assert {"days of shadow", "overlaps with a person", "Wilson 95% lower bound"} <= failing
    text = format_check(early)
    assert text.startswith("RED:") and "FAIL  overlaps with a person" in text

    full = check(shadow_log(sets, tmp_path), sets["live_labels"], Bars(min_recall=0.7))
    assert full.passed, format_check(full)
    assert format_check(full).startswith("GREEN")


def test_skips_stay_advisory_until_they_earn_it(sets, tmp_path):
    report = check(shadow_log(sets, tmp_path, days=1), sets["live_labels"], Bars())
    assert not report.skips_earned and "Skips: advisory" in format_check(report)
    full = check(shadow_log(sets, tmp_path), sets["live_labels"], Bars())
    assert full.skip_overlaps >= 50 and full.skips_earned


def test_an_unavailable_model_fails_its_bar(sets, tmp_path):
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] % 10 == 0:
            raise ConnectionError("503")
        return candidate(request)

    report = check(shadow_log(sets, tmp_path, agent=flaky), sets["live_labels"], Bars())
    row = next(r for r in report.rows if r.name == "model unavailable")
    assert not row.ok and row.value.startswith("30/300")


def test_slow_decisions_fail_the_latency_bar(sets, tmp_path):
    entries = shadow_log(sets, tmp_path)
    for e in entries[:20]:
        e["latency_s"] = 90.0
    report = check(entries, sets["live_labels"], Bars())
    assert not next(r for r in report.rows if r.name == "p95 decision time").ok
