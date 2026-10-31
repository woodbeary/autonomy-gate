import pytest

from autonomy_gate.lock import DataReuseError, GateLock, LockError
from autonomy_gate.records import Record
from autonomy_gate.replay import replay, run_agent
from autonomy_gate.synthetic import candidate
from autonomy_gate.threshold import Bars


def tuned(sets, tmp_path):
    history = sets["history"]
    report = replay(history, run_agent(candidate, history), Bars())
    lock = GateLock.from_replay(history, report, fingerprint="fp")
    lock.save(tmp_path / "gate.lock")
    return GateLock.load(tmp_path / "gate.lock")


def test_retest_refuses_tuning_data(sets, tmp_path):
    lock = tuned(sets, tmp_path)
    with pytest.raises(DataReuseError, match="ids were already used"):
        lock.check_fresh(sets["history"][:10])


def test_retest_refuses_the_same_request_under_a_new_id(sets, tmp_path):
    lock = tuned(sets, tmp_path)
    old = sets["history"][0]
    renamed = Record("brand-new-id", old.decided_at, old.request, old.captured_at, old.decision)
    with pytest.raises(DataReuseError, match="repeat requests already used"):
        lock.check_fresh([renamed], allow_older=True)


def test_retest_refuses_older_decisions_unless_allowed(sets, tmp_path):
    lock = tuned(sets, tmp_path)
    old = sets["history"][-1]
    fresh_id = Record("other", old.decided_at, {"text": "different"}, None, old.decision)
    with pytest.raises(DataReuseError, match="not newer than"):
        lock.check_fresh([fresh_id])
    lock.check_fresh([fresh_id], allow_older=True)


def test_one_held_out_result_per_lock(sets, tmp_path):
    lock = tuned(sets, tmp_path)
    heldout = sets["heldout"]
    result = lock.run_retest(heldout, run_agent(candidate, heldout))
    assert result["passed"], result["failures"]
    assert result["n"] == len(heldout)
    with pytest.raises(LockError, match="already holds a held-out result"):
        lock.run_retest(heldout, run_agent(candidate, heldout))


def test_a_new_lock_inherits_everything_the_old_one_saw(sets, tmp_path):
    lock = tuned(sets, tmp_path)
    heldout = sets["heldout"]
    lock.run_retest(heldout, run_agent(candidate, heldout))
    live = sets["live_labels"]
    report = replay(live, run_agent(candidate, live), Bars())
    relocked = GateLock.from_replay(live, report, fingerprint="fp2", previous=lock)
    for spent in (sets["history"][:3], heldout[:3]):
        with pytest.raises(DataReuseError):
            relocked.check_fresh(spent, allow_older=True)


def test_threshold_is_frozen(sets, tmp_path):
    lock = tuned(sets, tmp_path)
    before = lock.threshold
    lock.run_retest(sets["heldout"], run_agent(candidate, sets["heldout"]))
    assert GateLock.load(tmp_path / "gate.lock").threshold == before == lock.threshold
