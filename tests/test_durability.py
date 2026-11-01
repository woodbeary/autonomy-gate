"""The ledger keeps a committed ``sending`` row through a power cut, not only a process crash.

A power cut loses whatever the operating system had not yet written to disk: file contents
since that file's last fsync, and directory entries since the directory's last fsync.
``powerloss_shim.c`` is loaded with ``LD_PRELOAD`` into a child process. Each time SQLite
fsyncs a file in the ledger's directory, the shim copies that file's contents aside; each
time it fsyncs the directory, the shim records which names it holds. ``PowerCut.cut()`` then
rebuilds the directory from those copies alone, which is what the disk would hold after the
power came back.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from autonomy_gate.actuator import Guard
from autonomy_gate.killswitch import StaticKillSwitch
from autonomy_gate.ledger import Ledger
from autonomy_gate.lock import GateLock
from autonomy_gate.replay import replay, run_agent
from autonomy_gate.synthetic import FileOutbox, candidate
from autonomy_gate.threshold import Bars

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
REQUIRED = os.environ.get("AUTONOMY_GATE_REQUIRE_POWERLOSS") == "1"


def _unavailable(reason: str) -> None:
    if REQUIRED:
        pytest.fail(f"the power-cut simulation is required here but {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def shim(tmp_path_factory):
    if not sys.platform.startswith("linux"):
        _unavailable("it interposes fsync with LD_PRELOAD, which needs Linux")
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        _unavailable("there is no C compiler to build the shim")
    out = tmp_path_factory.mktemp("powerloss") / "powerloss.so"
    subprocess.run(
        [cc, "-shared", "-fPIC", "-O2", "-o", str(out), str(HERE / "powerloss_shim.c"), "-ldl"],
        check=True,
    )
    return out


class PowerCut:
    def __init__(self, root: Path, shim: Path) -> None:
        self.live = root / "live"
        self.durable = root / "durable"
        self.live.mkdir(parents=True)
        self.durable.mkdir()
        self.shim = shim

    def settle(self) -> None:
        names = sorted(p.name for p in self.live.iterdir() if not p.name.endswith("-shm"))
        for name in names:
            shutil.copyfile(self.live / name, self.durable / name)
        (self.durable / ".listing").write_text("".join(n + "\n" for n in names))

    def env(self, **extra: str) -> dict[str, str]:
        return {
            "PYTHONPATH": str(SRC),
            "LD_PRELOAD": str(self.shim),
            "AG_POWERLOSS_DIR": str(self.live),
            "AG_POWERLOSS_DURABLE": str(self.durable),
            **extra,
        }

    def cut(self) -> list[str]:
        names = (self.durable / ".listing").read_text().split()
        shutil.rmtree(self.live)
        self.live.mkdir()
        for name in names:
            if name.endswith("-shm"):
                continue
            kept = self.durable / name
            if kept.exists():
                shutil.copyfile(kept, self.live / name)
            else:
                (self.live / name).touch()
        return sorted(p.name for p in self.live.iterdir())


def run_child(script: str, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


RAW_COMMIT = textwrap.dedent(
    """
    import os, sqlite3, sys
    db = sqlite3.connect(sys.argv[1], isolation_level=None)
    for pragma in sys.argv[2:]:
        db.execute(pragma)
    db.execute("BEGIN IMMEDIATE")
    db.execute("INSERT INTO items VALUES ('x', 'sending')")
    db.execute("COMMIT")
    os._exit(137)
    """
)


@pytest.mark.parametrize(
    ("pragmas", "survives"),
    [
        (["PRAGMA synchronous=FULL"], False),
        (["PRAGMA journal_mode=WAL", "PRAGMA synchronous=NORMAL"], False),
        (["PRAGMA journal_mode=WAL", "PRAGMA synchronous=FULL"], True),
    ],
)
def test_the_simulated_cut_loses_exactly_what_sqlite_says_it_may(tmp_path, shim, pragmas, survives):
    power = PowerCut(tmp_path, shim)
    db = power.live / "ledger.db"
    setup = sqlite3.connect(db, isolation_level=None)
    for pragma in pragmas:
        setup.execute(pragma)
    setup.execute("CREATE TABLE items (id TEXT PRIMARY KEY, state TEXT)")
    setup.close()
    power.settle()
    proc = run_child(RAW_COMMIT, power.env(), str(db), *pragmas)
    assert proc.returncode == 137, proc.stderr
    power.cut()
    after = sqlite3.connect(db)
    rows = after.execute("SELECT * FROM items").fetchall()
    after.close()
    assert rows == ([("x", "sending")] if survives else [])


def test_the_ledger_sets_wal_full_and_fullfsync(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        db = ledger._db
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert db.execute("PRAGMA synchronous").fetchone() == (2,)
        assert db.execute("PRAGMA fullfsync").fetchone() == (1,)
        assert db.execute("PRAGMA checkpoint_fullfsync").fetchone() == (1,)


LEDGER_COMMIT = textwrap.dedent(
    """
    import os, sys
    from autonomy_gate.ledger import Ledger
    ledger = Ledger(sys.argv[1])
    ledger.record("lead-1", decided_at="2026-03-01T09:00:00Z", action="approve", reason="ok",
                  confidence=0.99, state="sending", fingerprint="fp")
    os._exit(137)
    """
)


def test_a_committed_sending_row_survives_a_power_cut(tmp_path, shim):
    power = PowerCut(tmp_path, shim)
    db = power.live / "ledger.db"
    Ledger(db).close()
    power.settle()
    proc = run_child(LEDGER_COMMIT, power.env(), str(db))
    assert proc.returncode == 137, proc.stderr
    power.cut()
    with Ledger(db) as ledger:
        row = ledger.get("lead-1")
    assert row is not None and row.state == "sending"


CRASH_IN_SEND = textwrap.dedent(
    """
    import sys
    from autonomy_gate.actuator import Guard
    from autonomy_gate.killswitch import StaticKillSwitch
    from autonomy_gate.ledger import Ledger
    from autonomy_gate.lock import GateLock
    from autonomy_gate.records import load_records
    from autonomy_gate.synthetic import FileOutbox

    db, lock_path, outbox, crash, requests = sys.argv[1:6]
    guard = Guard(lambda r: {"action": "approve", "reason": "ok", "confidence": 0.99},
                  Ledger(db), StaticKillSwitch(True), GateLock.load(lock_path),
                  fingerprint="fp", sender=FileOutbox(outbox, crash=crash))
    guard.cycle(load_records(requests))
    """
)


class Counting(FileOutbox):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.sends: list[str] = []
        self.lookups: list[str] = []

    def send(self, request_id, payload):
        self.sends.append(request_id)
        return super().send(request_id, payload)

    def lookup(self, request_id):
        self.lookups.append(request_id)
        return super().lookup(request_id)


@pytest.mark.parametrize(
    ("crash", "expected", "lines"),
    [("before_write", "unconfirmed_never_resubmitted", 0), ("after_write", "sent", 1)],
)
def test_a_power_cut_inside_send_is_settled_by_lookup_never_resent(
    sets, tmp_path, shim, crash, expected, lines
):
    history = sets["history"]
    lock = GateLock.from_replay(
        history, replay(history, run_agent(candidate, history), Bars()), fingerprint="fp"
    )
    lock.retest = {"passed": True}
    lock.golive = {"passed": True, "skips_earned": False}
    lock_path = tmp_path / "gate.lock"
    lock.save(lock_path)
    record = sets["live"][0]
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    outbox = tmp_path / "outbox.jsonl"

    power = PowerCut(tmp_path / "disk", shim)
    db = power.live / "ledger.db"
    Ledger(db).close()
    power.settle()
    proc = run_child(
        CRASH_IN_SEND, power.env(), str(db), str(lock_path), str(outbox), crash, str(requests)
    )
    assert proc.returncode == 137, proc.stderr
    power.cut()

    sender = Counting(outbox)
    with Ledger(db) as ledger:
        assert ledger.get(record.id).state == "sending"
        guard = Guard(
            lambda r: {"action": "approve", "reason": "ok", "confidence": 0.99},
            ledger,
            StaticKillSwitch(True),
            lock,
            fingerprint="fp",
            sender=sender,
        )
        report = guard.cycle([record])
        guard.cycle([record])
        assert ledger.get(record.id).state == expected
    assert report.outcomes[0][:2] == (record.id, expected)
    assert sender.sends == []
    assert len(sender.lookups) == 1
    written = outbox.read_text(encoding="utf-8").splitlines() if outbox.exists() else []
    assert len(written) == lines
