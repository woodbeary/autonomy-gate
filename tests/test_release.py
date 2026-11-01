"""The version, the changelog and the command line agree."""

import re
from pathlib import Path

import pytest

import autonomy_gate
from autonomy_gate.cli import main

ROOT = Path(__file__).resolve().parent.parent


def test_the_changelog_leads_with_this_version(capsys):
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    newest = re.search(r"^## (\S+)", changelog, flags=re.MULTILINE).group(1)
    assert newest == autonomy_gate.__version__
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == f"autonomy-gate {autonomy_gate.__version__}"
