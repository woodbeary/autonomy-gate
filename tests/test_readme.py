"""The README documents commands and settings that exist."""

import re
import shlex
from pathlib import Path

import pytest

from autonomy_gate.cli import build_parser
from autonomy_gate.ledger import DURABILITY

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def documented_commands():
    blocks = re.findall(r"```(?:sh|text)\n(.*?)```", README, flags=re.DOTALL)
    joined = [b.replace("\\\n", " ") for b in blocks]
    r_value = re.search(r'R="([^"]+)"', README)
    commands = []
    for block in joined:
        for line in block.splitlines():
            line = re.sub(r"\s+#.*$", "", line.strip().removeprefix("$ "))
            line = re.sub(r"^(?:[A-Z_]+=\S+\s+)+", "", line)
            if not line.startswith("autonomy-gate "):
                continue
            if r_value:
                line = line.replace("$R", r_value.group(1))
            line = re.sub(r"<[^>]+>", "1", line)
            commands.append(line)
    return commands


COMMANDS = documented_commands()


def test_the_readme_documents_every_subcommand():
    used = {shlex.split(c)[1] for c in COMMANDS}
    assert {"replay", "retest", "shadow", "check", "run", "switch", "clock-test"} <= used


@pytest.mark.parametrize("command", COMMANDS)
def test_each_documented_command_parses(command):
    try:
        build_parser().parse_args(shlex.split(command)[1:])
    except SystemExit as exc:
        pytest.fail(f"README command does not parse ({exc.code}): {command}")


def test_the_readme_names_the_durability_settings_the_ledger_uses():
    assert "WAL" in README
    for pragma in DURABILITY:
        name = pragma.split("=")[0]
        assert name in README, name
