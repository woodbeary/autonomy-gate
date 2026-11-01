"""Policy and shadow code must not be able to reach the network or anything that sends."""

import sys

import pytest

from autonomy_gate.isolation import IsolationError, assert_isolated, reachable

SENDERS = (
    "autonomy_gate.actuator",
    "autonomy_gate.ledger",
    "autonomy_gate.adapters",
    "autonomy_gate.killswitch",
    "autonomy_gate.cli",
)


@pytest.mark.parametrize("module", ["autonomy_gate.policy", "autonomy_gate.shadow"])
def test_policy_and_shadow_cannot_reach_network_or_send_code(module):
    assert_isolated(module, also=SENDERS)


def test_the_actuator_is_the_one_that_can_send():
    assert "autonomy_gate.ledger" in reachable("autonomy_gate.actuator")
    with pytest.raises(IsolationError):
        assert_isolated("autonomy_gate.actuator", also=("autonomy_gate.ledger",))


def test_the_walker_catches_an_indirect_import(tmp_path, monkeypatch):
    pkg = tmp_path / "leaky"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "policy.py").write_text("from . import helpers\n", encoding="utf-8")
    (pkg / "helpers.py").write_text(
        "def fetch():\n    import urllib.request\n    return urllib.request\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(IsolationError, match=r"leaky.policy -> leaky.helpers -> urllib.request"):
        assert_isolated("leaky.policy")
    sys.modules.pop("leaky", None)


def test_no_model_adapter_is_reachable_from_the_package_root():
    assert not any(m.startswith("autonomy_gate.adapters") for m in reachable("autonomy_gate"))
    assert not any(m in ("openai", "anthropic") for m in reachable("autonomy_gate"))


@pytest.mark.parametrize(
    "module",
    ["autonomy_gate.adapters.openai_responses", "autonomy_gate.adapters.anthropic_messages"],
)
def test_each_adapter_reaches_the_actuator_through_nothing(module):
    assert_isolated(module, forbidden=(), also=SENDERS[:2])
