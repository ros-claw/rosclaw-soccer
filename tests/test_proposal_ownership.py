from dataclasses import FrozenInstanceError, replace

import pytest

from rosclaw_soccer.diagnostics.proposal_ownership import (
    ProposalActivity,
    ProposalOwnershipSnapshot,
)

CONTEXT = "sha256:" + "a" * 64
TARGET = "sha256:" + "b" * 64


def snapshot():
    return ProposalOwnershipSnapshot(
        "blue.playmaker",
        428,
        CONTEXT,
        (ProposalActivity("capture", 0, True), ProposalActivity("navigation", 1, False)),
        TARGET,
    )


def test_terminal_receiver_and_one_navigation_owner():
    item = snapshot()
    assert item.selected_owner_id == "navigation"
    assert item.binding_hash == replace(item).binding_hash
    assert item.binding_hash == replace(item, activities=item.activities[::-1]).binding_hash
    with pytest.raises(FrozenInstanceError):
        item.frame = 0


def test_no_target_is_explicit_and_does_not_claim_fallback_execution():
    item = ProposalOwnershipSnapshot(
        "sensor.worker", 0, CONTEXT, (ProposalActivity("tracking", 0, False),), None
    )
    assert item.selected_owner_id is None
    assert item.binding_hash.startswith("sha256:")


def test_two_inferences_cannot_masquerade_as_one_selected_target():
    with pytest.raises(ValueError, match="multiple"):
        replace(
            snapshot(),
            activities=(ProposalActivity("capture", 1, False), ProposalActivity("nav", 1, False)),
        )


@pytest.mark.parametrize("count", [-1, 2, True, 1.0, None])
def test_invalid_counts(count):
    with pytest.raises(ValueError):
        ProposalActivity("capture", count, False)


@pytest.mark.parametrize("terminal", [None, 0, 1, "false"])
def test_terminal_requires_boolean(terminal):
    with pytest.raises(ValueError):
        ProposalActivity("capture", 0, terminal)


def test_terminal_owner_cannot_propose():
    with pytest.raises(ValueError, match="nonterminal"):
        ProposalActivity("capture", 1, True)


@pytest.mark.parametrize("identifier", ["", "a\n", "A", "a" * 129, 123, "../a"])
def test_invalid_owner_names(identifier):
    with pytest.raises(ValueError):
        ProposalActivity(identifier, 0, False)


@pytest.mark.parametrize(
    "changes",
    [
        {"subject_id": "../actor"},
        {"frame": -1},
        {"frame": True},
        {"frame": 1.0},
        {"frame": 2**31},
        {"context_hash": "sha256:abcd"},
        {"context_hash": "sha256:" + "A" * 64},
        {"context_hash": None},
        {"returned_target_hash": "bad"},
        {"returned_target_hash": None},
        {"activities": []},
        {"activities": ()},
        {"activities": ({"owner_id": "capture"},)},
        {"activities": tuple(ProposalActivity(f"a{i}", 0, False) for i in range(17))},
        {"activities": (ProposalActivity("same", 0, False), ProposalActivity("same", 0, True))},
        {"activities": (ProposalActivity("none", 0, False),)},
    ],
)
def test_invalid_snapshots_fail_closed(changes):
    with pytest.raises(ValueError):
        replace(snapshot(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"subject_id": "another.agent"},
        {"frame": 429},
        {"context_hash": TARGET},
        {"returned_target_hash": CONTEXT},
        {"activities": (ProposalActivity("other-owner", 1, False),)},
    ],
)
def test_binding_changes_with_each_material_field(changes):
    assert replace(snapshot(), **changes).binding_hash != snapshot().binding_hash


def test_no_training_simulator_or_core_dependency_in_fresh_process():
    import os
    import subprocess
    import sys

    code = """
import sys
from rosclaw_soccer.diagnostics.proposal_ownership import ProposalOwnershipSnapshot
for name in sys.modules:
    assert name.split('.')[0] not in {'torch','mujoco','rosclaw','onnxruntime'}
"""
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)
