import subprocess
import sys
from dataclasses import asdict, replace

import pytest
from test_receiving_feedback import Provider, locomotion, observation, schedule

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackSlot
from rosclaw_soccer.training.receiving_scene import ReceivingPeerState, ReceivingSceneContext


def scene():
    return ReceivingSceneContext(
        0,
        "blue.finisher",
        "sha256:" + "c" * 64,
        "receive",
        (ReceivingPeerState("red.defender", (1.0, 2.0), (0.0, 0.0)),),
        None,
        None,
        False,
        False,
        False,
        0.12,
    )


def test_memory_only_hash_is_unchanged():
    obs = replace(observation(), locomotion=locomotion())
    expected = asdict(obs)
    expected["locomotion"].pop("scene")
    expected["locomotion"]["memory"] = obs.locomotion.memory.state_hash
    assert obs.observation_hash == hash_json(expected)


def test_contract_can_be_imported_before_team_package():
    result = subprocess.run(
        [sys.executable, "-c", "import rosclaw_soccer.training.receiving_feedback"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_scene_is_opt_in_bound_to_current_agent_and_clock():
    old = replace(observation(), locomotion=locomotion())
    new = replace(old, locomotion=replace(locomotion(), scene=scene()))
    assert new.observation_hash != old.observation_hash
    source = schedule()
    provider = Provider(source)
    provider.requires_locomotion_memory = provider.requires_navigation_context = True
    assert ReceivingFeedbackSlot(provider, source).step(new) == provider.result
    slot = ReceivingFeedbackSlot(provider, source)
    with pytest.raises(ValueError):
        slot.step(old)
    assert slot.faulted
    for changes in ({"frame": 1}, {"agent_id": "blue.defender"}):
        with pytest.raises(ValueError):
            replace(old, locomotion=replace(locomotion(), scene=replace(scene(), **changes)))


@pytest.mark.parametrize(
    "changes",
    [
        {"frame": True},
        {"frame": 1000},
        {"intent": ""},
        {"world_config_hash": "unbound"},
        {"peers": []},
        {"peers": (scene().peers[0], scene().peers[0])},
        {"peers": (ReceivingPeerState("blue.finisher", (0.0, 0.0), (0.0, 0.0)),)},
        {"possession_agent_id": "unknown"},
        {"receive_lease_active": True},
        {"receive_lease_active": 1},
        {"navigation_overrides_present": 0},
        {"post_receive_hold": 1},
        {"receive_foot_lateral_offset_m": float("nan")},
        {"receive_foot_lateral_offset_m": -0.1},
        {"receive_foot_lateral_offset_m": True},
    ],
)
def test_invalid_scene(changes):
    with pytest.raises(ValueError):
        replace(scene(), **changes)


@pytest.mark.parametrize(
    "values",
    [[0.0, 0.0], (float("nan"), 0.0), (0.0, float("inf")), (True, 0.0), (0.0,), (1e5, 0.0)],
)
def test_invalid_peer(values):
    with pytest.raises(ValueError):
        ReceivingPeerState("red.defender", values, (0.0, 0.0))


@pytest.mark.parametrize("memory, navigation", [(False, True), (True, 1), (True, None)])
def test_invalid_requirement(memory, navigation):
    source = schedule()
    provider = Provider(source)
    provider.requires_locomotion_memory = memory
    provider.requires_navigation_context = navigation
    with pytest.raises(ValueError):
        ReceivingFeedbackSlot(provider, source)


def test_requirement_cannot_change_in_callback():
    source = schedule()
    provider = Provider(source)
    provider.requires_locomotion_memory = provider.requires_navigation_context = True

    def mutate(obs):
        provider.requires_navigation_context = False
        return provider.result

    provider.propose = mutate
    slot = ReceivingFeedbackSlot(provider, source)
    with pytest.raises(ValueError):
        slot.step(replace(observation(), locomotion=replace(locomotion(), scene=scene())))
    assert slot.faulted
