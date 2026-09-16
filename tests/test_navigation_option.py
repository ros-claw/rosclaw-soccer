from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.navigation_option import (
    NavigationDelta,
    NavigationObservation,
    NavigationSlot,
)


def observation(frame=0):
    return NavigationObservation(
        "blue.playmaker",
        frame,
        frame * 0.02,
        "playmaker",
        "pass",
        (0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.4, 0.1, 0.115),
        (0.1, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (0.2, 0.1),
        (0.3, 0.1, 0.0),
        (0.0, 0.0, 0.0),
        (("red.defender", 1.0, 1.0),),
    )


class Policy:
    agent_id = "blue.playmaker"
    contract_hash = "sha256:" + "a" * 64
    activation_ceiling = "SIM_ONLY"
    foundation_hash = "sha256:" + "c" * 64
    foundation_config_hash = "sha256:" + "d" * 64

    def __init__(self):
        self.calls = 0
        self.mode = "zero"

    def propose(self, value):
        self.calls += 1
        if self.mode == "idle":
            return None
        if self.mode == "error":
            raise KeyError("backend bug")
        proposal = NavigationDelta(value.agent_id, value.frame, value.time_sec, (0.0, 0.0, 0.0))
        if self.mode == "oversized":
            object.__setattr__(proposal, "velocity_delta", (10.0, 0.0, 0.0))
        if self.mode == "foreign":
            proposal = replace(proposal, agent_id="red.defender")
        if self.mode == "stale":
            proposal = replace(proposal, frame=value.frame + 1)
        return proposal


def test_zero_is_active_idle_is_not_and_observation_is_immutable():
    policy = Policy()
    slot = NavigationSlot(policy)
    assert slot.propose(observation()) == (0.0, 0.0, 0.0) and slot.active
    policy.mode = "idle"
    assert slot.propose(observation(1)) == (0.0, 0.0, 0.0) and not slot.active
    assert not slot.faulted
    with pytest.raises(AttributeError):
        observation().frame = 9


@pytest.mark.parametrize(
    "mode", ["error", "oversized", "foreign", "stale", "identity", "skip", "clock"]
)
def test_navigation_faults_latch_without_retry_or_unbounded_output(mode):
    policy = Policy()
    slot = NavigationSlot(policy)
    slot.propose(observation())
    policy.mode = mode
    value = observation(1)
    if mode == "identity":
        policy.contract_hash = "sha256:" + "b" * 64
    if mode == "skip":
        value = observation(2)
    if mode == "clock":
        value = replace(value, time_sec=0.1)
    assert slot.propose(value) == (0.0, 0.0, 0.0)
    assert slot.faulted and not slot.active and slot.fault_reason
    calls = policy.calls
    assert slot.propose(observation(3)) == (0.0, 0.0, 0.0)
    assert calls == policy.calls


def test_navigation_contract_rejects_mutable_foreign_and_nonfinite_values():
    value = observation()
    for change in (
        {"body_pose": (0.0,) * 7},
        {"ball_velocity": (float("nan"), 0.0, 0.0)},
        {"neighbors": (("blue.playmaker", 1.0, 1.0),)},
        {"frame": True},
        {"baseline_command": [0.0, 0.0, 0.0]},
    ):
        with pytest.raises(ValueError):
            replace(value, **change)
    for delta in ((0.2, 0.2, 0.0), (0.0, 0.0, 0.401), (float("inf"), 0.0, 0.0)):
        with pytest.raises(ValueError):
            NavigationDelta(value.agent_id, 0, 0.0, delta)
    p = Policy()
    p.activation_ceiling = "REAL"
    with pytest.raises(ValueError):
        NavigationSlot(p)


@pytest.mark.parametrize(
    "effectors",
    [
        [("foot", 0.0, 0.0, 0.0)],
        (("foot", float("nan"), 0.0, 0.0),),
        (("foot", 0.0, 0.0),),
        (("foot", 0.0, 0.0, 0.0),) * 2,
        (("right", 0.0, 0.0, 0.0), ("left", 0.0, 0.0, 0.0)),
    ],
)
def test_measured_effector_snapshot_rejects_invalid_or_ambiguous_values(effectors):
    with pytest.raises(ValueError):
        replace(observation(), effector_positions=effectors)


def test_receive_commitment_requires_explicit_boolean():
    with pytest.raises(ValueError):
        replace(observation(), committed_receiver=1)


@pytest.mark.parametrize("value", [0, 1, None, "completed"])
def test_motor_retirement_requires_explicit_boolean(value):
    with pytest.raises(ValueError):
        replace(observation(), motor_option_retired=value)


def test_motor_retirement_is_a_read_only_lifecycle_observation():
    obs = replace(observation(), motor_option_retired=True)
    assert obs.motor_option_retired and not obs.committed_receiver
    with pytest.raises(AttributeError):
        obs.motor_option_retired = False


def test_tampered_retirement_observation_faults_before_provider_call():
    policy = Policy()
    slot = NavigationSlot(policy)
    obs = observation()
    object.__setattr__(obs, "motor_option_retired", "success")
    assert slot.propose(obs) == (0.0, 0.0, 0.0)
    assert slot.faulted and policy.calls == 0


def test_tampered_effector_snapshot_faults_before_policy_is_called():
    policy = Policy()
    slot = NavigationSlot(policy)
    obs = observation()
    object.__setattr__(obs, "effector_positions", (("foot", float("nan"), 0.0, 0.0),))
    assert slot.propose(obs) == (0.0, 0.0, 0.0)
    assert slot.faulted and policy.calls == 0
