from dataclasses import replace

import pytest

from rosclaw_soccer.providers.g1.receiving_sonic import (
    ReceivingSonicBallFollowOption,
    ReceivingSonicOption,
)


class Navigation:
    contract_hash = "sha256:" + "1" * 64

    def __init__(self, root, agent_id, config):
        self.config = config
        self.calls = []
        self.starts = []

    def start_from_observation(self, observation):
        self.starts.append(observation)

    def propose(self, observation):
        self.calls.append(observation)
        return "proposal"


def motor(monkeypatch, **kwargs):
    monkeypatch.setattr("rosclaw_soccer.providers.g1.receiving_sonic.G1SonicNavigation", Navigation)
    return ReceivingSonicOption(None, "red.defender", start_frame=2, **kwargs)


def test_prefix_retains_original_world_and_entry_is_once(monkeypatch):
    from test_s304_sonic_navigation import observation

    option = motor(monkeypatch, velocity_scale=0.5)
    assert option.propose(observation(0)) is None
    assert option.propose(observation(1)) is None
    assert not option.navigation.calls and not option.navigation.starts
    assert option.propose(observation(2)) == "proposal"
    option.propose(observation(3))
    assert len(option.navigation.starts) == 1
    assert option.navigation.starts[0] == option.navigation.calls[0]
    assert option.navigation.calls[0].navigation_command == (0.25, 0.0, 0.0)
    assert option.navigation.config.model_variant == "low_latency"


def test_bad_prefix_observation_latches(monkeypatch):
    from test_s304_sonic_navigation import observation

    option = motor(monkeypatch)
    with pytest.raises(ValueError):
        option.propose(replace(observation(0), agent_id="blue.defender"))
    with pytest.raises(ValueError, match="latched"):
        option.propose(observation(0))
    assert not option.navigation.calls


def test_scale_binds_identity_and_cannot_increase_command(monkeypatch):
    assert motor(monkeypatch).contract_hash != motor(monkeypatch, velocity_scale=0.5).contract_hash
    for scale in (True, -0.1, 1.01, float("nan")):
        with pytest.raises(ValueError):
            motor(monkeypatch, velocity_scale=scale)


def test_longitudinal_brake_keeps_original_lateral_ball_follow(monkeypatch):
    from test_s304_sonic_navigation import observation

    monkeypatch.setattr("rosclaw_soccer.providers.g1.receiving_sonic.G1SonicNavigation", Navigation)
    common = dict(start_frame=0, response_gain=0.75, fast_replan=True, brake_distance_m=0.65)
    combined = ReceivingSonicBallFollowOption(None, "red.defender", **common)
    longitudinal = ReceivingSonicBallFollowOption(None, "red.defender", **common, brake_axis="x")
    old = observation()
    qpos = list(old.qpos)
    qvel = list(old.qvel)
    qpos[36] = 0.4
    qpos[37] = 0.2
    qvel[35] = -0.3
    qvel[36] = 0.4
    measured = replace(old, qpos=tuple(qpos), qvel=tuple(qvel))
    xy = combined._navigation_command(measured)
    x = longitudinal._navigation_command(measured)
    assert x[0] == pytest.approx(xy[0])
    assert x[1] > xy[1]
    assert longitudinal.contract_hash != combined.contract_hash


def test_longitudinal_brake_requires_declared_brake_distance(monkeypatch):
    monkeypatch.setattr("rosclaw_soccer.providers.g1.receiving_sonic.G1SonicNavigation", Navigation)
    with pytest.raises(ValueError, match="axis"):
        ReceivingSonicBallFollowOption(
            None, "red.defender", start_frame=0, response_gain=0.75, brake_axis="x"
        )
