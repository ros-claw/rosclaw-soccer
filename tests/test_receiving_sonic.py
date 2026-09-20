from dataclasses import replace

import pytest

from rosclaw_soccer.providers.g1.receiving_sonic import ReceivingSonicOption


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
