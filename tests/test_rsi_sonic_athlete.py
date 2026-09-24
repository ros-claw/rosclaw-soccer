from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation
from rosclaw_soccer.rsi.contracts import AthleteObservation, AthleticIntent, PolicyArtifact
from rosclaw_soccer.rsi.sonic_athlete import SonicAthleteAdapter
from rosclaw_soccer.sim.contracts import hash_json

H = "sha256:" + "a" * 64
J = "sha256:" + "b" * 64


class FakeBackend:
    def __init__(self, *_args):
        self.qualification = SimpleNamespace(
            qualification_hash=H,
            planner_hash=H,
            encoder_hash=H,
            decoder_hash=H,
            require_eligible=lambda: None,
        )
        self.kp = np.ones(29) * 50
        self.kd = np.ones(29)
        self.reference = np.zeros((710, 36))
        self._history = []
        self.measurements = []

    def reset(self, state):
        self._history.append((state.qpos.copy(), state.qvel.copy()))

    def observe(self, state):
        self.measurements.append((state.qpos.copy(), state.qvel.copy()))

    def navigation_tick(self, _state, _frame):
        return np.ones(29) * self.command[0] * 0.1


def _observation(frame=0):
    return AthleteObservation(
        body_id="red.defender",
        body_hash=H,
        joint_map_hash=H,
        physics_hash=H,
        frame=frame,
        time_sec=frame * 0.02,
        root_position_m=(0.0, 0.0, 0.75),
        root_velocity_mps=(0.0, 0.0, 0.0),
        root_angular_velocity_rad_s=(0.0, 0.0, 0.0),
        root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        joint_position=(0.0,) * 29,
        joint_velocity=(0.0,) * 29,
        contact_flags=(True, True),
    )


def _intent():
    return AthleticIntent((0.5, 0.0), 0.0, 0.0, 0.793, "locomotion", "soccer")


def _adapter(monkeypatch):
    monkeypatch.setattr(
        "rosclaw_soccer.providers.g1.sonic_navigation._StreamingBackend", FakeBackend
    )
    navigation = G1SonicNavigation(None, "red.defender")
    artifact = PolicyArtifact(
        artifact_id="red.defender.sonic",
        backend_id="sonic_navigation",
        backend_contract_hash=navigation.contract_hash,
        code_hash=H,
        weights_hash=hash_json({"planner": H, "encoder": H, "decoder": H}),
        body_hash=H,
        observation_hash=H,
        action_hash=H,
        joint_map_hash=H,
        physics_hash=H,
        gain_hash=hash_json({"kp": [50.0] * 29, "kd": [1.0] * 29}),
        action_kind="JOINT_TARGET",
        action_size=29,
        control_dt_s=0.02,
    )
    return navigation, artifact


def test_sonic_athlete_maps_only_robot_proprioception_and_preserves_clock(monkeypatch):
    navigation, artifact = _adapter(monkeypatch)
    adapter = SonicAthleteAdapter(navigation, artifact)
    first = adapter.step(_observation(285), _intent())
    second = adapter.step(_observation(286), _intent())
    assert first.joint_target == (0.05,) * 29
    assert second.frame == 286
    assert first.activation_ceiling == "SIM_ONLY"
    assert navigation.backend._history[0][0].shape == (43,)
    assert navigation.backend._history[0][1].shape == (41,)
    assert navigation.backend._history[0][0][39] == 1.0
    assert len(navigation.backend.measurements) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "foreign_body",
        "foreign_map",
        "foreign_physics",
        "contact_intent",
        "absolute_heading",
        "height_request",
        "target_position",
        "too_fast",
        "stale_frame",
    ],
)
def test_sonic_athlete_bad_state_latches_no_retry(monkeypatch, fault):
    navigation, artifact = _adapter(monkeypatch)
    adapter = SonicAthleteAdapter(navigation, artifact)
    obs = _observation()
    intent = _intent()
    if fault == "foreign_body":
        obs = replace(obs, body_hash=J)
    elif fault == "foreign_map":
        obs = replace(obs, joint_map_hash=J)
    elif fault == "foreign_physics":
        obs = replace(obs, physics_hash=J)
    elif fault == "contact_intent":
        intent = replace(intent, contact_intent="kick")
    elif fault == "absolute_heading":
        intent = replace(intent, heading_rad=0.4)
    elif fault == "height_request":
        intent = replace(intent, body_height_m=0.7)
    elif fault == "target_position":
        intent = replace(intent, future_target_xy_m=(1.0, 0.0))
    elif fault == "too_fast":
        intent = replace(intent, velocity_xy_mps=(0.8, 0.0))
    else:
        adapter.step(obs, intent)
    with pytest.raises(ValueError, match="latched off"):
        adapter.step(obs, intent)
    with pytest.raises(ValueError, match="fault latched"):
        adapter.step(_observation(1), _intent())


def test_sonic_athlete_refuses_wrong_artifact_and_runtime_gain_drift(monkeypatch):
    navigation, artifact = _adapter(monkeypatch)
    with pytest.raises(ValueError, match="binding mismatch"):
        SonicAthleteAdapter(navigation, replace(artifact, physics_hash=J, weights_hash=J))
    adapter = SonicAthleteAdapter(navigation, artifact)
    navigation.backend.kp *= 2
    with pytest.raises(ValueError, match="latched off"):
        adapter.step(_observation(), _intent())
