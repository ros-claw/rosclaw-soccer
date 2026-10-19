"""Synthetic current-state/capture algebra, not integrated physical evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import born_release_parity as mod
from rosclaw_soccer.rsi.born_postcontact_release import BORN_RELEASE_ACTIONS
from rosclaw_soccer.rsi.impact_torque_screen import impact_torque_candidates
from tests.physics.test_g1_feedback_forecast import world
from tests.rsi.test_impact_torque_screen import sample
from tests.rsi.test_navigation_recovery_parity import command_capture


@pytest.fixture
def setup(monkeypatch):
    sim, _, _ = world()

    class Sampler:
        def __init__(self, model, **kw):
            self._qi = list(range(7, 36))
            self._vi = list(range(6, 35))

        def sample(self, data):
            return sample()

    monkeypatch.setattr(mod, "G1JointTaskKinematicSampler", Sampler)
    base = command_capture("baseline")
    base["root_pose_xyzw_m"][:, 0, 2] = 0.79
    base["root_pose_xyzw_m"][:, 0, 6] = 1.0
    base["canonical_qpos"][:, 0, 3] = 1.0
    base["canonical_qpos"][:, 0, 39] = 1.0
    base.update(
        born_release_requested_offset_nm=np.zeros((300, 1, 29)),
        born_release_applied_offset_nm=np.zeros((300, 1, 10, 29)),
        born_release_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
    )
    return sim, base


def capture(base, action):
    out = copy.deepcopy(base)
    if action != "baseline":
        name = action.split(":", 1)[1]
        held = dict(impact_torque_candidates(sample(), 0)).get(name, np.zeros(29))
        out["born_release_requested_offset_nm"][186:189, 0] = held
        out["born_release_applied_offset_nm"][186:189, 0] = np.tile(held, (10, 1))
        out["torque_nm"][186:189, 0] = np.tile(held, (10, 1))
    return out


@pytest.mark.parametrize("action", ["baseline", *BORN_RELEASE_ACTIONS])
def test_literal_reconstruction_matches_original_mapper_on_all_actions(setup, action):
    sim, base = setup
    report = mod.review_born_release(
        capture(base, action), base, action_name=action, model=sim, world_hash="synthetic"
    )
    assert report["start_frame"] == 186 and report["new_physical_steps"] == 0
    assert report["native_torque_attribution_rebuilt"]


@pytest.mark.parametrize(
    "change",
    ["request", "applied", "prefix", "first_body", "forecast", "old_namespace", "invalid_unsafe"],
)
def test_invalid_capture_rejected_without_integration(setup, change):
    sim, base = setup
    action = "born_release:forward_minus"
    out = capture(base, action)
    if change == "request":
        out["born_release_requested_offset_nm"][186, 0, 0] += 1e-12
    elif change == "applied":
        out["born_release_applied_offset_nm"][186, 0, 0, 0] += 0.1
    elif change == "prefix":
        out["joint_target_rad"][100, 0, 0] = 1
    elif change == "first_body":
        out["canonical_qpos"][186, 0, 0] = 1
    elif change == "forecast":
        out["born_release_forecast_steps"][186, 0, 0] = 1
    elif change == "old_namespace":
        out["postcontact_release_requested_offset_nm"] = out.pop("born_release_requested_offset_nm")
    elif change == "invalid_unsafe":
        out["root_pose_xyzw_m"][186, 0, 2] = 0.6
        action = "born_release:unknown"
    with pytest.raises(ValueError):
        mod.review_born_release(out, base, action_name=action, model=sim, world_hash="synthetic")


def test_new_observed_nonfoot_contact_cancels_remaining_lease(setup):
    sim, base = setup
    action = "born_release:forward_minus"
    out = capture(base, action)
    out["force_n"][186, 0, 2] = 2.0
    for key in ("born_release_requested_offset_nm", "born_release_applied_offset_nm", "torque_nm"):
        out[key][187:] = 0
    report = mod.review_born_release(
        out, base, action_name=action, model=sim, world_hash="synthetic"
    )
    assert report["current_jacobian_requests_rebuilt"]
