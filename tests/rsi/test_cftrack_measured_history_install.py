import copy

import numpy as np
import pytest
from test_shared_omnicontact_player import Policy
from test_shared_robonaldo_player import setup

from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
from rosclaw_soccer.rsi.cftrack_measured_history_install import install_owned_cftrack_history
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


class ScratchPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.obs_history_buffer = np.zeros((5, 141), np.float32)
        self.action = np.zeros(29, np.float32)
        self.default_angles_lab = np.zeros(29, np.float32)
        self.action_scale_lab = np.ones(29, np.float32)
        self.active_object_name = "ball"

    def _get_fk_info(self):
        pass


def prepare(monkeypatch):
    world = setup()
    history = CFtrackLiveHistory(world, "a/")
    for i in range(5):
        world.advance({"a/": np.full(29, 0.001 * (i + 1)), "b/": np.zeros(29)})
        history.capture()
    athlete = SharedOmniContactPlayer(
        world,
        "a/",
        ScratchPolicy(),
        expected_policy_type=ScratchPolicy,
        goal=np.array([5.0, 0.0, 0.11]),
    )
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.cftrack_measured_history_install.copy_cftrack_policy", copy.deepcopy
    )
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.cftrack_measured_history_install.cftrack_state_hash", lambda p: "same"
    )

    def assemble(p):
        row = np.zeros(141, np.float32)
        row[:29], row[79:108] = p.state_cmd.q, p.action
        return row

    monkeypatch.setattr(
        "rosclaw_soccer.rsi.cftrack_measured_history_install.assemble_current_ball_observation",
        assemble,
    )
    return world, history, athlete


def test_owned_install_uses_actual_prior_targets_without_touching_physics(monkeypatch):
    world, history, athlete = prepare(monkeypatch)
    before = world.data.qpos.copy(), world.data.qvel.copy(), world.data.ctrl.copy(), world.data.time
    receipt = install_owned_cftrack_history(athlete, history)
    assert receipt["sample_controls"] == [1, 2, 3, 4, 5]
    assert receipt["previous_actions_are_implied_from_executed_targets"]
    assert not receipt["policy_reference_qualified"]
    np.testing.assert_allclose(athlete.policy.action, 0.005)
    for i in range(4):
        np.testing.assert_allclose(
            athlete.policy.obs_history_buffer[i + 1, 79:108], (i + 1) * 0.001
        )
    assert not athlete.policy.obs_history_buffer[0].any()
    for actual, expected in zip(
        (world.data.qpos, world.data.qvel, world.data.ctrl, world.data.time), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):
        install_owned_cftrack_history(athlete, history)


def test_stale_or_foreign_history_never_installs(monkeypatch):
    world, history, athlete = prepare(monkeypatch)
    world.advance({"a/": np.zeros(29), "b/": np.zeros(29)})
    with pytest.raises(ValueError):
        install_owned_cftrack_history(athlete, history)
    assert not athlete.policy.action.any() and not athlete.policy.obs_history_buffer.any()
