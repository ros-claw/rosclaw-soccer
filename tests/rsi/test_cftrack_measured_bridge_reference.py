import sys
from types import ModuleType

import numpy as np
import pytest
from test_shared_omnicontact_player import player
from test_shared_robonaldo_player import setup

from rosclaw_soccer.rsi.cftrack_measured_bridge_reference import condition_measured_bridge_reference
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def test_unqualified_or_mismatched_live_state_rejects_without_mutation(monkeypatch):
    policy_module = ModuleType("policy")
    omni_module = ModuleType("policy.omnicontact")
    locomotion = ModuleType("policy.omnicontact.CFgen_meta1_loco")
    locomotion.DEFAULT_JOINT_POS_MJ = np.zeros(29, np.float32)
    for name in (
        "_append_fk_block",
        "_append_turn",
        "_append_walk",
        "_fk_reference_sequence_from_joints",
    ):
        setattr(locomotion, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.omnicontact", omni_module)
    monkeypatch.setitem(sys.modules, "policy.omnicontact.CFgen_meta1_loco", locomotion)
    world = setup()
    athlete = player(world)
    q, _ = athlete.layout.canonical(world.data)
    roots = np.tile(q[:7], (21, 1))
    joints = np.tile(q[7:36], (21, 1))
    before = world.data.qpos.copy(), world.data.qvel.copy(), world.data.ctrl.copy(), world.data.time
    with pytest.raises(ValueError, match="installed-history"):
        condition_measured_bridge_reference(
            athlete, roots, joints, expected_model_hash=compiled_model_hash(world.model)
        )
    athlete.measured_history_installed = True
    roots[0, 0] += 0.001
    with pytest.raises(ValueError, match="actual initial state"):
        condition_measured_bridge_reference(
            athlete, roots, joints, expected_model_hash=compiled_model_hash(world.model)
        )
    assert not athlete.measured_bridge_reference_installed
    for actual, expected in zip(
        (world.data.qpos, world.data.qvel, world.data.ctrl, world.data.time), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
