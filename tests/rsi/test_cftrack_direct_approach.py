import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach


def fixture(monkeypatch):
    class Builder:
        def finalize(self):
            return {
                "ref_phase": np.full(3, 12, np.int32),
                "ref_contact": np.zeros((3, 4), np.float32),
                "ref_base_pos": np.array(
                    [[0, 0, 0.77], [0.4, 0.06, 0.77], [0.8, 0.12, 0.77]], np.float32
                ),
                "ref_base_quat": np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1)),
            }

    def walk(builder, phase, **kwargs):
        assert phase == 12 and kwargs["step_linear"] == 0.016
        np.testing.assert_array_equal(kwargs["pelvis_start"], [0, 0, 0.77])

    monkeypatch.setitem(
        sys.modules,
        "policy.omnicontact.CFgen_meta1_loco",
        SimpleNamespace(DEFAULT_JOINT_POS_MJ=np.zeros(29), _append_loco_walk=walk),
    )
    state = SimpleNamespace(
        base_pos=np.array([0.0, 0.0, 0.77]),
        base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        q=np.zeros(29),
        ball_pos=np.array([1.0, 0.0, 0.11]),
        obj_pos=np.array([1.0, 0.0, 0.11]),
        obj_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    p = SimpleNamespace(
        task="kickball",
        reference_source="CFgen",
        counter_step=0,
        async_stage_thread=None,
        obs_history_buffer=np.zeros((5, 141)),
        action=np.zeros(29),
        state_cmd=state,
        goal_pos=np.array([5.0, 0.0, 0.11]),
        traj_generator=SimpleNamespace(step_linear=0.016, _new_builder=Builder),
        ref_phase=np.array(
            [11] * 5 + [12] * 5 + [13] * 2 + [14] * 3 + [15] * 2 + [16] * 3, np.int32
        ),
        ref_contact=np.arange(80, dtype=np.float32).reshape(20, 4),
        ref_base_pos=np.tile(np.array([0.8, 0.12, 0.77], np.float32), (20, 1)),
        ref_base_quat=np.tile(np.array([1, 0, 0, 0], np.float32), (20, 1)),
    )
    return p


def test_direct_reference_preserves_swing_recovery_and_private_history(monkeypatch):
    p = fixture(monkeypatch)
    suffix = {
        k: getattr(p, k)[10:].copy()
        for k in ("ref_phase", "ref_contact", "ref_base_pos", "ref_base_quat")
    }
    state, history = p.state_cmd, p.obs_history_buffer
    receipt = condition_direct_kick_approach(p)
    assert receipt["original_frames"] == 20 and receipt["conditioned_frames"] == 13
    assert receipt["original_kick_and_recovery_suffix_exact"]
    for key, expected in suffix.items():
        np.testing.assert_array_equal(getattr(p, key)[3:], expected)
    assert p.state_cmd is state and p.obs_history_buffer is history and not history.any()
    assert p.counter_step == 0 and not p.action.any()


@pytest.mark.parametrize("defect", ["advanced", "history", "yaw", "behind", "pose", "shape"])
def test_invalid_entry_leaves_all_reference_arrays_untouched(monkeypatch, defect):
    p = fixture(monkeypatch)
    if defect == "advanced":
        p.counter_step = 1
    elif defect == "history":
        p.obs_history_buffer[0, 0] = 1
    elif defect == "yaw":
        p.state_cmd.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
    elif defect == "behind":
        p.state_cmd.base_pos[0] = 2
    elif defect == "pose":
        p.state_cmd.q[0] = 0.1
    else:
        p.ref_contact = np.zeros((19, 4), np.float32)
    old = {k: v.copy() for k, v in vars(p).items() if k.startswith("ref_")}
    with pytest.raises(ValueError):
        condition_direct_kick_approach(p)
    for key, expected in old.items():
        np.testing.assert_array_equal(getattr(p, key), expected)
