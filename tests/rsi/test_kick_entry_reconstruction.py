from dataclasses import asdict

import numpy as np
import pytest
from test_kick_measured_history import sample

from rosclaw_soccer.rsi.kick_entry_reconstruction import sample_from_numeric


def test_persisted_actual_history_roundtrip_is_lossless():
    s = sample(5)
    assert asdict(sample_from_numeric(s.numeric_record())) == asdict(s)


@pytest.mark.parametrize("defect", ["short", "fractional", "nan", "wrongclock"])
def test_invalid_history_is_not_silently_repaired(defect):
    x = sample(5).numeric_record()
    if defect == "short":
        x = x[:-1]
    elif defect == "fractional":
        x[0] += 0.1
    elif defect == "nan":
        x[5] = np.nan
    else:
        x[2] = x[0]
    with pytest.raises(ValueError):
        sample_from_numeric(x)


def test_full_trace_carries_original_action_and_clock_without_physics(monkeypatch):
    from types import SimpleNamespace

    from test_shared_robonaldo_player import setup

    from rosclaw_soccer.rsi.kick_entry_reconstruction import KickEntryReconstructor

    world = setup()
    policy = SimpleNamespace(
        runtime_mode="sim",
        use_body_frame_ball=False,
        state_cmd=SimpleNamespace(),
        default_q_mj=np.zeros(29, np.float32),
        action_scale_mj=np.ones(29, np.float32),
        action_clip_lo_il=np.full(29, -1, np.float32),
        action_clip_hi_il=np.ones(29, np.float32),
    )
    clocks = []

    def build():
        clocks.append(policy.time_step)
        return np.r_[policy.last_action_il + np.float32(0.001), np.zeros(518, np.float32)]

    policy._build_obs = build
    policy.ort_session = SimpleNamespace(run=lambda out, feed: [feed["obs"][:, :29].copy()])
    reconstructor = KickEntryReconstructor(world.model, world.layouts["a/"], policy, np.arange(29))

    def first(row, arrays):
        policy.time_step = 100
        reconstructor.last_raw_action = np.full(29, 0.01, np.float32)
        reconstructor.last_clipped_action = reconstructor.last_raw_action.copy()
        return np.r_[reconstructor.last_raw_action, np.zeros(520, np.float32)]

    monkeypatch.setattr(reconstructor, "reconstruct", first)
    targets = np.zeros((8, 29), np.float64)
    action = np.full(29, 0.01, np.float32)
    for frame in (5, 6, 7):
        targets[frame] = action
        action = action + np.float32(0.001)
    arrays = dict(
        qpos=np.repeat(world.data.qpos[None], 80, axis=0),
        qvel=np.repeat(world.data.qvel[None], 80, axis=0),
        torque=np.zeros((80, world.model.nu)),
        joint_targets=targets,
    )
    row = dict(entry_control=5, controls=8)
    memory = reconstructor.reconstruct_trace(row, arrays)
    assert clocks == [101, 102]
    assert memory["observations"].shape == (3, 547)
    np.testing.assert_array_equal(memory["control_frames"], [5, 6, 7])
    np.testing.assert_array_equal(memory["original_raw_actions"], targets[5:].astype(np.float32))
    assert world.native_steps == 0 and reconstructor.data.time == 0
    arrays["joint_targets"][7, 0] += 0.1
    with pytest.raises(ValueError, match="frame7"):
        reconstructor.reconstruct_trace(row, arrays)
