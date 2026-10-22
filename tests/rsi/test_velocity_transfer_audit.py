import numpy as np
import pytest

from rosclaw_soccer.rsi.velocity_transfer_audit import summarize_velocity_raw, validate_velocity_raw


def raw_fixture():
    widths = dict(
        qpos_before=(43,),
        qvel_before=(41,),
        observation=(98,),
        action=(29,),
        target=(29,),
        command_body=(3,),
        command_world=(3,),
        omega=(3,),
        phase=(),
        qpos=(10, 43),
        qvel=(10, 41),
        torque=(10, 29),
        force=(10, 29),
        ball_force=(10,),
    )
    f32 = {"observation", "action", "command_body", "phase"}
    raw = {
        k: np.zeros((300, *shape), np.float32 if k in f32 else np.float64)
        for k, shape in widths.items()
    }
    raw["qpos_before"][:, 2] = raw["qpos"][:, :, 2] = 0.8
    raw["qpos_before"][:, 3] = raw["qpos"][:, :, 3] = 1
    return raw


def test_complete_shape_and_recomputed_outcome():
    raw = raw_fixture()
    assert validate_velocity_raw(raw) == 300
    own = summarize_velocity_raw(raw, 0, "velocity_home", 0.0)
    assert own["body_stable"] and own["stop_drift_m"] == 0
    assert own["native_steps"] == 3000 and own["neural_calls"] == 300


@pytest.mark.parametrize("bad", ["splice", "nan", "dtype", "shape", "extra"])
def test_unchecked_corrupt_arrays_cannot_reach_native_replay(bad):
    raw = raw_fixture()
    if bad == "splice":
        raw["qvel_before"][1, 0] = 1
    elif bad == "nan":
        raw["command_world"][0, 0] = np.nan
    elif bad == "dtype":
        raw["phase"] = raw["phase"].astype(np.float64)
    elif bad == "shape":
        raw["action"] = raw["action"][:, :28]
    else:
        raw["unverified"] = np.zeros(300)
    with pytest.raises(ValueError):
        validate_velocity_raw(raw)
