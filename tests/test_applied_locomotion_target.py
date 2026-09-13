import numpy as np
import pytest

from rosclaw_soccer.providers.g1.locomotion_action_frame import encode_applied_locomotion_target
from rosclaw_soccer.providers.g1.mujoco_primitives import mirror_g1_joint_positions


def inputs():
    return dict(
        target=np.linspace(-0.5, 0.5, 29, dtype=np.float32),
        default_angles=np.linspace(0.1, 0.2, 29, dtype=np.float32),
        action_scale=0.25,
        joint_to_motor=np.arange(29)[::-1],
        reflected=False,
    )


@pytest.mark.parametrize("reflected", [False, True])
def test_explicit_applied_target_mapping(reflected):
    args = inputs()
    args["reflected"] = reflected
    raw = encode_applied_locomotion_target(**args)
    decoded = np.empty(29, dtype=np.float32)
    decoded[args["joint_to_motor"]] = raw * 0.25 + args["default_angles"]
    if reflected:
        decoded = mirror_g1_joint_positions(decoded)
    np.testing.assert_allclose(decoded, args["target"], rtol=0, atol=6e-8)
    assert raw.dtype == np.float32
    original = args["target"].copy()
    raw[:] = 0
    np.testing.assert_array_equal(args["target"], original)


@pytest.mark.parametrize(
    "change",
    [
        dict(target=np.zeros(29, dtype=np.float16)),
        dict(target=np.full(29, np.nan, dtype=np.float32)),
        dict(target=np.zeros(28, dtype=np.float32)),
        dict(default_angles=np.full(29, 11.0, dtype=np.float32)),
        dict(joint_to_motor=np.zeros(29, dtype=int)),
        dict(reflected=1),
        dict(action_scale=0),
        dict(action_scale=float("inf")),
        dict(action_scale=0.001),
    ],
)
def test_bad_frame_or_unrepresentable_target(change):
    args = inputs()
    args.update(change)
    with pytest.raises(ValueError):
        encode_applied_locomotion_target(**args)


def test_float64_applied_target_is_not_rounded_before_recoding():
    args = inputs()
    args["target"] = args["target"].astype(np.float64) + 2e-9
    result = encode_applied_locomotion_target(**args)
    expected = ((args["target"][args["joint_to_motor"]] - args["default_angles"]) / 0.25).astype(
        np.float32
    )
    np.testing.assert_array_equal(result, expected)
