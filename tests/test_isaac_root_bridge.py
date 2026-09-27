"""Cross-simulator root observation contract tests; no Isaac runtime needed."""

import numpy as np
import pytest

from rosclaw_soccer.sim.isaac_root_bridge import isaac_root_to_mujoco


def test_identity_pose_preserves_world_linear_and_body_angular() -> None:
    q, v = isaac_root_to_mujoco(
        pose_xyzw=np.asarray((1.0, 2.0, 0.793, 0.0, 0.0, 0.0, 1.0)),
        velocity_world=np.asarray((0.3, -0.2, 0.0, 1.0, 2.0, 3.0)),
        asset_quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
    )
    np.testing.assert_allclose(q, (1.0, 2.0, 0.793, 1.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(v, (0.3, -0.2, 0.0, 1.0, 2.0, 3.0))


def test_yaw_rotation_converts_world_angular_to_body() -> None:
    half = np.sqrt(0.5)
    q, v = isaac_root_to_mujoco(
        pose_xyzw=np.asarray((0.0, 0.0, 0.793, 0.0, 0.0, half, half)),
        velocity_world=np.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
        asset_quaternion_xyzw=np.asarray((0.0, 0.0, half, half)),
    )
    np.testing.assert_allclose(q[3:], (1.0, 0.0, 0.0, 0.0), atol=1e-15)
    np.testing.assert_allclose(v, (1.0, 0.0, 0.0, 0.0, -1.0, 0.0), atol=1e-15)


@pytest.mark.parametrize(
    ("pose", "velocity", "asset"),
    [
        ((0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0)),
        ((0, 0, 0, 0, 0, 0, 1), (0, 0, 0, float("nan"), 0, 0), (0, 0, 0, 1)),
    ],
)
def test_invalid_root_fails_closed(pose: tuple, velocity: tuple, asset: tuple) -> None:
    with pytest.raises(ValueError):
        isaac_root_to_mujoco(
            pose_xyzw=np.asarray(pose, dtype=float),
            velocity_world=np.asarray(velocity, dtype=float),
            asset_quaternion_xyzw=np.asarray(asset, dtype=float),
        )
