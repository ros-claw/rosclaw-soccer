"""Legacy formula copied as a test oracle; no physical performance claims."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.feedback_kick_approach import velocity_matched_approach
from rosclaw_soccer.rsi.walk_kick_approach_command import approach_command


def old_command(q, v, rotation, cached_ball_position, previous, parameters, fast, feedback, halt):
    desired_world = np.array(
        [
            parameters["speed_mps"],
            np.clip(
                2 * (cached_ball_position[1] + parameters["root_lateral_offset_m"] - q[1]),
                -0.4,
                0.4,
            ),
        ]
    )
    if feedback:
        desired_world = velocity_matched_approach(
            root_position=q[:3],
            ball_position=q[36:39],
            ball_velocity=v[35:38],
            cruise_speed=parameters["speed_mps"],
            entry_gap=parameters["entry_gap_m"],
            lateral_offset=parameters["root_lateral_offset_m"],
        )
    if halt:
        desired_world[:] = 0.0
    xy = rotation[:2, :2].T @ desired_world
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    wanted = np.array(
        [
            np.clip(xy[0], -0.5, 2.0 if fast else 1.0),
            np.clip(xy[1], -0.5, 0.5),
            np.clip(-2 * yaw, -1, 1),
        ],
        np.float32,
    )
    if halt:
        wanted[:] = 0.0
    if fast:
        norm = float(np.linalg.norm(wanted[:2]))
        if norm > 2:
            wanted[:2] *= np.float32(2 / norm)
    return previous + np.clip(wanted - previous, -0.05, 0.05)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("halt", [False, True])
def test_factored_command_is_bit_exact_and_inputs_unchanged(fast, feedback, halt):
    rng = np.random.default_rng(1701)
    for _ in range(128):
        yaw = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        inputs = dict(
            q=rng.normal(size=43),
            v=rng.normal(size=41),
            rotation=np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64),
            cached_ball_position=rng.normal(size=3),
            previous=rng.uniform(-1, 1, 3).astype(np.float32),
            parameters=dict(
                speed_mps=float(rng.uniform(0.5, 2)),
                entry_gap_m=float(rng.uniform(0.45, 1.1)),
                root_lateral_offset_m=float(rng.uniform(-0.15, 0.15)),
            ),
            fast=fast,
            feedback=feedback,
            halt=halt,
        )
        copies = {k: v.copy() for k, v in inputs.items() if isinstance(v, np.ndarray)}
        expected = old_command(**inputs)
        actual = approach_command(**inputs)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        for k, v in copies.items():
            np.testing.assert_array_equal(v, inputs[k])
