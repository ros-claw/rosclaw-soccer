from __future__ import annotations

import numpy as np

from rosclaw_soccer.training.adaptive_target_teacher_discovery import (
    default_adaptive_target_teacher_probes,
)
from rosclaw_soccer.training.causal_transition_growth import CausalTransitionContext


def _context(index: int) -> CausalTransitionContext:
    return CausalTransitionContext(
        f"test.s130.teacher.{index}",
        (5.1, -0.164, 0.0),
        -0.11 + 0.005 * index,
        1.34 - 0.005 * index,
        (1.19 + 0.002 * index, -0.15),
        0.8,
        0.09 + 0.002 * index,
    )


def test_adaptive_teacher_curriculum_has_dense_three_axis_support() -> None:
    contexts = tuple(_context(index) for index in range(6))
    probes = default_adaptive_target_teacher_probes(contexts)
    assert len(probes) == 24
    assert len({probe.probe_hash for probe in probes}) == 24
    targets = np.asarray(
        [probe.action.target_foot_velocity_xyz_mps for probe in probes], dtype=np.float64
    )
    assert all(np.ptp(targets[:, axis]) > 0.0 for axis in range(3))
    assert set(targets[:, 1]) == {-3.0, -1.0, 0.0, 3.0}
    assert set(targets[:, 2]) == {-1.0, 3.0}
    assert all(probe.teacher_config().maximum_foot_ball_distance_m == 0.5 for probe in probes)
