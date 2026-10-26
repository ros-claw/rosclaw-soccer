from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_stage_boundary import CFtrackStageBoundary


def test_actual_waypoint_and_ball_projection_gate_before_continuation():
    qpos = np.zeros(43)
    qvel = np.zeros(41)
    qpos[:3] = [1.0, 0.024, 0.11]
    qpos[7:14] = [0.0, 0.08, 0.77, 1.0, 0.0, 0.0, 0.0]
    qvel[:3] = [0.0, 0.12, 0.0]
    data = SimpleNamespace(qpos=qpos, qvel=qvel)
    world = SimpleNamespace(
        data=data, control_steps=10, failed=False, _require_current_boundary=lambda: None
    )
    layout = SimpleNamespace(rq=7, bq=0, bv=0)
    guard = CFtrackStageBoundary(10, (0.0, 0.08), (1.0, 0.1728, 0.11), (5.0, 0.0, 0.11), 1.44)
    assert guard.check(world, layout)["ready"]
    data.qpos[1] += 0.2
    result = guard.check(world, layout)
    assert not result["ready"] and result["ball_projection_error_m"] > 0.08
    world.control_steps = 11
    with pytest.raises(ValueError, match="exact actual"):
        guard.check(world, layout)


@pytest.mark.parametrize("horizon", [0.9, 2.1, float("nan")])
def test_invalid_stage_commitment_fails_before_physics(horizon):
    with pytest.raises(ValueError, match="commitment"):
        CFtrackStageBoundary(10, (0.0, 0.08), (1.0, 0.1728, 0.11), (5.0, 0.0, 0.11), horizon)
