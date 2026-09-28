"""Physical ball-foot relative velocity is measured, finite and read-only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.contact_point_velocity import (
    measure_contact_relative_velocity_world_mps,
)


def test_world_point_jacobians_measure_counterpart_minus_ball(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mujoco

    model = SimpleNamespace(nv=2, ngeom=2, geom_bodyid=np.asarray((0, 1)))
    data = SimpleNamespace(qvel=np.asarray((1.0, 2.0)))

    def jac(_model, _data, linear, angular, point, body):
        assert np.array_equal(point, (0.5, 0.0, 0.1))
        linear[:] = 0
        linear[0, body] = 1
        angular[:] = 0

    monkeypatch.setattr(mujoco, "mj_jac", jac)
    measured = measure_contact_relative_velocity_world_mps(
        model=model,
        data=data,
        ball_geom=0,
        counterpart_geom=1,
        contact_position_world_m=np.asarray((0.5, 0.0, 0.1)),
    )
    assert measured == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        measure_contact_relative_velocity_world_mps(
            model=model,
            data=data,
            ball_geom=0,
            counterpart_geom=1,
            contact_position_world_m=np.asarray((float("nan"), 0.0, 0.1)),
        )
