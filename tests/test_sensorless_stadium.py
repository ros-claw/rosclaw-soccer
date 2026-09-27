from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    build_g1_stadium_model,
    build_g1_stadium_sensorless_model,
)


def test_sensorless_training_model_preserves_cpu_physics() -> None:
    root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not root:
        pytest.skip("set ROSCLAW_G1_ASSET_ROOT for physical model equivalence")
    import mujoco

    goal = G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    canonical = build_g1_stadium_model(Path(root), goal)
    sensorless = build_g1_stadium_sensorless_model(Path(root), goal)
    assert canonical.nsensor > 0 and sensorless.nsensor == 0
    assert (canonical.nq, canonical.nv, canonical.nu, canonical.ngeom) == (
        sensorless.nq,
        sensorless.nv,
        sensorless.nu,
        sensorless.ngeom,
    )
    for attribute in (
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_pos",
        "geom_friction",
        "dof_damping",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        np.testing.assert_array_equal(getattr(canonical, attribute), getattr(sensorless, attribute))
    first = mujoco.MjData(canonical)
    second = mujoco.MjData(sensorless)
    qpos = np.zeros(canonical.nq)
    qpos[:7] = (0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0)
    qpos[7:36] = G1SonicRunupController.default_angles
    qpos[36:43] = (2.5, 0.1, 0.11, 1.0, 0.0, 0.0, 0.0)
    for data in (first, second):
        data.qpos[:] = qpos
        mujoco.mj_forward(canonical if data is first else sensorless, data)
    for _ in range(100):
        mujoco.mj_step(canonical, first)
        mujoco.mj_step(sensorless, second)
        np.testing.assert_allclose(first.qpos, second.qpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(first.qvel, second.qvel, atol=1e-12, rtol=0)
