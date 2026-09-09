"""Numerical backend qualification before any learning episode is declared.

Checking serialized coefficients is insufficient: older MJWarp kernels apply
the first free-joint damping coefficient to all six DOFs. Test actual passive
forces against native MuJoCo, without changing the model or taking a step.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def validate_damping_forces(expected: Any, measured: Any) -> float:
    expected, measured = np.asarray(expected), np.asarray(measured)
    if (
        expected.ndim != 2
        or expected.shape != measured.shape
        or not expected.size
        or not np.isfinite(expected).all()
        or not np.isfinite(measured).all()
    ):
        raise ValueError("finite matching passive damping force samples required")
    error = float(np.max(np.abs(expected - measured)))
    if not np.allclose(expected, measured, rtol=1e-5, atol=1e-7):
        raise ValueError(
            f"MJWarp per-DOF damping differs from native MuJoCo (max error {error:.6g}); "
            "do not train on this backend; qualify an isolated corrected build"
        )
    return error


def qualify_mjwarp_damping(cpu_model: Any, *, device: str) -> dict[str, Any]:
    """Use scratch state, actual kernels and three signed velocity probes.

    This is a component contract, not contact/trajectory parity or promotion.
    The caller's model and episode state are never modified. No global patch
    or callback is installed. Backend kernel identity is bound in the result.
    """
    if not isinstance(device, str) or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError("explicit CUDA qualification device required")
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from mujoco_warp._src import passive

    wp.init()
    with wp.ScopedDevice(device):
        gm = mjw.put_model(cpu_model)
        native = mujoco.MjData(cpu_model)
        mujoco.mj_forward(cpu_model, native)
        gpu = mjw.put_data(cpu_model, native, nworld=1, nconmax=256, njmax=1024)
        expected, measured = [], []
        for scale in (0.25, -0.75, 2.0):
            velocity = scale * np.linspace(0.5, 1.5, cpu_model.nv)
            native.qvel[:] = velocity
            mujoco.mj_forward(cpu_model, native)
            gpu.qvel.assign(velocity.astype(np.float32)[None])
            mjw.forward(gm, gpu)
            expected.append(native.qfrc_damper.copy())
            measured.append(gpu.qfrc_damper.numpy()[0].copy())
    expected_array, measured_array = np.asarray(expected), np.asarray(measured)
    error = validate_damping_forces(expected_array, measured_array)
    kernel_path = Path(passive.__file__)
    payload = {
        "schema": "rosclaw_soccer.mjwarp_passive_damping.v1",
        "mujoco_version": mujoco.__version__,
        "warp_version": wp.__version__,
        "device": device,
        "passive_kernel_hash": hash_bytes(kernel_path.read_bytes()),
        "dof_damping_hash": hash_bytes(np.asarray(cpu_model.dof_damping).tobytes()),
        "expected_force_hash": hash_bytes(expected_array.tobytes()),
        "measured_force_hash": hash_bytes(measured_array.tobytes()),
        "maximum_absolute_force_error": error,
        "probes": 3,
        "dofs": int(cpu_model.nv),
        "physics_steps": 0,
        "activation_ceiling": "SIM_ONLY",
    }
    return {**payload, "contract_hash": hash_json(payload)}
