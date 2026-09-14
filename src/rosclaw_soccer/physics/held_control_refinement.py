"""Offline MuJoCo refinement with the external servo input held unchanged.

This is a mutating simulation experiment, not a runtime executor or a safety
certificate. Callers own the model/data exclusively and must retain failed runs.
The observer runs at every microstep; ordinary macrostep telemetry is not enough
to certify contact identity after refinement.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

_NATIVE_STEP = mujoco.mj_step


@dataclass(frozen=True)
class HeldControlRefinementReceipt:
    start_sec: float
    end_sec: float
    macro_dt_sec: float
    micro_dt_sec: float
    microsteps: int
    observed_microsteps: int
    activation_ceiling: str = "SIM_ONLY"
    promotion_eligible: bool = False


def step_with_held_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    refinement: int,
    observer: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
) -> HeldControlRefinementReceipt:
    """Advance one macro interval without increasing external control frequency.

    Refinement changes integration, not contact materials or collision geometry.
    It does not guarantee better accuracy or success. The optional observer must
    only read model/data. Input/clock corruption aborts the episode; timestep is
    restored on any exception, but partially advanced simulation is NOT rolled
    back. Installed MuJoCo control callbacks are refused because they could run
    a controller again inside each native microstep.
    """
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("native MuJoCo model and data are required")
    if data.model is not model:
        raise ValueError("data must belong to the exact supplied model")
    if type(refinement) is not int or refinement not in (1, 2, 4, 8, 16):
        raise ValueError("refinement must be 1, 2, 4, 8 or 16")
    if mujoco.get_mjcb_control() is not None:
        raise ValueError("native control callbacks would change servo cadence")
    if observer is not None and not callable(observer):
        raise TypeError("observer must be callable")
    macro_dt = float(model.opt.timestep)
    start = float(data.time)
    if not math.isfinite(macro_dt) or not 1e-5 <= macro_dt <= 0.02:
        raise ValueError("macro timestep must be finite and bounded")
    if not math.isfinite(start) or not 0 <= start <= 3600:
        raise ValueError("simulation time must be finite and bounded")
    if model.nv > 1024 or model.nu > 1024 or model.nbody > 1024:
        raise ValueError("model exceeds offline probe budget")
    held = (data.ctrl.copy(), data.qfrc_applied.copy(), data.xfrc_applied.copy())
    if not all(np.isfinite(a).all() for a in (*held, data.qpos, data.qvel)):
        raise ValueError("simulation state and held controls must be finite")
    micro_dt = macro_dt / refinement
    observed = 0
    try:
        model.opt.timestep = micro_dt
        for index in range(refinement):
            _NATIVE_STEP(model, data)
            if observer is not None:
                observer(model, data)
                observed += 1
            if float(model.opt.timestep) != micro_dt:
                raise ValueError("observer changed integration timestep")
            if not all(
                np.array_equal(expected, actual)
                for expected, actual in zip(
                    held, (data.ctrl, data.qfrc_applied, data.xfrc_applied), strict=True
                )
            ):
                raise ValueError("external control changed inside refined step")
            expected_time = start + (index + 1) * micro_dt
            if not math.isclose(float(data.time), expected_time, rel_tol=0, abs_tol=1e-10):
                raise ValueError("refined simulation clock mismatch")
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError("nonfinite refined simulation state")
    finally:
        model.opt.timestep = macro_dt
    return HeldControlRefinementReceipt(
        start_sec=start,
        end_sec=float(data.time),
        macro_dt_sec=macro_dt,
        micro_dt_sec=micro_dt,
        microsteps=refinement,
        observed_microsteps=observed,
    )
