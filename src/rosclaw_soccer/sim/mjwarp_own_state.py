"""Slow, explicit GPU-own-state bridge for closed-loop qualification only.

The host MjData is a mirror of GPU outputs, not a CPU forward/step solution.
Only control inputs cross back to the device after initialization. This is
deliberately not a training/vectorization backend or an agreement certificate.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from rosclaw_soccer.sim.mjwarp_contract import _put_model_checked


class MjWarpOwnStateDiagnostic:
    """Single-world diagnostic; never installs global MuJoCo replacements.

    Initialize from a caller-owned, already-forwarded native initial state.
    Subsequently read observations from that same host object and call step.
    CPU mj_objectVelocity/mj_jacBody/mj_contactForce may read the exported GPU
    intermediates, but CPU mj_forward/mj_step must not be called on the mirror.
    Constraint overflow is rejected *before* the backend's clamping exporter.
    A failure permanently invalidates this instance; no implicit retry/reset.
    """

    activation_ceiling = "SIM_ONLY"
    qualification = "UNQUALIFIED_GPU_OWN_STATE_DIAGNOSTIC"

    def __init__(
        self, model: Any, data: Any, *, device: str, nconmax: int = 256, njmax: int = 1024
    ) -> None:
        if not isinstance(device, str) or re.fullmatch(r"cuda:[0-9]+", device) is None:
            raise ValueError("explicit CUDA diagnostic device required")
        for value in (nconmax, njmax):
            if type(value) is not int or value <= 0:
                raise ValueError("positive integer diagnostic capacities required")
        import mujoco_warp as mjw
        import warp as wp

        self._backend = mjw
        self._wp = wp
        self._device = device
        self._model = model
        self._host = data
        self._valid = False
        self.physics_steps = 0
        with wp.ScopedDevice(device):
            self._gpu_model = _put_model_checked(mjw, model)
            self._gpu = mjw.put_data(model, data, nworld=1, nconmax=nconmax, njmax=njmax)
            # No forward: preserve the precise pre-/post-integration convention
            # of put_data and step, including cached body positions/velocities.
            self._export()
        self._valid = True

    def _export(self) -> None:
        gpu = self._gpu
        for name, capacity in (("nacon", gpu.naconmax), ("nefc", gpu.njmax)):
            count = np.asarray(getattr(gpu, name).numpy())
            if (
                count.shape != (1,)
                or count.dtype.kind not in "iu"
                or count[0] < 0
                or count[0] > capacity
            ):
                raise FloatingPointError(f"GPU {name} overflow/invalid count; no truncated export")
        self._backend.get_data_into(self._host, self._model, gpu, world_id=0)
        for name in (
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "cvel",
            "cdof",
            "subtree_com",
            "actuator_force",
            "efc_force",
        ):
            if not np.isfinite(getattr(self._host, name)).all():
                raise FloatingPointError(f"nonfinite exported GPU {name}")
        if not np.isfinite(self._host.time):
            raise FloatingPointError("nonfinite exported GPU time")
        self._last_qpos = self._host.qpos.copy()
        self._last_qvel = self._host.qvel.copy()
        self._last_time = self._host.time

    def step(self, model: Any, data: Any) -> None:
        """Advance GPU physics once, then export its own solved intermediates."""
        if not self._valid:
            raise RuntimeError("GPU diagnostic invalid; construct a new isolated episode")
        try:
            if model is not self._model or data is not self._host:
                raise ValueError("diagnostic is bound to one exact model/state object")
            if (
                not np.array_equal(data.qpos, self._last_qpos)
                or not np.array_equal(data.qvel, self._last_qvel)
                or data.time != self._last_time
            ):
                raise ValueError("host state was overwritten; GPU owns physical state")
            control = np.asarray(data.ctrl)
            if control.shape != (model.nu,) or not np.isfinite(control).all():
                raise ValueError("finite complete control vector required")
            if np.any(np.abs(control) > np.finfo(np.float32).max):
                raise ValueError("control exceeds finite GPU float32 representation")
            with self._wp.ScopedDevice(self._device):
                self._gpu.ctrl.assign(control.astype(np.float32)[None])
                self._backend.step(self._gpu_model, self._gpu)
                self.physics_steps += 1
                self._export()
        except Exception:
            self._valid = False
            raise
