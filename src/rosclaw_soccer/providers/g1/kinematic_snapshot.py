"""Private current-qpos MuJoCo Warp geometry for simulation observations.

Stepping may leave derived geometry one integration tick behind qpos. This
reader recomputes geometry into private buffers, never into the live solver.
The synchronous simulation owner remains responsible for stream/execution
serialization; this is not a concurrent access lock or a hardware interface.
"""

import dataclasses
import hashlib
import inspect
from pathlib import Path
from typing import Any

_KINEMATICS_HASH = "3a58e70bfc349e7f053c1e263dffa04ff51b07b8733689fd8c2ab0ba732047f7"
_SMOOTH_SOURCE_HASH = "9df102b1cea990dc83afad88b57f2db31e16ac657faa3597410b8a85eb890d47"
_OUTPUTS = (
    "xpos",
    "xquat",
    "xanchor",
    "xaxis",
    "xmat",
    "xipos",
    "ximat",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
)
_INPUTS = ("qpos", "qvel", "mocap_pos", "mocap_quat", "time")


class WarpKinematicsSnapshot:
    """Read-only simulation adapter for one explicitly qualified kernel write set.

    Kernel changes require a new write-set and nonmutation qualification, not
    automatic tolerance widening. No contact forces are recomputed or exposed
    as current geometry: forces belong to their completed physics interval.
    """

    def __init__(self, model: Any, data: Any) -> None:
        import mujoco_warp as mjw
        import warp as wp

        digest = hashlib.sha256(inspect.getsource(mjw.kinematics).encode()).hexdigest()
        if (
            digest != _KINEMATICS_HASH
            or not dataclasses.is_dataclass(data)
            or isinstance(data, type)
        ):
            raise ValueError("unqualified simulation kinematics write set")
        source = inspect.getsourcefile(inspect.unwrap(mjw.kinematics))
        if (
            source is None
            or hashlib.sha256(Path(source).read_bytes()).hexdigest() != _SMOOTH_SOURCE_HASH
        ):
            raise ValueError("unqualified simulation kinematics kernel implementation")
        nworld = getattr(data, "nworld", None)
        if type(nworld) is not int or not 1 <= nworld <= 4096:
            raise ValueError("bounded simulation snapshot batch required")
        self._model = model
        self._kinematics = mjw.kinematics
        self._live = data
        self._invalid = False
        copy_source: Any = data
        self._shadow = dataclasses.replace(
            copy_source, **{key: wp.clone(getattr(data, key)) for key in (*_OUTPUTS, *_INPUTS)}
        )
        for key in (*_OUTPUTS, *_INPUTS):
            source, target = getattr(data, key), getattr(self._shadow, key)
            if source.size and source.ptr == target.ptr:
                raise ValueError("snapshot storage aliases live simulation state")

    def read(self) -> dict[str, Any]:
        """Return private Torch copies, including the captured native simulation time."""
        import torch
        import warp as wp

        if self._invalid:
            raise RuntimeError("simulation kinematics snapshot is invalid")
        try:
            for key in _INPUTS:
                wp.copy(getattr(self._shadow, key), getattr(self._live, key))
                value = wp.to_torch(getattr(self._shadow, key))
                if not bool(torch.isfinite(value).all()) or bool((value.abs() > 1e6).any()):
                    raise ValueError("invalid simulation snapshot input")
            self._kinematics(self._model, self._shadow)
            result = {
                key: wp.to_torch(getattr(self._shadow, key)).clone()
                for key in ("qpos", "qvel", "time", "xpos", "xmat", "xaxis", "xanchor")
            }
            if any(not bool(torch.isfinite(value).all()) for value in result.values()):
                raise ValueError("nonfinite current simulation geometry")
            if any(
                not torch.equal(wp.to_torch(getattr(self._live, key)), result[key])
                for key in ("qpos", "qvel", "time")
            ):
                raise RuntimeError("simulation advanced during kinematics capture")
        except Exception:
            self._invalid = True
            raise
        return result


class CpuKinematicsSnapshot:
    """Independent CPU MuJoCo FK with the same private-copy observation fields.

    The caller owns matching model/data identity and synchronous stepping.
    Native float64 values are retained; consumers explicitly choose conversion.
    Contact/solver state is never recomputed in the live simulation objects.
    """

    def __init__(self, model: Any, data: Any) -> None:
        import mujoco

        if (
            not isinstance(model, mujoco.MjModel)
            or not isinstance(data, (list, tuple))
            or not 1 <= len(data) <= 32
            or any(
                not isinstance(item, mujoco.MjData)
                or item.qpos.shape != (model.nq,)
                or item.qvel.shape != (model.nv,)
                for item in data
            )
        ):
            raise ValueError("explicit bounded CPU simulation model/data batch required")
        self._model = model
        self._live = tuple(data)
        self._shadow = tuple(mujoco.MjData(model) for _ in data)
        self._invalid = False

    def read(self) -> dict[str, Any]:
        import mujoco
        import numpy as np
        import torch

        if self._invalid:
            raise RuntimeError("simulation kinematics snapshot is invalid")
        try:
            fields = ("qpos", "qvel", "time", "xpos", "xmat", "xaxis", "xanchor")
            values: dict[str, list[Any]] = {key: [] for key in fields}
            for live, shadow in zip(self._live, self._shadow, strict=True):
                time = float(live.time)
                if not np.isfinite(time) or abs(time) > 1e6:
                    raise ValueError("invalid CPU simulation time")
                for key in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
                    item = getattr(live, key).copy()
                    if not np.isfinite(item).all() or (np.abs(item) > 1e6).any():
                        raise ValueError("invalid CPU simulation snapshot input")
                    getattr(shadow, key)[:] = item
                shadow.time = time
                mujoco.mj_kinematics(self._model, shadow)
                for key in fields:
                    item = np.asarray(getattr(shadow, key)).copy()
                    if key == "xmat":
                        item = item.reshape(self._model.nbody, 3, 3)
                    if not np.isfinite(item).all():
                        raise ValueError("nonfinite CPU simulation geometry")
                    values[key].append(item)
                if live.time != time or any(
                    not np.array_equal(getattr(live, key), getattr(shadow, key))
                    for key in ("qpos", "qvel")
                ):
                    raise RuntimeError("CPU simulation advanced during kinematics capture")
            return {key: torch.from_numpy(np.stack(items)) for key, items in values.items()}
        except Exception:
            self._invalid = True
            raise
