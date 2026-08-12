"""Validated trajectory sampling primitives for evidence-downstream media."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024
Array: TypeAlias = NDArray[np.generic]


def load_g1_ball_trajectory(path: Path) -> dict[str, Array]:
    """Load a bounded no-pickle trajectory and validate its render contract."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("G1 render trajectory is missing")
    if not 1 <= resolved.stat().st_size <= _MAX_TRAJECTORY_BYTES:
        raise ValueError("G1 render trajectory is empty or exceeds the size limit")
    with np.load(resolved, allow_pickle=False) as archive:
        value = {name: archive[name] for name in archive.files}
    for name, shape in {
        "time": (),
        "pelvis_pose": (7,),
        "joint_position": (29,),
        "ball_pose": (7,),
    }.items():
        array = np.asarray(value.get(name))
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
            raise ValueError(f"G1 render trajectory {name} has invalid shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"G1 render trajectory {name} is non-finite")
    lengths = {
        len(np.asarray(value[name]))
        for name in ("time", "pelvis_pose", "joint_position", "ball_pose")
    }
    if len(lengths) != 1:
        raise ValueError("G1 render trajectory arrays have inconsistent lengths")
    time = np.asarray(value["time"], dtype=np.float64)
    if len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("G1 render trajectory time must be strictly increasing")
    for name in ("pelvis_pose", "ball_pose"):
        quaternion_norm = np.linalg.norm(np.asarray(value[name])[:, 3:], axis=1)
        if np.any(quaternion_norm <= 1e-12):
            raise ValueError(f"G1 render trajectory {name} contains a zero quaternion")
    return value


def sample_g1_ball_trajectory(
    trajectory: dict[str, Array],
    simulation_time: float,
) -> tuple[int, Array, Array, Array]:
    """Interpolate evidence poses for smooth playback without changing physics."""

    times = np.asarray(trajectory["time"], dtype=np.float64)
    upper = int(np.searchsorted(times, simulation_time, side="right"))
    if upper <= 0:
        index = 0
        ratio = 0.0
        upper = 0
    elif upper >= len(times):
        index = len(times) - 1
        ratio = 0.0
        upper = index
    else:
        index = upper - 1
        ratio = float((simulation_time - times[index]) / (times[upper] - times[index]))
    pelvis = _interpolate_pose(
        trajectory["pelvis_pose"][index],
        trajectory["pelvis_pose"][upper],
        ratio,
    )
    joints = _lerp(
        trajectory["joint_position"][index],
        trajectory["joint_position"][upper],
        ratio,
    )
    ball = _interpolate_pose(
        trajectory["ball_pose"][index],
        trajectory["ball_pose"][upper],
        ratio,
    )
    trail_index = upper if ratio >= 0.5 else index
    return trail_index, pelvis, joints, ball


def append_sphere(
    mujoco: object,
    scene: object,
    position: Array,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Append one visualization-only sphere when the scene has capacity."""

    if scene.ngeom >= scene.maxgeom:  # type: ignore[attr-defined]
        return
    mujoco.mjv_initGeom(  # type: ignore[attr-defined]
        scene.geoms[scene.ngeom],  # type: ignore[attr-defined]
        mujoco.mjtGeom.mjGEOM_SPHERE,  # type: ignore[attr-defined]
        np.asarray((radius,) * 3, dtype=np.float64),
        position,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1  # type: ignore[attr-defined]


def escape_filtergraph_option(value: str) -> str:
    """Apply FFmpeg option-value and filtergraph escaping layers."""

    def escape_level(text: str, special: str) -> str:
        return "".join(
            ("\\" + character) if character in special else character for character in text
        )

    return escape_level(escape_level(value, "\\':"), "\\'[],;")


def _interpolate_pose(left: Array, right: Array, ratio: float) -> Array:
    result: NDArray[np.float64] = np.empty(7, dtype=np.float64)
    result[:3] = _lerp(left[:3], right[:3], ratio)
    result[3:] = _slerp_wxyz(left[3:], right[3:], ratio)
    return result


def _lerp(left: Array, right: Array, ratio: float) -> Array:
    return np.asarray(left, dtype=np.float64) + ratio * (
        np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    )


def _slerp_wxyz(left: Array, right: Array, ratio: float) -> Array:
    start = np.asarray(left, dtype=np.float64)
    end = np.asarray(right, dtype=np.float64)
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = start + ratio * (end - start)
        return np.asarray(value / np.linalg.norm(value), dtype=np.float64)
    angle = float(np.arccos(dot))
    scale = float(np.sin(angle))
    return np.asarray(
        np.sin((1.0 - ratio) * angle) / scale * start + np.sin(ratio * angle) / scale * end,
        dtype=np.float64,
    )


__all__ = [
    "append_sphere",
    "escape_filtergraph_option",
    "load_g1_ball_trajectory",
    "sample_g1_ball_trajectory",
]
