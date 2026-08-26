"""Explicit data bridges from Motion Atlas clips to external learner formats."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.athlete_foundation.full_body_goalkeeper_motion import (
    load_full_body_goalkeeper_motion_library,
)


def export_humanoid_gpt_reference(
    *,
    motion_manifest_path: Path,
    family: str,
    output_path: Path,
    source_checkout: Path,
) -> dict[str, Any]:
    """Export one 36-qpos clip for Humanoid-GPT's safe ``--convert`` path."""

    target = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if target == checkout or checkout in target.parents:
        raise ValueError("external backend input must remain outside the source checkout")
    if target.suffix != ".npz" or target.exists():
        raise ValueError("Humanoid-GPT adapter requires a new NPZ output")
    library = load_full_body_goalkeeper_motion_library(motion_manifest_path)
    if family not in library.families:
        raise ValueError("Humanoid-GPT adapter motion family is invalid")
    frame_count = dict(library.manifest.family_frame_counts)[family]
    times = np.arange(frame_count, dtype=np.float64) / library.manifest.source_frame_rate_hz
    qpos = np.zeros((frame_count, 36), dtype=np.float32)
    initial = library.sample(family, time_sec=0.0)
    initial_xyzw = initial.root_quaternion_xyzw
    initial_yaw = math.atan2(
        2.0 * (initial_xyzw[3] * initial_xyzw[2] + initial_xyzw[0] * initial_xyzw[1]),
        1.0 - 2.0 * (initial_xyzw[1] ** 2 + initial_xyzw[2] ** 2),
    )
    yaw_inverse = np.asarray(
        (0.0, 0.0, -math.sin(0.5 * initial_yaw), math.cos(0.5 * initial_yaw)),
        dtype=np.float64,
    )
    cosine = math.cos(initial_yaw)
    sine = math.sin(initial_yaw)
    for index, time_sec in enumerate(times):
        frame = library.sample(family, time_sec=float(time_sec))
        position = frame.root_position_local
        qpos[index, 0] = cosine * position[0] + sine * position[1]
        qpos[index, 1] = -sine * position[0] + cosine * position[1]
        qpos[index, 2] = 0.80 + position[2] - initial.root_position_local[2]
        xyzw = _quaternion_multiply(yaw_inverse, frame.root_quaternion_xyzw)
        qpos[index, 3:7] = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
        qpos[index, 7:] = frame.qpos_29
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        qpos=qpos,
        frequency=np.asarray(library.manifest.source_frame_rate_hz, dtype=np.float32),
    )
    payload: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.humanoid_gpt_reference_adapter.v1",
        "backend_id": "humanoid-gpt",
        "family": family,
        "archive_file": target.name,
        "archive_hash": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_motion_manifest_hash": library.manifest.manifest_hash,
        "frame_count": frame_count,
        "frame_rate_hz": library.manifest.source_frame_rate_hz,
        "qpos_order": "root_xyz_root_quaternion_wxyz_g1_29dof",
        "root_frame_normalization": "INITIAL_YAW_REMOVED_XY_ROTATED_BASE_HEIGHT_0.80M",
        "unobserved_joint_policy": "EXPLICIT_NEUTRAL",
        "source_joint_mask": list(library.manifest.source_joint_mask),
        "research_only": True,
        "commercial_use_allowed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["adapter_report_hash"] = hash_json(payload)
    report_path = target.with_suffix(".json")
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def export_opentrack_reference(
    *,
    canonical_reference_path: Path,
    output_path: Path,
    model_xml_path: Path,
    source_checkout: Path,
    family: str,
) -> dict[str, Any]:
    """Export a canonical 36-qpos reference in OpenTrack's native trajectory schema.

    The bridge deliberately writes an *incomplete* OpenTrack trajectory containing
    only joint state.  OpenTrack then owns forward-kinematics completion and its
    smooth start/end transition, which prevents the Soccer integration from
    silently inventing task-space labels in a different robot model.
    """

    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - optional simulator boundary
        raise RuntimeError("OpenTrack export requires the optional mujoco package") from error

    source = canonical_reference_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    model_path = model_xml_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if target == checkout or checkout in target.parents:
        raise ValueError("external backend input must remain outside the source checkout")
    if target.suffix != ".npz" or target.exists():
        raise ValueError("OpenTrack adapter requires a new NPZ output")
    if not source.is_file() or not model_path.is_file():
        raise FileNotFoundError("canonical reference and OpenTrack model XML must exist")

    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"frequency", "qpos"}:
            raise ValueError("canonical reference has unexpected keys")
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        frequency = float(np.asarray(archive["frequency"]).item())
    if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] != 36:
        raise ValueError("OpenTrack G1 reference must be shaped [frames, 36]")
    if not np.isfinite(qpos).all() or not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("OpenTrack reference must contain finite values and a positive frequency")
    quaternion_norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.max(np.abs(quaternion_norms - 1.0)) > 1.0e-3:
        raise ValueError("OpenTrack reference root quaternions must be normalized")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv, model.njnt, model.nu) != (36, 35, 30, 29):
        raise ValueError("OpenTrack model is not the expected Unitree G1 29-DoF model")
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    if any(name is None for name in joint_names):
        raise ValueError("OpenTrack model contains unnamed joints")

    dt = 1.0 / frequency
    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    for index in range(1, qpos.shape[0]):
        mujoco.mj_differentiatePos(
            model,
            qvel[index],
            dt,
            qpos[index - 1],
            qpos[index],
        )
    qvel[0] = qvel[1]

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        frequency=np.asarray(frequency, dtype=np.float64),
        joint_names=np.asarray(joint_names, dtype=np.str_),
        njnt=np.asarray(model.njnt, dtype=np.int64),
        jnt_type=np.asarray(model.jnt_type, dtype=np.int32),
        qpos=qpos,
        qvel=qvel,
        split_points=np.asarray((0, qpos.shape[0]), dtype=np.int32),
    )
    payload: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.opentrack_reference_adapter.v1",
        "backend_id": "opentrack",
        "family": family,
        "archive_file": target.name,
        "archive_hash": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest(),
        "canonical_reference_hash": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
        "frame_count": int(qpos.shape[0]),
        "frame_rate_hz": frequency,
        "qpos_order": "root_xyz_root_quaternion_wxyz_g1_29dof",
        "qvel_method": "MUJOCO_DIFFERENTIATE_POS",
        "completion_owner": "OPENTRACK_FORWARD_KINEMATICS_AND_TRANSITION_SMOOTHER",
        "research_only": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["adapter_report_hash"] = hash_json(payload)
    target.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    value = np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


__all__ = ["export_humanoid_gpt_reference", "export_opentrack_reference"]
