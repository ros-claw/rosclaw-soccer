"""Read-only G1 motion/body compatibility, kinematics only, no physics steps.

Motion category labels and reference poses are not physical football teachers.
Report source units, named joints, limits and floor alignment before tracking.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_motion(path, world):
    expected = (
        "root_pos_x(m)",
        "root_pos_y(m)",
        "root_pos_z(m)",
        "root_rot_w",
        "root_rot_x",
        "root_rot_y",
        "root_rot_z",
    )
    expected += tuple(f"dof_{name}(rad)" for name in G1_DDS_JOINT_NAMES)
    with path.open(newline="") as stream:
        if tuple(next(csv.reader(stream))) != expected:
            raise ValueError("named 29-joint radians, meter root and explicit wxyz CSV required")
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if not 2 <= len(values) <= 200000 or values.shape[1] != 36 or not np.isfinite(values).all():
        raise ValueError("bounded complete finite motion clip required")
    error = float(np.max(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1)))
    if error > 1e-3:
        raise ValueError("source quaternion norm mismatch; do not silently renormalize")
    model = mujoco.MjModel.from_binary_path(str(world))
    if (model.nq, model.nv, model.nu) != (43, 41, 29):
        raise ValueError("named G1-plus-ball diagnostic world required")
    joints = np.asarray([model.joint(name).id for name in G1_DDS_JOINT_NAMES])
    qi = model.jnt_qposadr[joints]
    floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if (
        len(floor) != 1
        or model.geom_bodyid[floor[0]] != 0
        or not np.array_equal(model.geom_quat[floor[0]], [1, 0, 0, 0])
    ):
        raise ValueError("single unrotated world-floor plane required for clearance diagnostic")
    floor_z = float(model.geom_pos[floor[0], 2])
    limited = model.jnt_limited[joints].astype(bool)
    outside = limited[None] & (
        (values[:, 7:] < model.jnt_range[joints, 0]) | (values[:, 7:] > model.jnt_range[joints, 1])
    )
    feet = {}
    for side in ("left", "right"):
        body = model.body(f"{side}_ankle_roll_link").id
        ids = np.flatnonzero(
            (model.geom_bodyid == body)
            & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
            & (model.geom_contype != 0)
        )
        if len(ids) != 4 or not np.all(model.geom_size[ids, 0] == 0.005):
            raise ValueError(
                "four original 5mm support spheres required, not inferred sole geometry"
            )
        feet[side] = ids
    data = mujoco.MjData(model)
    heights = {side: [] for side in feet}
    for row in values:
        data.qpos[:7] = row[:7]
        data.qpos[qi] = row[7:]
        mujoco.mj_kinematics(model, data)
        for side, ids in feet.items():
            heights[side].append(
                float(np.min(data.geom_xpos[ids, 2] - model.geom_size[ids, 0] - floor_z))
            )
    return dict(
        frames=len(values),
        columns=36,
        canonical_joint_count=29,
        root_position_units="meters per CSV headers, not generic README millimeters",
        joint_angle_units="radians",
        quaternion_order="wxyz",
        maximum_quaternion_norm_error=error,
        time_step_embedded_in_csv=False,
        source_fps_not_inferred=True,
        source_root_height_min_m=float(np.min(values[:, 2])),
        source_root_height_max_m=float(np.max(values[:, 2])),
        joint_limit_violations_per_name={
            name: int(np.count_nonzero(outside[:, i])) for i, name in enumerate(G1_DDS_JOINT_NAMES)
        },
        frames_with_any_joint_limit_violation=int(np.count_nonzero(np.any(outside, axis=1))),
        original_support_sphere_clearance_m={
            side: dict(
                minimum=float(np.min(h)),
                fifth_percentile=float(np.percentile(h, 5)),
                median=float(np.median(h)),
                frames_below_floor_by_more_than_1cm=int(np.count_nonzero(np.asarray(h) < -0.01)),
            )
            for side, h in heights.items()
        },
        floor_height_m=floor_z,
        source_root_pose_modified=False,
        joint_values_clipped=False,
        quaternions_renormalized=False,
        used_mj_kinematics_only=True,
        new_physics_substeps=0,
        neural_model_calls=0,
        optimizer_updates=0,
        physical_teacher_qualified=False,
        hardware_authorized=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-csv", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--dataset-readme", type=Path, required=True)
    parser.add_argument("--dataset-license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("unused asset/body audit output required")
    paths = (args.motion_csv, args.world, args.dataset_readme, args.dataset_license, Path(__file__))
    pins = {str(p): digest(p) for p in paths}
    result = inspect_motion(args.motion_csv, args.world)
    if any(digest(Path(p)) != value for p, value in pins.items()):
        raise ValueError("asset/body/source changed during read-only audit")
    result.update(
        schema="soccer.rsi.motion_csv_body_compatibility.v1",
        input_file_hashes=pins,
        source_license_review_required_before_publicity=True,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
