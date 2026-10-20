"""Bounded kinematic audit of actual reference70 inputs and ball placement.

No neural inference, time integration, motor execution or policy promotion.
Reference body poses exist only in a separate scratch MjData.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("review-root", "export-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("review-unit", "review-hash", "export-unit", "export-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean frozen source required")
    inputs = []
    pins = {}
    for root, unit, expected in (
        (a.review_root, a.review_unit, a.review_hash),
        (a.export_root, a.export_unit, a.export_hash),
    ):
        wait_for_terminal_collection(unit)
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("immutable actual completed prerequisites required")
        inputs.append(json.loads(path.read_text()))
        pins[str(path)] = expected
    review, exported = inputs
    if (
        review["status"] != "MJLAB_PAIRED_DEV_MOTION_TRACKING_NOT_FOOTBALL_QUALIFIED"
        or not review["stable_prior_rehearsal"]
        or not review["all_initial_states_paired_exact"]
        or exported["status"] != "MJLAB_NATIVE_EXPORT_ONE_CONTROL_DIAGNOSTIC_NOT_LONG_RUN_QUALIFIED"
    ):
        raise ValueError("actual paired reference policy review and original body export required")
    pins[str(a.export_root / "compiled_model.mjb")] = exported["model_hash"]
    for index in range(2):
        result = next(
            r
            for r in review["results"]
            if r["source"].startswith(f"soccer-standard-{(7, 10)[index]:03d}_")
            and r["arm"] == "initial"
        )
        pins[str(a.review_root / f"{index}-initial-raw.npz")] = result["raw_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            public_dev_sources=[7, 10],
            arm="initial",
            max_native_kinematic_calls=90000,
            max_source_rows=30000,
            reference_channel_error_limit=1e-5,
            scratch_reference_pose_not_physics=True,
            new_physics_steps=0,
            new_neural_calls=0,
            new_optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    calls = 0
    seen = 0
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.native_motion_reference import reference_inputs70
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        model = mujoco.MjModel.from_binary_path(str(a.export_root / "compiled_model.mjb"))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        torso = model.body("robot/torso_link").id
        foot_bodies = [
            model.body("robot/" + side + "_ankle_roll_link").id for side in ("left", "right")
        ]
        foot_geoms = [
            np.flatnonzero(
                (model.geom_bodyid == body)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            ).tolist()
            for body in foot_bodies
        ]
        ball_geom = model.geom("ball/ball_geom").id
        if any(not ids for ids in foot_geoms):
            raise ValueError("actual collidable foot geometry required")
        measured, scratch = mujoco.MjData(model), mujoco.MjData(model)
        results = []
        for index in range(2):
            with np.load(a.review_root / f"{index}-initial-raw.npz", allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            maximum_suffix = maximum_command = 0.0
            minimum_gap = np.full((32, 2), np.inf)
            anchor_error = 0.0
            for frame in range(len(raw["qpos_before"])):
                for world in range(32):
                    if not raw["active_before"][frame, world]:
                        continue
                    seen += 1
                    measured.qpos[:] = raw["qpos_before"][frame, world]
                    mujoco.mj_kinematics(model, measured)
                    calls += 1
                    command, suffix = reference_inputs70(
                        root_rotation=measured.xmat[layout.pelvis].reshape(3, 3),
                        anchor_position=measured.xpos[torso],
                        anchor_rotation=measured.xmat[torso].reshape(3, 3),
                        joint_position=raw["reference_joint"][frame, world],
                        joint_velocity=raw["reference_joint_velocity"][frame, world],
                        reference_position=raw["reference_anchor"][frame, world],
                        reference_quaternion=raw["reference_anchor_quat"][frame, world],
                        reference_linear_velocity=raw["reference_anchor_linear_velocity"][
                            frame, world
                        ],
                        reference_angular_velocity=raw["reference_anchor_angular_velocity"][
                            frame, world
                        ],
                    )
                    maximum_suffix = max(
                        maximum_suffix,
                        float(np.abs(suffix - raw["observation"][frame, world, 98:]).max()),
                    )
                    maximum_command = max(
                        maximum_command,
                        float(np.abs(command - raw["observation"][frame, world, 6:9]).max()),
                    )
                    if max(maximum_suffix, maximum_command) > 1e-5:
                        np.savez_compressed(
                            a.output / "mismatched-reference.npz",
                            expected_suffix=suffix,
                            actual_suffix=raw["observation"][frame, world, 98:],
                            expected_command=command,
                            actual_command=raw["observation"][frame, world, 6:9],
                            frame=frame,
                            world=world,
                        )
                        raise ValueError(
                            "actual reference channels disagree with independent native geometry"
                        )
                    scratch.qpos[:] = measured.qpos
                    scratch.qpos[layout.rq : layout.rq + 7] = [0, 0, 0, 1, 0, 0, 0]
                    scratch.qpos[layout.qi] = raw["reference_joint"][frame, world]
                    # Keep the episode's initially declared ball location, not
                    # its later physically observed trajectory.
                    scratch.qpos[layout.bq : layout.bq + 7] = raw["qpos_before"][
                        0, world, layout.bq : layout.bq + 7
                    ]
                    mujoco.mj_kinematics(model, scratch)
                    calls += 1
                    offset = scratch.xpos[torso].copy()
                    reference_rot = np.zeros(9)
                    mujoco.mju_quat2Mat(
                        reference_rot, raw["reference_anchor_quat"][frame, world].astype(np.float64)
                    )
                    root_rotation = (
                        reference_rot.reshape(3, 3) @ scratch.xmat[torso].reshape(3, 3).T
                    )
                    root_quaternion = np.zeros(4)
                    mujoco.mju_mat2Quat(root_quaternion, root_rotation.reshape(9))
                    scratch.qpos[layout.rq : layout.rq + 3] = (
                        raw["reference_anchor"][frame, world] - root_rotation @ offset
                    )
                    scratch.qpos[layout.rq + 3 : layout.rq + 7] = root_quaternion
                    mujoco.mj_kinematics(model, scratch)
                    calls += 1
                    anchor_error = max(
                        anchor_error,
                        float(
                            np.linalg.norm(
                                scratch.xpos[torso] - raw["reference_anchor"][frame, world]
                            )
                        ),
                    )
                    for foot, ids in enumerate(foot_geoms):
                        gap = min(
                            mujoco.mj_geomDistance(model, scratch, g, ball_geom, 100, None)
                            for g in ids
                        )
                        minimum_gap[world, foot] = min(minimum_gap[world, foot], gap)
                    if calls > 90000 or seen > 30000:
                        raise ValueError("predeclared nonphysical audit budget exceeded")
            position = raw["reference_anchor"][:, 0, :]
            velocity = raw["reference_anchor_linear_velocity"][:, 0, :]
            integral = (velocity[:-1] + velocity[1:]).sum(axis=0) * 0.01
            results.append(
                dict(
                    source=(7, 10)[index],
                    maximum_reference70_error=maximum_suffix,
                    maximum_velocity_command_error=maximum_command,
                    scratch_anchor_position_error_m=anchor_error,
                    reference_anchor_displacement_m=(position[-1] - position[0]).tolist(),
                    integral_reference_velocity_m=integral.tolist(),
                    velocity_integral_endpoint_error_m=float(
                        np.linalg.norm(position[-1] - position[0] - integral)
                    ),
                    minimum_planned_foot_ball_geometry_gap_m=minimum_gap.tolist(),
                    worlds_with_planned_geometry_overlap=int((minimum_gap.min(axis=1) <= 0).sum()),
                )
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_REFERENCE_GEOMETRY_AUDIT_NOT_PHYSICAL_CONTACT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                source_rows=seen,
                native_kinematic_calls=calls,
                new_physics_steps=0,
                new_neural_calls=0,
                new_optimizer_steps=0,
                hardware_authorized=False,
                goal_complete=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                source_rows=seen,
                native_kinematic_calls=calls,
                new_physics_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
