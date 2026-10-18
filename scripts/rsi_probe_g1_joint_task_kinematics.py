"""Fixed real G1 FK/Jacobian checks and bounded-task proposals, zero integration.

No claim of balance, contact exit, controller integration or trained skill.
The ten actual states include foot contact and later pinning/nonfoot contact.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

FRAMES = (0, 74, 75, 80, 85, 90, 99, 100, 101, 110)


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    output = Path(cfg["output"])
    if (
        output.exists()
        or output.with_suffix(".started.json").exists()
        or cfg["frames"] != list(FRAMES)
    ):
        raise ValueError("fixed ten source poses and unused diagnostic output required")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("clean frozen source required")
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    from rosclaw_soccer.physics.bounded_joint_tasks import (
        LinearJointTask,
        solve_bounded_joint_tasks,
    )
    from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from scripts.rsi_atomic_artifacts import write_once

    path = Path(cfg["receipt"])
    r = json.loads(path.read_text())
    if (
        digest(path) != cfg["receipt_file_hash"]
        or r["receipt_hash"] != cfg["receipt_hash"]
        or r["receipt_hash"] != hash_json({k: v for k, v in r.items() if k != "receipt_hash"})
    ):
        raise ValueError("exact original unsafe episode required")
    pins[str(path)] = digest(path)
    for w in r["witnesses"].values():
        if digest(w["path"]) != w["sha256"]:
            raise ValueError("original witness bytes changed")
        pins[w["path"]] = w["sha256"]
    write_once(
        output.with_suffix(".started.json"),
        dict(
            source_head=cfg["source_head"],
            input_file_hashes=pins,
            original_receipt_hash=r["receipt_hash"],
            new_integrated_steps=0,
            automatic_retry_authorized=False,
        ),
    )
    try:
        model = mujoco.MjModel.from_binary_path(r["witnesses"]["world"]["path"])
        sampler = G1JointTaskKinematicSampler(
            model, expected_world_hash=r["witnesses"]["world"]["sha256"]
        )
        with np.load(r["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        qi = [int(model.jnt_qposadr[j]) for j in joints]
        vi = [int(model.jnt_dofadr[j]) for j in joints]
        ball = int(model.body("ball").id)
        bj = int(model.body_jntadr[ball])
        bq = int(model.jnt_qposadr[bj])
        bv = int(model.jnt_dofadr[bj])
        pelvis = int(model.body("pelvis").id)
        geoms = [int(model.geom(f"task_contact_surface_{s}").id) for s in range(2)]
        data = mujoco.MjData(model)
        perturbed = mujoco.MjData(model)
        rows = []

        def forward(d):
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)

        def measurements(d, local):
            return np.concatenate(
                [
                    d.subtree_com[pelvis],
                    *[
                        d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ p
                        for g, p in zip(geoms, local, strict=True)
                    ],
                ]
            )

        for frame in FRAMES:
            q, v = trace["canonical_qpos"][frame, 0], trace["canonical_qvel"][frame, 0]
            data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = q[:7], q[7:36], q[36:43]
            data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6] = v[:6], v[6:35], v[35:41]
            forward(data)
            before = {
                k: np.array(getattr(data, k), copy=True)
                for k in (
                    "qpos",
                    "qvel",
                    "ctrl",
                    "qacc_warmstart",
                    "xpos",
                    "xquat",
                    "geom_xpos",
                    "geom_xmat",
                    "qfrc_constraint",
                )
            }
            sample = sampler.sample(data)
            if any(not np.array_equal(getattr(data, k), a) for k, a in before.items()):
                raise ValueError("sampler mutated caller state or cached observations")
            analytic = np.vstack(
                [sample.com_jacobian_m_per_rad, *sample.foot_material_point_jacobian_m_per_rad]
            )
            numeric = np.zeros((9, 29))
            eps = 1e-6
            for column, address in enumerate(qi):
                values = []
                for direction in (-1, 1):
                    perturbed.qpos[:] = data.qpos
                    perturbed.qpos[address] += direction * eps
                    forward(perturbed)
                    values.append(measurements(perturbed, sample.foot_material_point_mesh_local_m))
                numeric[:, column] = (values[1] - values[0]) / (2 * eps)
            error = float(np.max(abs(analytic - numeric)))
            if error > 1e-7:
                raise ValueError("actual G1 held-point/COM Jacobian finite difference mismatch")
            com_delta = np.clip(
                sample.foot_origin_world_m[1] - sample.robot_subtree_com_world_m,
                [-0.02, -0.02, 0.0],
                [0.02, 0.02, 0.0],
            )
            com_jac = sample.com_jacobian_m_per_rad.copy()
            com_jac[2] = 0
            tasks = (
                LinearJointTask(
                    "swing_clearance",
                    sample.foot_material_point_jacobian_m_per_rad[0],
                    np.array([0.01, 0.0, 0.04]),
                    1.0,
                ),
                LinearJointTask(
                    "other_foot_hold", sample.foot_origin_jacobian_m_per_rad[1], np.zeros(3), 10.0
                ),
                LinearJointTask("com_xy", com_jac, com_delta, 2.0),
            )
            nominal = trace["joint_target_rad"][frame, 0]
            previous = trace["joint_target_rad"][max(frame - 1, 0), 0]
            proposal = solve_bounded_joint_tasks(
                sample.joint_position_rad,
                nominal,
                previous,
                sample.joint_limits_rad,
                tasks,
                observation_step_cap_rad=0.2,
                command_step_cap_rad=0.04,
                nominal_trust_rad=0.35,
                posture_cost=0.01,
            )
            nonlinear = None
            if proposal.target_rad is not None:
                perturbed.qpos[:] = data.qpos
                perturbed.qpos[qi] = proposal.target_rad
                forward(perturbed)
                old = measurements(data, sample.foot_material_point_mesh_local_m)
                new = measurements(perturbed, sample.foot_material_point_mesh_local_m)
                nonlinear = dict(
                    robot_com_displacement_m=(new[:3] - old[:3]).tolist(),
                    swing_material_point_displacement_m=(new[3:6] - old[3:6]).tolist(),
                    other_material_point_displacement_m=(new[6:9] - old[6:9]).tolist(),
                )
            rows.append(
                dict(
                    frame=frame,
                    maximum_held_point_com_jacobian_error_m_per_rad=error,
                    input_joint_positions_hash=hash_bytes(sample.joint_position_rad.tobytes()),
                    caller_state_unchanged=True,
                    proposal_status=proposal.status,
                    proposal_target_rad=proposal.target_rad,
                    predicted_residual_m=proposal.task_residual_m,
                    nonlinear_fk=nonlinear,
                    actual_ground_support_measured=False,
                    physically_qualified=False,
                )
            )
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source/input bytes changed")
        result = dict(
            schema="soccer.rsi.g1_joint_task_static_probe.v1",
            source_head=cfg["source_head"],
            input_file_hashes=pins,
            rows=rows,
            finite_difference_eps_rad=1e-6,
            held_material_points_not_closest_selection_map=True,
            actuated_joint_columns=29,
            root_or_ball_optimizer_columns=False,
            rotation_jacobians_finite_difference_qualified=False,
            new_integrated_steps=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            controller_integrated=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(output, result)
        print(result["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            output.with_suffix(".failure.json"),
            dict(type=type(error).__name__, message=str(error), new_integrated_steps=0),
        )
        raise


if __name__ == "__main__":
    main()
