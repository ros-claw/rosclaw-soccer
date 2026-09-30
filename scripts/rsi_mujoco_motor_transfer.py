"""Actual CPU MuJoCo closed-loop transfer probe, not an Isaac evidence replay.

No robot transport or vendor SDK is imported. Asset differences remain explicit;
this diagnostic cannot authorize promotion or physical execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker
from rosclaw_soccer.rsi.contact_motor_contract import load_policy, motor_delta
from rosclaw_soccer.rsi.contact_time_phase_features import (
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.rsi.online_motor_actor_critic import configure_preview, validate_model
from rosclaw_soccer.rsi.taskspace_gate_memory import select_taskspace_gate
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    choose_swing_side,
    release_joint_delta,
    swing_joint_delta,
)
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def main() -> None:
    import mujoco
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "model-root", "late-swing-policy", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--neural-model", type=Path)
    group.add_argument("--motor-policy", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lane", type=int, required=True, choices=range(16))
    args = parser.parse_args()
    # This first diagnostic only accepts already consumed courses.
    consumed = {
        (20261177, 0),
        (20261227, 4),
        (20261282, 0),
        (20261360, 0),
        (20261378, 0),
        (20261440, 2),
        (20261446, 0),
        (20261095, 2),
        (20261097, 2),
        (20261146, 4),
        (20261148, 2),
        (20260975, 4),
    }
    if (args.seed, args.lane) not in consumed:
        parser.error("transfer diagnostic cannot open a fresh course")
    torch.set_num_threads(1)
    scene = args.scene.resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002

    def identifier(kind: Any, name: str) -> int:
        value = int(mujoco.mj_name2id(model, kind, name))
        if value < 0:
            raise ValueError(f"missing simulation element {name}")
        return value

    names = list(G1_DDS_JOINT_NAMES)
    joints = [identifier(mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qi = np.asarray([model.jnt_qposadr[j] for j in joints], dtype=int)
    vi = np.asarray([model.jnt_dofadr[j] for j in joints], dtype=int)
    ai = np.asarray([identifier(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names], dtype=int)
    if model.nu != 29 or not np.array_equal(model.actuator_trnid[ai, 0], joints):
        raise ValueError("canonical named 29-joint torque contract required")
    pelvis = identifier(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
    ]
    bodies = [identifier(mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
    contact_bodies = (
        bodies[:2]
        + [
            identifier(mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link"),
            identifier(mujoco.mjtObj.mjOBJ_BODY, "right_ankle_pitch_link"),
        ]
        + bodies[2:]
    )
    ball = identifier(mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom = identifier(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    bj = int(model.body_jntadr[ball])
    bq, bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
    # Explicit physics adaptation, not an assertion that the XML equals USD.
    model.geom_size[ball_geom, 0] = 0.11
    model.body_mass[ball] = 0.43
    model.body_inertia[ball] = 0.4 * 0.43 * 0.11**2
    model.dof_damping[bv : bv + 6] = 0
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    if box >= 0:
        mask = model.geom_bodyid == box
        model.geom_contype[mask] = 0
        model.geom_conaffinity[mask] = 0
    mujoco.mj_setConst(model, data)
    navigation = G1SonicNavigation(
        args.model_root,
        "cpu.first_touch.0",
        SonicNavigationConfig(
            maximum_frames=300,
            planner_seed=30300,
            model_variant="low_latency",
            experimental_maximum_speed_mps=1.5,
            inference_threads=1,
        ),
    )
    navigation.backend.qualification.require_eligible()
    foundation = FrozenSonicG1Torch(args.model_root, variant="low_latency", device="cpu")
    late = load_late_swing_actor(args.late_swing_policy)
    neural = json.loads(args.neural_model.read_text()) if args.neural_model else None
    if neural:
        validate_model(neural)
    policy, knots = load_policy(args.motor_policy) if args.motor_policy else (None, None)
    x, y, vx = sample_training_courses(args.seed, 16)[args.lane]
    data.qpos[:7] = (0, 0, 0.793, 1, 0, 0, 0)
    data.qpos[qi] = navigation.backend.default_angles
    data.qpos[bq : bq + 7] = (x, y, 0.13, 1, 0, 0, 0)
    data.qvel[bv : bv + 6] = (vx, 0, 0, 0, vx / 0.11, 0)
    mujoco.mj_forward(model, data)
    args.output_root.mkdir(parents=True, exist_ok=False)
    assets = {
        str(p.relative_to(scene.parent)): hash_bytes(p.read_bytes())
        for p in sorted(scene.parent.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png", ".jpg"}
    }
    commitment = dict(
        schema="soccer.rsi.cpu_motor_transfer.v1",
        seed=args.seed,
        lane=args.lane,
        course=[x, y, vx],
        partition="CONSUMED_TRANSFER_DIAGNOSTIC",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        assets=assets,
        foundation_hash=navigation.backend.qualification.qualification_hash,
        late_actor_hash=late["actor_hash"],
        model_hash=neural["model_hash"] if neural else None,
        motor_policy_hash=policy["policy_hash"] if policy else None,
        physics=dict(
            engine="MuJoCo",
            version=mujoco.__version__,
            device="cpu",
            dt_s=0.002,
            control_dt_s=0.02,
            ball_radius_m=0.11,
            ball_mass_kg=0.43,
            ball_free_damping=0.0,
            auxiliary_box_collision_disabled=True,
            robot_xml_not_usd=True,
            contact_settings="original XML pairs retained",
        ),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    (args.output_root / "commitment.json").write_text(json.dumps(commitment, indent=2))
    history: dict[str, list[Any]] = {
        k: []
        for k in (
            "root_pose_xyzw_m",
            "root_velocity_world",
            "ball_position_before_step_m",
            "ball_linear_velocity_before_step_m_s",
            "foot_geometry_position_before_step_m",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "joint_target_rad",
            "force_n",
            "ball_position_after_step_m",
            "torque_nm",
            "pre_motor_joint_target_rad",
            "motor_delta_rad",
            "navigation_command",
            "actual_actuator_force_nm",
        )
    }
    tracker = None
    tracking = y < 0
    gate = False
    side, contact_frame = -1, None
    swing_contact_delta = np.zeros(6)
    previous, motor_contact_delta = np.zeros(12), np.zeros(12)
    leg_ids = [np.asarray([names.index(n) for n in row]) for row in LEG_NAMES]
    motor_ids = np.concatenate(leg_ids)
    limits = model.jnt_range[joints].copy()
    minimum_pelvis = float("inf")
    first_contact_frame = None
    first_contact_bodies: list[int] = []
    for frame in range(300):
        root_pose = np.concatenate((data.xpos[pelvis], data.xquat[pelvis][[1, 2, 3, 0]]))
        root_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, root_vel, 0)
        root_vel = root_vel[[3, 4, 5, 0, 1, 2]]
        ball_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ball, ball_vel, 0)
        for key, value in (
            ("root_pose_xyzw_m", root_pose),
            ("root_velocity_world", root_vel),
            ("ball_position_before_step_m", data.xpos[ball]),
            ("ball_linear_velocity_before_step_m_s", ball_vel[3:]),
            ("foot_geometry_position_before_step_m", data.xpos[bodies]),
            ("joint_position_rad", data.qpos[qi]),
            ("joint_velocity_rad_s", data.qvel[vi]),
        ):
            history[key].append(value.copy()[None])
        gap = data.xpos[ball] - data.xpos[pelvis]
        lateral = (
            float(np.clip(1.2 * gap[1], -0.2, 0.2))
            if tracking and gap[0] > 0.95 and contact_frame is None
            else 0.0
        )
        qpos = np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
        qvel = np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6]))
        observation = TeamMotorObservation(
            agent_id=navigation.agent_id,
            frame=frame,
            time_sec=frame * 0.02,
            intent="other",
            prospective_owner=False,
            qpos=tuple(float(v) for v in qpos),
            qvel=tuple(float(v) for v in qvel),
            target_position_m=(0.0, 0.0, 0.0),
            navigation_command=(1.4, lateral, 0.0),
            navigation_envelope=navigation.navigation_envelope,
        )
        if frame == 0:
            navigation.start_from_observation(observation)
        navigation.prepare_batched_proposal(observation)
        if tracker is None:
            tracker = BatchedSonicTracker(
                foundation,
                navigation.backend.reference[None],
                low_latency_legacy_encoder_layout=True,
            )
            tracker.reset(qpos[None], qvel[None])
        else:
            tracker.observe(qpos[None], qvel[None])
            if frame % 20 == 0:
                tracker.refresh_unexecuted_reference(
                    frame, navigation.backend.reference[None], unchanged_lookahead_frames=10
                )
        tracker.update(frame, qpos[None], qvel[None])
        proposal = navigation.commit_batched_action(tracker.action[0].numpy())
        target = np.asarray(proposal.target_rad).copy()
        if frame == 30:
            raw = current_context(
                root_pose[None], root_vel[None], data.xpos[ball][None], ball_vel[None, 3:]
            )
            features = gait_phase_features(
                raw, predict_contact_time(raw, np.asarray(late["contact_time_weights"]))
            )
            gate = bool(
                select_taskspace_gate(
                    features,
                    np.asarray(late["memory_features"]),
                    np.asarray(late["memory_clean"]),
                    np.asarray(late["memory_reward"]),
                    np.asarray(late["memory_groups"]),
                    neighbors=late["neighbors"],
                    confidence=late["confidence"],
                    baseline_clean_ceiling=late["baseline_clean_ceiling"],
                )[0]
                and vx < 0
            )
            if neural:
                policy = configure_preview(neural, history, contact_frame)
                knots = np.asarray(policy["knots_rad"])
        swing_delta = np.zeros(6)
        if gate:
            if contact_frame is None:
                side = choose_swing_side(
                    data.xpos[bodies[:2]],
                    data.xpos[ball],
                    side,
                    acquisition_max_gap_m=0.55,
                    revalidate_swing_side=False,
                )
            if side >= 0:
                ids = leg_ids[side]
                if contact_frame is not None:
                    swing_delta = release_joint_delta(swing_contact_delta, frame - contact_frame)
                else:
                    jac = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, jac, None, bodies[side])
                    swing_delta = swing_joint_delta(
                        data.xpos[bodies[side]],
                        data.xpos[ball],
                        jac[:, vi[ids]],
                        target[ids],
                        limits[ids],
                        forward_cap_m=0.08,
                        lateral_cap_m=0.05,
                        lateral_lead_m=0.0,
                        vertical_offset_m=0.04,
                    )
                target[ids] += swing_delta
        history["pre_motor_joint_target_rad"].append(target.copy()[None])
        if policy is not None:
            if knots is None:
                raise ValueError("motor policy requires authenticated knots")
            previous = motor_delta(
                policy,
                knots,
                float(gap[0]),
                target[motor_ids],
                limits[motor_ids],
                previous,
                motor_contact_delta,
                frame - contact_frame if contact_frame is not None else None,
            )
            target[motor_ids] += previous
        history["motor_delta_rad"].append(previous.copy()[None])
        history["navigation_command"].append(np.asarray((1.4, lateral, 0.0))[None])
        history["joint_target_rad"].append(target[None].copy())
        forces = np.zeros(6)
        torques = []
        actual_forces = []
        for _ in range(10):
            torque = np.clip(
                (target - data.qpos[qi]) * navigation.backend.kp
                - data.qvel[vi] * navigation.backend.kd,
                -np.asarray(G1_HARD_TORQUE_LIMITS),
                np.asarray(G1_HARD_TORQUE_LIMITS),
            )
            data.ctrl[ai] = torque
            torques.append(torque.copy())
            mujoco.mj_step(model, data)
            actual_forces.append(data.actuator_force[ai].copy())
            for c in range(data.ncon):
                contact = data.contact[c]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if ball_geom not in (g1, g2):
                    continue
                other = g2 if g1 == ball_geom else g1
                body = int(model.geom_bodyid[other])
                if body not in contact_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, force)
                index = contact_bodies.index(body)
                forces[index] = max(forces[index], float(np.linalg.norm(force[:3])))
            minimum_pelvis = min(minimum_pelvis, float(data.xpos[pelvis, 2]))
        if contact_frame is None and np.any(forces > 1):
            contact_frame = frame
            first_contact_frame = frame
            first_contact_bodies = np.flatnonzero(forces > 1).tolist()
            swing_contact_delta = swing_delta.copy()
            motor_contact_delta = previous.copy()
        history["force_n"].append(forces[None])
        history["ball_position_after_step_m"].append(data.xpos[ball].copy()[None])
        history["torque_nm"].append(np.asarray(torques)[None])
        history["actual_actuator_force_nm"].append(np.asarray(actual_forces)[None])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("nonfinite physical MuJoCo state")
    trace = args.output_root / "physical_trace.npz"
    arrays: dict[str, Any] = {k: np.asarray(v) for k, v in history.items()}
    np.savez_compressed(trace, **arrays)
    result = dict(
        commitment,
        commitment_hash=hash_json(commitment),
        physical_trace_hash=hash_bytes(trace.read_bytes()),
        first_contact_frame=first_contact_frame,
        first_contact_bodies=first_contact_bodies,
        clean_foot_only=bool(
            first_contact_bodies
            and all(i < 2 for i in first_contact_bodies)
            and not np.any(np.asarray(history["force_n"])[:, :, 2:] > 1)
        ),
        minimum_pelvis_z_m=minimum_pelvis,
        final_ball_position_m=data.xpos[ball].tolist(),
        taskspace_gate_selected=gate,
        executed_motor_policy=policy,
        qualification="UNQUALIFIED_CPU_TRANSFER_DIAGNOSTIC",
    )
    result["report_hash"] = hash_json(result)
    (args.output_root / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "report_hash",
                    "first_contact_frame",
                    "clean_foot_only",
                    "minimum_pelvis_z_m",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
