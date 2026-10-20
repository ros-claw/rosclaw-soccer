"""Paired actual NN reference-conditioned native contact pilot; no goal claim.

Two fixed public DEV source plans, declared world, initial/final, full independent repeat.
Only declared absolute reference commands are reused; all relative features are
recomputed from CURRENT native geometry. Future GPU states/actions are unused.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("learner-root", "plan-root", "export-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "learner-unit",
        "learner-hash",
        "plan-unit",
        "plan-hash",
        "export-unit",
        "export-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--position-gain", type=float, choices=(0.0, 1.0), default=0.0)
    p.add_argument("--command-point", choices=("torso", "pelvis"), default="torso")
    p.add_argument("--world-index", type=int, choices=range(32), default=0)
    a = p.parse_args()
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import (
        motion_checkpoint_filename,
        require_completed_stable_motion_course,
        require_completed_stable_motion_smoke,
    )
    from rosclaw_soccer.rsi.velocity_command_protocol import require_same_command_protocol
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen native contact pilot required")
    pins = {}
    receipts = []
    for root, unit, expected in (
        (a.learner_root, a.learner_unit, a.learner_hash),
        (a.plan_root, a.plan_unit, a.plan_hash),
        (a.export_root, a.export_unit, a.export_hash),
    ):
        wait_for_terminal_collection(unit)
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("actual immutable terminal prerequisite required")
        receipts.append(json.loads(path.read_text()))
        pins[str(path)] = expected
    learner, plan, export = receipts
    if learner["status"] == "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_SMOKE_NOT_SKILL_QUALIFIED":
        require_completed_stable_motion_smoke(learner)
        require_same_command_protocol(learner, a.command_point, a.position_gain)
    elif learner["status"] == "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_COURSE_NOT_SKILL_QUALIFIED":
        require_completed_stable_motion_course(learner)
        require_same_command_protocol(learner, a.command_point, a.position_gain)
    elif learner["status"] != "MJLAB_BOUNDED_REFERENCE_IMITATION_NOT_SKILL_QUALIFIED":
        raise ValueError("completed stable online smoke or bounded imitation required")
    if (
        plan["status"] != "MJLAB_PAIRED_DEV_MOTION_TRACKING_NOT_FOOTBALL_QUALIFIED"
        or not plan["stable_prior_rehearsal"]
        or export["status"] != "MJLAB_NATIVE_EXPORT_ONE_CONTROL_DIAGNOSTIC_NOT_LONG_RUN_QUALIFIED"
    ):
        raise ValueError(
            "declared learned candidate, public source plans and original body required"
        )
    checkpoint = a.learner_root / motion_checkpoint_filename(learner)
    pins[str(checkpoint)] = learner["checkpoint_hash"]
    model_path = a.export_root / "compiled_model.mjb"
    pins[str(model_path)] = export["model_hash"]
    for index, source in enumerate((7, 10)):
        row = next(
            r
            for r in plan["results"]
            if r["source"].startswith(f"soccer-standard-{source:03d}_") and r["arm"] == "initial"
        )
        if row["completed_count"] != 32:
            raise ValueError("complete first-episode absolute source plans required")
        pins[str(a.plan_root / f"{index}-initial-raw.npz")] = row["raw_hash"]
    motion_files = {}
    if a.command_point == "pelvis":
        for source, side in ((7, "left"), (10, "right")):
            candidates = {
                k: v
                for k, v in plan["input_file_hashes"].items()
                if k.endswith(f"/soccer-standard-{source:03d}_{side}.npz")
            }
            if not candidates or len(set(candidates.values())) != 1:
                raise ValueError(
                    "unique qualified reference data bytes required for root semantics"
                )
            path = next(iter(candidates))
            pins[path] = candidates[path]
            motion_files[source] = Path(path)
    for k, v in learner["input_file_hashes"].items():
        if k.endswith("/policy/velocity/v0/exported/policy.onnx") or k.endswith(
            "/policy/velocity/v0/params/deploy.yaml"
        ):
            pins[k] = v
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            sources=[7, 10],
            world_indices=[a.world_index],
            arms=["initial", "final"],
            full_independent_repeats=2,
            maximum_native_steps=24000,
            maximum_neural_calls=2400,
            reference_features_recomputed_from_current_native_body=True,
            nominal_future_no_scheduled_gpu_pushes=True,
            compiled_default_mocap_and_zero_warmstart=True,
            foot_force_threshold_n=1.0,
            velocity_command_position_feedback_gain=a.position_gain,
            no_joint_target_patch_or_external_body_force=True,
            velocity_command_anatomical_point=a.command_point,
            original_motion_body_indices=dict(pelvis=0, torso=9),
            maximum_reference_kinematic_calls=600,
            ball_speed_research_threshold_mps=0.5,
            not_gpu_long_run_agreement=True,
            new_optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    steps = calls = attempts = 0
    reference_fk_calls = 0
    rows = []
    data = None
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.foot_contact_timing import contact_timing_metrics
        from rosclaw_soccer.rsi.native_motion_reference import (
            point_velocity_command,
            reference_inputs70,
        )
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import LearnedPriorNativeSession

        torch.set_num_threads(1)
        model = mujoco.MjModel.from_binary_path(str(model_path))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        torso = model.body("robot/torso_link").id
        feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
        onnx = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/exported/policy.onnx"))
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        ai = layout.ai
        kp = model.actuator_gainprm[ai, 0]
        bias = model.actuator_biasprm[ai]
        if not np.all(model.actuator_forcelimited[ai]) or mujoco.get_mjcb_control() is not None:
            raise ValueError(
                "original force-limited native actuators without external callback required"
            )
        results = []
        for index, source in enumerate((7, 10)):
            with np.load(a.plan_root / f"{index}-initial-raw.npz", allow_pickle=False) as z:
                # Do not load future physical q/v or GPU action outputs.
                initial_q = z["qpos_before"][0, a.world_index].copy()
                initial_v = z["qvel_before"][0, a.world_index].copy()
                references = {
                    k: z[k][:, a.world_index].copy()
                    for k in (
                        "reference_joint",
                        "reference_joint_velocity",
                        "reference_anchor",
                        "reference_anchor_quat",
                        "reference_anchor_linear_velocity",
                        "reference_anchor_angular_velocity",
                    )
                }
            horizon = len(references["reference_joint"])
            if not 1 <= horizon <= 300:
                raise ValueError("bounded whole source plan required")
            root_reference = None
            if a.command_point == "pelvis":
                with np.load(motion_files[source], allow_pickle=False) as z:
                    if z["fps"].size != 1 or float(z["fps"].item()) != 50:
                        raise ValueError("original50Hz source data required")
                    bp = z["body_pos_w"][:horizon].copy()
                    bq = z["body_quat_w"][:horizon].copy()
                    bl = z["body_lin_vel_w"][:horizon].copy()
                    ba = z["body_ang_vel_w"][:horizon].copy()
                origin = references["reference_anchor"][0] - bp[0, 9]
                if not np.allclose(
                    bp[:, 9] + origin, references["reference_anchor"], atol=1e-5, rtol=0
                ) or not np.allclose(
                    bq[:, 9], references["reference_anchor_quat"], atol=1e-6, rtol=0
                ):
                    raise ValueError("source anatomical body map or plan origin differs")
                root_reference = dict(
                    position=bp[:, 0] + origin,
                    quaternion=bq[:, 0],
                    linear=bl[:, 0],
                    angular=ba[:, 0],
                )
                scratch = mujoco.MjData(model)
                for frame in range(horizon):
                    scratch.qpos[:] = initial_q
                    scratch.qpos[layout.rq : layout.rq + 3] = root_reference["position"][frame]
                    scratch.qpos[layout.rq + 3 : layout.rq + 7] = root_reference["quaternion"][
                        frame
                    ]
                    scratch.qpos[layout.qi] = references["reference_joint"][frame]
                    mujoco.mj_kinematics(model, scratch)
                    reference_fk_calls += 1
                    expected_torso_rotation = np.zeros(9)
                    mujoco.mju_quat2Mat(
                        expected_torso_rotation,
                        references["reference_anchor_quat"][frame].astype(np.float64),
                    )
                    if not np.allclose(
                        scratch.xpos[torso],
                        references["reference_anchor"][frame],
                        atol=1e-5,
                        rtol=0,
                    ):
                        raise ValueError(
                            "native FK does not bind the claimed pelvis/torso data identities"
                        )
                    if (
                        not np.allclose(
                            scratch.xmat[torso], expected_torso_rotation, atol=1e-5, rtol=0
                        )
                        or reference_fk_calls > 600
                    ):
                        raise ValueError(
                            "reference orientation identity or kinematic budget differs"
                        )
            for arm in ("initial", "final"):
                first = None
                for repeat in range(2):
                    actor = LearnedPriorNativeSession(
                        onnx,
                        config,
                        expected_model_hash=pins[str(onnx)],
                        expected_config_hash=pins[str(config)],
                        checkpoint=checkpoint,
                        expected_checkpoint_hash=learner["checkpoint_hash"],
                        arm=arm,
                        reference_mode="declared70",
                    )
                    data = mujoco.MjData(model)
                    data.qpos[:] = initial_q
                    data.qvel[:] = initial_v
                    warning = tuple(int(w.number) for w in data.warning)
                    rows = []
                    for frame in range(horizon):
                        mujoco.mj_forward(model, data)
                        q, v = layout.canonical(data)
                        command, reference = reference_inputs70(
                            root_rotation=data.xmat[layout.pelvis].reshape(3, 3),
                            anchor_position=data.xpos[torso],
                            anchor_rotation=data.xmat[torso].reshape(3, 3),
                            joint_position=references["reference_joint"][frame],
                            joint_velocity=references["reference_joint_velocity"][frame],
                            reference_position=references["reference_anchor"][frame],
                            reference_quaternion=references["reference_anchor_quat"][frame],
                            reference_linear_velocity=references[
                                "reference_anchor_linear_velocity"
                            ][frame],
                            reference_angular_velocity=references[
                                "reference_anchor_angular_velocity"
                            ][frame],
                            position_gain=a.position_gain,
                        )
                        if root_reference is not None:
                            command = point_velocity_command(
                                root_rotation=data.xmat[layout.pelvis].reshape(3, 3),
                                current_position=data.xpos[layout.pelvis],
                                reference_position=root_reference["position"][frame],
                                reference_linear_velocity=root_reference["linear"][frame],
                                reference_angular_velocity=root_reference["angular"][frame],
                                position_gain=a.position_gain,
                            )
                        target = actor.propose_reference(
                            q,
                            v,
                            layout.angular_velocity_body(data),
                            command,
                            reference,
                            frame=frame,
                        )
                        calls += 1
                        row = dict(
                            native_qpos_before=data.qpos.copy(),
                            canonical_qpos_before=q,
                            canonical_qvel_before=v,
                            observation=np.concatenate(
                                (actor.last_observation, actor.last_reference)
                            ),
                            action=actor.last_action.copy(),
                            target_rad=target.copy(),
                            command=command.copy(),
                            anchor_xy_error=np.linalg.norm(
                                data.xpos[torso, :2] - references["reference_anchor"][frame, :2]
                            ),
                        )
                        data.ctrl[ai] = target
                        qs = []
                        vs = []
                        forces = []
                        contacts = []
                        for _ in range(10):
                            before = data.time
                            attempts += 1
                            mujoco.mj_step(model, data)
                            steps += 1
                            if (
                                not np.isclose(data.time - before, 0.002, atol=1e-10, rtol=0)
                                or tuple(int(w.number) for w in data.warning) != warning
                            ):
                                raise ValueError("actual native warning or implicit restart")
                            actual = data.actuator_force[ai].copy()
                            expected = np.clip(
                                kp * target
                                + bias[:, 0]
                                + bias[:, 1] * data.actuator_length[ai]
                                + bias[:, 2] * data.actuator_velocity[ai],
                                model.actuator_forcerange[ai, 0],
                                model.actuator_forcerange[ai, 1],
                            )
                            if not np.allclose(actual, expected, rtol=0, atol=1e-8):
                                raise ValueError(
                                    "actual force differs from original compiled PD encoding"
                                )
                            cq, cv = layout.canonical(data)
                            qs.append(cq)
                            vs.append(cv)
                            forces.append(actual)
                            foot_force = np.zeros(2)
                            for contact in range(data.ncon):
                                c = data.contact[contact]
                                b1 = int(model.geom_bodyid[c.geom1])
                                b2 = int(model.geom_bodyid[c.geom2])
                                other = b2 if b1 == layout.ball else b1 if b2 == layout.ball else -1
                                if other in feet:
                                    wrench = np.zeros(6)
                                    mujoco.mj_contactForce(model, data, contact, wrench)
                                    foot = feet.index(other)
                                    foot_force[foot] = max(
                                        foot_force[foot], np.linalg.norm(wrench[:3])
                                    )
                            contacts.append(foot_force)
                        row.update(
                            canonical_qpos_after=np.asarray(qs),
                            canonical_qvel_after=np.asarray(vs),
                            actual_force=np.asarray(forces),
                            foot_ball_contact_force_n=np.asarray(contacts),
                        )
                        rows.append(row)
                    raw = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
                    if any(not np.isfinite(v).all() for v in raw.values()):
                        raise ValueError("all raw actual physical observations must be finite")
                    if not np.array_equal(
                        raw["canonical_qpos_before"][1:], raw["canonical_qpos_after"][:-1, -1]
                    ) or not np.array_equal(
                        raw["canonical_qvel_before"][1:], raw["canonical_qvel_after"][:-1, -1]
                    ):
                        raise ValueError("continuous actual native state required")
                    folder = a.output / f"source{source}-{arm}-repeat{repeat}"
                    folder.mkdir()
                    np.savez_compressed(folder / "native.npz", **raw)
                    if first is None:
                        first = raw
                    elif any(not np.array_equal(v, first[k]) for k, v in raw.items()):
                        raise ValueError("independent full native/neural repeat differs")
                    cq = raw["canonical_qpos_after"].reshape(-1, 43)
                    cv = raw["canonical_qvel_after"].reshape(-1, 41)
                    contact = raw["foot_ball_contact_force_n"].reshape(-1, 2)
                    touch = np.flatnonzero(contact.max(axis=1) > 1)
                    upright = 1 - 2 * (cq[:, 4] ** 2 + cq[:, 5] ** 2)
                    speed = np.linalg.norm(cv[:, 35:38], axis=1)
                    post_speed = float(speed[touch[0] :].max()) if len(touch) else 0.0
                    intended_foot = 0 if source == 7 else 1
                    result = dict(
                        source=source,
                        world_index=a.world_index,
                        arm=arm,
                        repeat=repeat,
                        native_steps=horizon * 10,
                        neural_calls=horizon,
                        maximum_foot_ball_force_n=contact.max(axis=0).tolist(),
                        reference_kicking_foot="left" if intended_foot == 0 else "right",
                        intended_foot_contact_detected=bool(contact[:, intended_foot].max() > 1),
                        foot_contact_detected=bool(len(touch)),
                        maximum_ball_speed_after_first_foot_contact_mps=post_speed,
                        contact_and_ball_speed_research_gate=bool(len(touch) and post_speed >= 0.5),
                        maximum_ball_xy_displacement_m=float(
                            np.linalg.norm(
                                cq[:, 36:38] - initial_q[layout.bq : layout.bq + 2], axis=1
                            ).max()
                        ),
                        maximum_ball_height_m=float(cq[:, 38].max()),
                        minimum_pelvis_height_m=float(cq[:, 2].min()),
                        minimum_upright=float(upright.min()),
                        stable_research_guard=bool(cq[:, 2].min() >= 0.65 and upright.min() >= 0.9),
                        mean_anchor_xy_error_m=float(raw["anchor_xy_error"].mean()),
                        velocity_command_position_feedback_gain=a.position_gain,
                        velocity_command_anatomical_point=a.command_point,
                        raw_hash=digest(folder / "native.npz"),
                        goal_success=False,
                        football_skill_proven=False,
                        **contact_timing_metrics(
                            contact, speed, intended_foot=intended_foot, timestep=0.002
                        ),
                    )
                    write_once(folder / "complete.json", result)
                    results.append(result)
                    rows = []
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_REFERENCE_CONTACT_PILOT_NOT_GOAL_OR_TEAM_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                native_steps=steps,
                attempted_native_steps=attempts,
                neural_calls=calls,
                reference_kinematic_calls=reference_fk_calls,
                all_full_independent_repeats_exact=True,
                new_optimizer_steps=0,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        if rows:
            import numpy as np

            np.savez_compressed(
                a.output / "failed-partial.npz",
                **{k: np.asarray([r[k] for r in rows]) for k in rows[0]},
            )
        if data is not None:
            import numpy as np

            np.savez_compressed(
                a.output / "failed-live-state.npz", qpos=data.qpos.copy(), qvel=data.qvel.copy()
            )
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                native_steps=steps,
                attempted_native_steps=attempts,
                completed_neural_calls=calls,
                reference_kinematic_calls=reference_fk_calls,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
