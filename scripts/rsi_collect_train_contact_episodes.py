"""TRAIN8 native neural experience, not teacher labels or learned improvement."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("starts-root", "learner-root", "export-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("starts-hash", "learner-hash", "export-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("new output and frozen collector required")
    pins = {}
    receipts = []
    for root, h in (
        (a.starts_root, a.starts_hash),
        (a.learner_root, a.learner_hash),
        (a.export_root, a.export_hash),
    ):
        f = root / "complete.json"
        if digest(f) != h:
            raise ValueError("immutable prerequisite required")
        d = json.loads(f.read_text())
        receipts.append(d)
        pins.update(d["input_file_hashes"])
        pins[str(f)] = h
    starts, learner, export = receipts
    placement = starts.get("ball_placement_protocol", "legacy-subset-full-v0")
    if placement not in ("legacy-subset-full-v0", "body-bound-v1"):
        raise ValueError("explicit known source ball-placement protocol required")
    from rosclaw_soccer.rsi.ball_aware_inputs import require_ball_aware_learning
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES

    require_ball_aware_learning(learner)
    if (
        starts.get("status") != "TRAIN8_RESET_PLANS_NOT_CONTACT_EXPERIENCE"
        or starts.get("source_split") != "train"
        or starts.get("worlds") != 64
        or set(starts.get("plan_hashes", {})) != set(TRAIN_MOTION_NAMES)
    ):
        raise ValueError("all and only eight TRAIN sources required")
    initial_path = a.starts_root / "train-starts.npz"
    pins[str(initial_path)] = starts["starts_hash"]
    for name, h in starts["plan_hashes"].items():
        pins[str(a.starts_root / (name.removesuffix(".npz") + "-plan.npz"))] = h
    checkpoint = a.learner_root / learner["checkpoint_file"]
    pins[str(checkpoint)] = learner["checkpoint_hash"]
    model_path = a.export_root / "compiled_model.mjb"
    pins[str(model_path)] = export["model_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            source_split="train",
            ball_placement_protocol=placement,
            worlds=64,
            repeated_worlds=list(range(8)),
            maximum_native_steps=216000,
            maximum_neural_calls=21600,
            repeat_is_not_independent_training_sample=True,
            actor_parent="frozen original course2701, not rejected new candidates",
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    steps = calls = 0
    rows = []
    data = None
    results = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.ball_aware_inputs import ball_goal9
        from rosclaw_soccer.rsi.native_motion_reference import (
            point_velocity_command,
            reference_inputs70,
        )
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import LearnedPriorNativeSession

        torch.set_num_threads(1)
        model = mujoco.MjModel.from_binary_path(str(model_path))
        if (
            model.opt.timestep != 0.002
            or model.nu != 29
            or not np.all(model.actuator_forcelimited)
            or mujoco.get_mjcb_control() is not None
        ):
            raise ValueError(
                "original 2ms force-limited PD model without external control required"
            )
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        torso = model.body("robot/torso_link").id
        feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
        onnx = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/exported/policy.onnx"))
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        with np.load(initial_path, allow_pickle=False) as z:
            initial = {k: z[k].copy() for k in z.files}
        if not np.array_equal(initial["motion_id"], np.arange(64) % 8):
            raise ValueError("unbalanced or nonTRAIN reset bank")
        ai = layout.ai
        kp = model.actuator_gainprm[ai, 0]
        bias = model.actuator_biasprm[ai]
        for world in range(64):
            sid = int(initial["motion_id"][world])
            name = TRAIN_MOTION_NAMES[sid]
            with np.load(
                a.starts_root / (name.removesuffix(".npz") + "-plan.npz"), allow_pickle=False
            ) as z:
                plan = {k: z[k].copy() for k in z.files}
            horizon = len(plan["joint"])
            origin = initial["origins"][world]
            goal = initial["goal_world"][world]
            first = None
            for repeat in range(2 if world < 8 else 1):
                actor = LearnedPriorNativeSession(
                    onnx,
                    config,
                    expected_model_hash=pins[str(onnx)],
                    expected_config_hash=pins[str(config)],
                    checkpoint=checkpoint,
                    expected_checkpoint_hash=learner["checkpoint_hash"],
                    arm="final",
                    reference_mode="declared79",
                )
                data = mujoco.MjData(model)
                data.qpos[:] = initial["qpos"][world]
                data.qvel[:] = initial["qvel"][world]
                warnings = tuple(int(w.number) for w in data.warning)
                rows = []
                for frame in range(horizon):
                    mujoco.mj_forward(model, data)
                    q, v = layout.canonical(data)
                    _, ref = reference_inputs70(
                        root_rotation=data.xmat[layout.pelvis].reshape(3, 3),
                        anchor_position=data.xpos[torso],
                        anchor_rotation=data.xmat[torso].reshape(3, 3),
                        joint_position=plan["joint"][frame],
                        joint_velocity=plan["joint_velocity"][frame],
                        reference_position=plan["torso_position"][frame] + origin,
                        reference_quaternion=plan["torso_quaternion"][frame],
                        reference_linear_velocity=plan["torso_linear"][frame],
                        reference_angular_velocity=plan["torso_angular"][frame],
                    )
                    command = point_velocity_command(
                        root_rotation=data.xmat[layout.pelvis].reshape(3, 3),
                        current_position=data.xpos[layout.pelvis],
                        reference_position=plan["pelvis_position"][frame] + origin,
                        reference_linear_velocity=plan["pelvis_linear"][frame],
                        reference_angular_velocity=plan["pelvis_angular"][frame],
                        position_gain=1.0,
                    )
                    perception = ball_goal9(
                        pelvis_rotation=data.xmat[layout.pelvis].reshape(3, 3),
                        pelvis_position=data.xpos[layout.pelvis],
                        pelvis_linear_velocity=data.qvel[layout.rv : layout.rv + 3],
                        ball_position=data.xpos[layout.ball],
                        ball_linear_velocity=data.qvel[layout.bv : layout.bv + 3],
                        target_position=goal,
                    )
                    row = dict(
                        native_qpos_before=data.qpos.copy(),
                        native_qvel_before=data.qvel.copy(),
                        qacc_warmstart_before=data.qacc_warmstart.copy(),
                        previous_action=actor.last_action.copy(),
                        phase_before=float(actor.phase),
                        ball_to_feet=data.xpos[layout.ball] - data.xpos[feet],
                    )
                    target = actor.propose_ball_aware(
                        q,
                        v,
                        layout.angular_velocity_body(data),
                        command,
                        ref,
                        perception,
                        frame=frame,
                    )
                    calls += 1
                    if calls > 21600:
                        raise ValueError("predeclared neural call budget exceeded")
                    observation = np.concatenate((actor.last_observation, actor.last_reference))
                    if frame == 0 and not np.allclose(
                        observation, initial["observation"][world], atol=2e-5, rtol=0
                    ):
                        raise ValueError(
                            "native initial177 observation differs from actual reset export"
                        )
                    row.update(
                        observation=observation,
                        action=actor.last_action.copy(),
                        target_rad=target.copy(),
                    )
                    data.ctrl[ai] = target
                    qs = []
                    vs = []
                    forces = []
                    nonfoot = []
                    for _ in range(10):
                        before = data.time
                        mujoco.mj_step(model, data)
                        steps += 1
                        if steps > 216000:
                            raise ValueError("predeclared native step budget exceeded")
                        if (
                            not np.isclose(data.time - before, 0.002, atol=1e-10, rtol=0)
                            or tuple(int(w.number) for w in data.warning) != warnings
                        ):
                            raise ValueError("native warning/restart")
                        expected = np.clip(
                            kp * target
                            + bias[:, 0]
                            + bias[:, 1] * data.actuator_length[ai]
                            + bias[:, 2] * data.actuator_velocity[ai],
                            model.actuator_forcerange[ai, 0],
                            model.actuator_forcerange[ai, 1],
                        )
                        if not np.allclose(data.actuator_force[ai], expected, atol=1e-8, rtol=0):
                            raise ValueError("native PD force contract differs")
                        cq, cv = layout.canonical(data)
                        qs.append(cq)
                        vs.append(cv)
                        foot = np.zeros(2)
                        nf = 0.0
                        for index in range(data.ncon):
                            c = data.contact[index]
                            x, y = (int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
                            other = y if x == layout.ball else x if y == layout.ball else -1
                            if other >= 0 and (model.body(other).name or "").startswith("robot/"):
                                wrench = np.zeros(6)
                                mujoco.mj_contactForce(model, data, index, wrench)
                                magnitude = float(np.linalg.norm(wrench[:3]))
                                if other in feet:
                                    foot[feet.index(other)] = max(
                                        foot[feet.index(other)], magnitude
                                    )
                                else:
                                    nf = max(nf, magnitude)
                        forces.append(foot)
                        nonfoot.append(nf)
                    row.update(
                        canonical_qpos_after=np.array(qs),
                        canonical_qvel_after=np.array(vs),
                        foot_force=np.array(forces),
                        nonfoot_force=np.array(nonfoot),
                    )
                    rows.append(row)
                raw = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
                if any(not np.isfinite(v).all() for v in raw.values()):
                    raise ValueError("nonfinite physical bank trajectory")
                if repeat == 0:
                    first = raw
                elif any(not np.array_equal(v, first[k]) for k, v in raw.items()):
                    raise ValueError("full TRAIN representative repeat differs")
                path = a.output / f"world{world}-repeat{repeat}.npz"
                np.savez_compressed(path, **raw)
                results.append(
                    dict(
                        world=world,
                        source=name,
                        repeat=repeat,
                        controls=horizon,
                        raw_hash=digest(path),
                        first_foot_contact_substep=next(
                            (
                                i
                                for i, f in enumerate(raw["foot_force"].reshape(-1, 2))
                                if (f > 10).any()
                            ),
                            None,
                        ),
                        maximum_nonfoot_force_n=float(raw["nonfoot_force"].max()),
                    )
                )
                rows = []
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="TRAIN8_NATIVE_CONTACT_EPISODES_NOT_LEARNED_SKILL",
                input_file_hashes=pins,
                source_head=a.expected_head,
                results=results,
                native_steps=steps,
                neural_calls=calls,
                independent_episodes=64,
                source_split="train",
                ball_placement_protocol=placement,
                representative_full_repeats_exact=True,
                new_optimizer_steps=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc), native_steps=steps, neural_calls=calls, hardware_authorized=False
        )
        # Even if the output disk is full, the journal retains exact counters.
        print(json.dumps({"collection_failed": failure}), flush=True)
        write_once(a.output / "failed.json", failure)
        if rows:
            import numpy as np

            np.savez_compressed(
                a.output / "partial-failed.npz",
                **{k: np.asarray([r[k] for r in rows]) for k in rows[0]},
            )
        raise


if __name__ == "__main__":
    main()
