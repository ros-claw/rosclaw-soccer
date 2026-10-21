"""Qualify all256 measured GPU resets, CPU/NN one-control parity and boundaries."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "bank-root",
        "cpu-proof-root",
        "role-proof-root",
        "learner-root",
        "starts-root",
        "reference-root",
        "output",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "bank-hash",
        "cpu-proof-hash",
        "role-proof-hash",
        "learner-hash",
        "starts-hash",
        "expected-head",
    ):
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
        raise ValueError("new output and frozen GPU restore qualifier required")
    pins = {}
    reports = []
    for key in ("bank", "cpu_proof", "role_proof", "learner", "starts"):
        root = getattr(a, key + "_root")
        h = getattr(a, key + "_hash")
        f = root / "complete.json"
        if digest(f) != h:
            raise ValueError("exact completed prerequisite required")
        r = json.loads(f.read_text())
        reports.append(r)
        pins.update(r["input_file_hashes"])
        pins[str(f)] = h
    bank, cpu, role, learner, starts = reports
    if (
        cpu.get("status") != "NATIVE_PRECONTACT_RESET_SUFFIX_EXACT_NOT_SKILL"
        or cpu.get("scope") != "all-bank"
        or cpu.get("measured_reset_states") != 256
        or len(cpu.get("results", [])) != 512
        or any(r.get("passed") is not True for r in cpu["results"])
        or cpu["input_file_hashes"].get(str(a.bank_root / "complete.json")) != a.bank_hash
        or role.get("status") != "FOOT_ONLY_SENSOR_REWARD_QUALIFIED_NOT_SKILL"
        or role.get("ball_placement_protocol") != "body-bound-v1"
        or role.get("probe_passed") is not True
        or role["input_file_hashes"].get(str(a.learner_root / "complete.json")) != a.learner_hash
        or bank.get("ball_placement_protocol") != "body-bound-v1"
    ):
        raise ValueError("full measured-state CPU proof and fresh named-body role proof required")
    for r in cpu["results"]:
        pins[
            str(
                a.cpu_proof_root
                / f"world{r['world']}-lead{r['lead']}-repeat{r['repeat']}/native.npz"
            )
        ] = r["raw_hash"]
    role_source = next(
        Path(k).parent.parent
        for k in role["input_file_hashes"]
        if k.endswith("/scripts/rsi_qualify_foot_only.py")
    )
    for relative in (
        "scripts/rsi_mjlab_motion_backend.py",
        "scripts/rsi_mjlab_ball_aware_backend.py",
        "scripts/rsi_mjlab_foot_only_backend.py",
        "scripts/rsi_mjwarp_addressed_contact_force.py",
        "src/rosclaw_soccer/rsi/identity_kick_credit.py",
    ):
        if digest(Path.cwd() / relative) != role["input_file_hashes"].get(
            str(role_source / relative)
        ):
            raise ValueError("same actually qualified body/contact/reward source required")
    bp = a.bank_root / "precontact-resets.npz"
    pins[str(bp)] = bank["bank_hash"]
    checkpoint = a.learner_root / learner["checkpoint_file"]
    pins[str(checkpoint)] = learner["checkpoint_hash"]
    for n, h in starts["plan_hashes"].items():
        pins[str(a.starts_root / (n.removesuffix(".npz") + "-plan.npz"))] = h
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            worlds=256,
            maximum_gpu_world_steps=5120,
            maximum_cpu_steps=2560,
            maximum_gpu_nn_rows=512,
            maximum_cpu_nn_rows=256,
            optimizer_updates=0,
            initial_observation_atol=2e-5,
            target_atol_rad=2e-5,
            qpos_atol_mixed=1e-3,
            qvel_atol_mixed=0.02,
            force_atol_n=2.0,
            force_rtol=0.05,
            forced_deadline_worlds=list(range(8)),
            deadline_fixture_only_not_skill=True,
            body_state_not_overwritten_during_active_step=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    gpu_steps = native_steps = gpu_nn = cpu_nn = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from scripts.rsi_mjwarp_addressed_contact_force import install

        install()
        sys.path.insert(0, str(a.reference_root))
        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.ball_aware_inputs import ball_goal9
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_motion_reference import (
            point_velocity_command,
            reference_inputs70,
        )
        from rosclaw_soccer.rsi.native_policy_history import restore_native_policy_history
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import (
            LearnedPriorNativeSession,
            VelocityPriorActor,
        )
        from scripts.rsi_foot_only_calibration import body_bindings
        from scripts.rsi_mjlab_precontact_backend import (
            PrecontactMotionEnv,
            make_precontact_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        onnx = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/exported/policy.onnx"))
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        actor = (
            VelocityPriorActor.from_onnx(onnx, expected_hash=pins[str(onnx)], reference_features=79)
            .cuda()
            .eval()
        )
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            {k: loaded["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()}
        )
        cfg = make_precontact_training_cfg(
            parameters=params,
            command_mode="reference_position_velocity",
            command_point="pelvis",
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=256,
            seed=6601,
            balanced_initial_sources=True,
            ball_placement_protocol="body-bound-v1",
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = PrecontactMotionEnv(
            cfg=cfg,
            device="cuda:0",
            render_mode=None,
            reset_bank=bp,
            reset_bank_hash=bank["bank_hash"],
            lead_controls=None,
        )
        obs, _ = env.reset(seed=6601)
        model = env.sim.mj_model
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        body_bindings(model, layout)
        torso = model.body("robot/torso_link").id
        feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
        q0 = env.sim.data.qpos.cpu().numpy().copy()
        v0 = env.sim.data.qvel.cpu().numpy().copy()
        warm0 = env.sim.data.qacc_warmstart.cpu().numpy().copy()
        initial_obs = obs["actor"].cpu().numpy().copy()
        index = env.course_bank_index.cpu().numpy().copy()
        if not np.array_equal(index, np.arange(256)):
            raise ValueError("all bank states must actually be exercised")
        with np.load(bp, allow_pickle=False) as z:
            b = {k: z[k].copy() for k in z.files}
        if not np.array_equal(initial_obs[:, 69:98], b["previous_action"]):
            raise ValueError("previous action not restored in actual177 observation")
        if not np.array_equal(env.episode_length_buf.cpu().numpy(), b["reference_frame"]):
            raise ValueError("gait phase source frame differs")
        with torch.no_grad():
            actions = actor(obs["actor"])
        gpu_nn += 256
        cnq = []
        cnv = []
        cff = []
        cnf = []
        cobs = []
        ctarget = []
        for i in range(256):
            name = TRAIN_MOTION_NAMES[int(b["source_id"][i])]
            frame = int(b["reference_frame"][i])
            with np.load(
                a.starts_root / (name.removesuffix(".npz") + "-plan.npz"), allow_pickle=False
            ) as z:
                plan = {k: z[k].copy() for k in z.files}
            session = LearnedPriorNativeSession(
                onnx,
                config,
                expected_model_hash=pins[str(onnx)],
                expected_config_hash=pins[str(config)],
                checkpoint=checkpoint,
                expected_checkpoint_hash=learner["checkpoint_hash"],
                arm="final",
                reference_mode="declared79",
            )
            restore_native_policy_history(
                session,
                frame=frame,
                previous_action=b["previous_action"][i],
                phase_before=float(b["phase_before"][i]),
            )
            d = mujoco.MjData(model)
            d.qpos[:] = q0[i]
            d.qvel[:] = v0[i]
            d.time = frame * 0.02
            if frame:
                d.ctrl[layout.ai] = target29(b["previous_action"][i], session.params)
            d.qacc_warmstart[:] = warm0[i]
            mujoco.mj_forward(model, d)
            d.qacc_warmstart[:] = warm0[i]
            q, v = layout.canonical(d)
            origin = env.scene.env_origins[i].cpu().numpy()
            goal = b["goal_world"][i] + origin - b["origin"][i]
            _, ref = reference_inputs70(
                root_rotation=d.xmat[layout.pelvis].reshape(3, 3),
                anchor_position=d.xpos[torso],
                anchor_rotation=d.xmat[torso].reshape(3, 3),
                joint_position=plan["joint"][frame],
                joint_velocity=plan["joint_velocity"][frame],
                reference_position=plan["torso_position"][frame] + origin,
                reference_quaternion=plan["torso_quaternion"][frame],
                reference_linear_velocity=plan["torso_linear"][frame],
                reference_angular_velocity=plan["torso_angular"][frame],
            )
            cmd = point_velocity_command(
                root_rotation=d.xmat[layout.pelvis].reshape(3, 3),
                current_position=d.xpos[layout.pelvis],
                reference_position=plan["pelvis_position"][frame] + origin,
                reference_linear_velocity=plan["pelvis_linear"][frame],
                reference_angular_velocity=plan["pelvis_angular"][frame],
                position_gain=1.0,
            )
            sense = ball_goal9(
                pelvis_rotation=d.xmat[layout.pelvis].reshape(3, 3),
                pelvis_position=d.xpos[layout.pelvis],
                pelvis_linear_velocity=d.qvel[layout.rv : layout.rv + 3],
                ball_position=d.xpos[layout.ball],
                ball_linear_velocity=d.qvel[layout.bv : layout.bv + 3],
                target_position=goal,
            )
            target = session.propose_ball_aware(
                q, v, layout.angular_velocity_body(d), cmd, ref, sense, frame=frame
            )
            cpu_nn += 1
            cobs.append(np.concatenate((session.last_observation, session.last_reference)))
            ctarget.append(target)
            d.ctrl[layout.ai] = target
            ff = []
            nf = []
            for _ in range(10):
                mujoco.mj_step(model, d)
                native_steps += 1
                row = np.zeros(2)
                otherforce = 0.0
                for j in range(d.ncon):
                    c = d.contact[j]
                    x, y = (int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
                    other = y if x == layout.ball else x if y == layout.ball else -1
                    if other >= 0 and (model.body(other).name or "").startswith("robot/"):
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model, d, j, wrench)
                        mag = float(np.linalg.norm(wrench[:3]))
                        if other in feet:
                            row[feet.index(other)] = max(row[feet.index(other)], mag)
                        else:
                            otherforce = max(otherforce, mag)
                ff.append(row)
                nf.append(otherforce)
            cnq.append(d.qpos.copy())
            cnv.append(d.qvel.copy())
            cff.append(ff)
            cnf.append(nf)
        obs, _, term, trunc, _ = env.step(actions)
        gpu_steps += 2560
        snap = env.physical_step_data()
        physical = {k: v.cpu().numpy().copy() for k, v in snap.items()}
        errors = dict(
            observation=float(np.abs(initial_obs - np.array(cobs)).max()),
            target=float(np.abs(physical["processed_joint_target"] - ctarget).max()),
            qpos=float(np.abs(physical["qpos_after"] - cnq).max()),
            qvel=float(np.abs(physical["qvel_after"] - cnv).max()),
        )
        forces_ok = bool(
            np.allclose(physical["foot_force_history"], cff, atol=2.0, rtol=0.05)
            and np.allclose(physical["nonfoot_force_history"], cnf, atol=2.0, rtol=0.05)
        )
        labels_ok = bool(
            np.array_equal(physical["foot_force_history"] > 10.0, np.array(cff) > 10.0)
            and np.array_equal(physical["nonfoot_force_history"] > 1.0, np.array(cnf) > 1.0)
        )
        first_failed = int(term.sum())
        first_timeout = int(trunc.sum())
        np.savez_compressed(
            a.output / "first-control.npz",
            initial_qpos=q0,
            initial_qvel=v0,
            initial_warmstart=warm0,
            bank_index=index,
            gpu_observation=initial_obs,
            cpu_observation=cobs,
            cpu_target=ctarget,
            cpu_qpos=cnq,
            cpu_qvel=cnv,
            cpu_foot=cff,
            cpu_nonfoot=cnf,
            **physical,
        )
        # Inject an earlier DEADLINE, not a body/ball pose or action. This is a
        # lifecycle fixture, never a claimed football success.
        env._course_start_frame[:8] = env.episode_length_buf[:8] - 95
        with torch.no_grad():
            next_action = actor(obs["actor"])
        gpu_nn += 256
        _, _, term2, trunc2, _ = env.step(next_action)
        gpu_steps += 2560
        new_frame = env._course_bank["reference_frame"][env.course_bank_index]
        ended = term2 | trunc2
        boundary_ok = bool(
            torch.all(trunc2[:8])
            and torch.equal(env.episode_length_buf[ended], new_frame[ended])
            and torch.equal(
                env.command_manager.get_term("motion").time_steps[ended], new_frame[ended]
            )
        )
        np.savez_compressed(
            a.output / "boundary.npz",
            bank_index=env.course_bank_index.cpu().numpy(),
            phase=env.command_manager.get_term("motion").time_steps.cpu().numpy(),
            episode_phase=env.episode_length_buf.cpu().numpy(),
            terminated=term2.cpu().numpy(),
            truncated=trunc2.cpu().numpy(),
        )
        passed = (
            errors["observation"] <= 2e-5
            and errors["target"] <= 2e-5
            and errors["qpos"] <= 1e-3
            and errors["qvel"] <= 0.02
            and forces_ok
            and labels_ok
            and first_failed == 0
            and first_timeout == 0
            and boundary_ok
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="GPU_PRECONTACT_RESET_CONTRACT_QUALIFIED_NOT_SKILL"
                if passed
                else "GPU_PRECONTACT_RESET_CONTRACT_REJECTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                errors=errors,
                forces_ok=forces_ok,
                labels_ok=labels_ok,
                first_failed_worlds=first_failed,
                first_timeout_worlds=first_timeout,
                boundary_ok=boundary_ok,
                gpu_world_steps=gpu_steps,
                native_steps=native_steps,
                gpu_nn_rows=gpu_nn,
                cpu_nn_rows=cpu_nn,
                first_raw_hash=digest(a.output / "first-control.npz"),
                boundary_hash=digest(a.output / "boundary.npz"),
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            gpu_world_steps=gpu_steps,
            native_steps=native_steps,
            gpu_nn_rows=gpu_nn,
            cpu_nn_rows=cpu_nn,
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
