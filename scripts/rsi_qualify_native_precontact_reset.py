"""Actual native NN/physics suffix qualification for measured TRAIN reset states."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=("representative", "all-bank"), default="representative")
    for k in ("bank-root", "episodes-root", "starts-root", "learner-root", "export-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "bank-hash",
        "episodes-hash",
        "starts-hash",
        "learner-hash",
        "export-hash",
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
        raise ValueError("new output and exact frozen reset qualifier required")
    pins = {}
    reports = []
    for key in ("bank", "episodes", "starts", "learner", "export"):
        root = getattr(a, key + "_root")
        h = getattr(a, key + "_hash")
        f = root / "complete.json"
        if digest(f) != h:
            raise ValueError("actual immutable prerequisite required")
        d = json.loads(f.read_text())
        reports.append(d)
        pins.update(d["input_file_hashes"])
        pins[str(f)] = h
    bank, episodes, starts, learner, export = reports
    if (
        bank.get("status") != "TRAIN8_PRECONTACT_RESET_BANK_NOT_TRAINED_OR_RESTORE_QUALIFIED"
        or bank.get("ball_placement_protocol") != "body-bound-v1"
        or bank.get("states") != 256
        or episodes.get("status") != "TRAIN8_NATIVE_CONTACT_EPISODES_NOT_LEARNED_SKILL"
        or episodes.get("ball_placement_protocol") != "body-bound-v1"
        or bank["input_file_hashes"].get(str(a.episodes_root / "complete.json")) != a.episodes_hash
        or episodes["input_file_hashes"].get(str(a.starts_root / "complete.json")) != a.starts_hash
        or episodes["input_file_hashes"].get(str(a.learner_root / "complete.json"))
        != a.learner_hash
    ):
        raise ValueError("same TRAIN body-bound bank, episodes, starts and neural parent required")
    bank_path = a.bank_root / "precontact-resets.npz"
    pins[str(bank_path)] = bank["bank_hash"]
    checkpoint = a.learner_root / learner["checkpoint_file"]
    pins[str(checkpoint)] = learner["checkpoint_hash"]
    model_path = a.export_root / "compiled_model.mjb"
    pins[str(model_path)] = export["model_hash"]
    for name, h in starts["plan_hashes"].items():
        pins[str(a.starts_root / (name.removesuffix(".npz") + "-plan.npz"))] = h
    worlds = 8 if a.scope == "representative" else 64
    maximum_calls = worlds * 4 * 2 * 300
    maximum_steps = maximum_calls * 10
    for world in range(worlds):
        row = next(r for r in episodes["results"] if r["world"] == world and r["repeat"] == 0)
        pins[str(a.episodes_root / f"world{world}-repeat0.npz")] = row["raw_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            worlds=list(range(worlds)),
            scope=a.scope,
            leads=[10, 20, 40, 80],
            full_independent_repeats=2,
            complete_original_suffix_not_fixed_short_slice=True,
            maximum_native_steps=maximum_steps,
            maximum_neural_calls=maximum_calls,
            exact_comparison_not_widened_tolerance=True,
            restore_qacc_warmstart_after_initial_forward=True,
            optimizer_updates=0,
            no_future_physics_or_action_used_for_control=True,
            hardware_authorized=False,
        ),
    )
    results = []
    steps = calls = 0
    rows = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import target29
        from rosclaw_soccer.rsi.ball_aware_inputs import ball_goal9
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_motion_reference import (
            point_velocity_command,
            reference_inputs70,
        )
        from rosclaw_soccer.rsi.native_policy_history import restore_native_policy_history
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
            raise ValueError("original 2ms bounded PD model required")
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        torso = model.body("robot/torso_link").id
        feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
        ai = layout.ai
        kp = model.actuator_gainprm[ai, 0]
        bias = model.actuator_biasprm[ai]
        onnx = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/exported/policy.onnx"))
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        with np.load(bank_path, allow_pickle=False) as z:
            states = {k: z[k].copy() for k in z.files}
        for world in range(worlds):
            with np.load(a.episodes_root / f"world{world}-repeat0.npz", allow_pickle=False) as z:
                original = {k: z[k].copy() for k in z.files}
            name = TRAIN_MOTION_NAMES[world % 8]
            with np.load(
                a.starts_root / (name.removesuffix(".npz") + "-plan.npz"), allow_pickle=False
            ) as z:
                plan = {k: z[k].copy() for k in z.files}
            for lead in (10, 20, 40, 80):
                ids = np.flatnonzero((states["world"] == world) & (states["lead_controls"] == lead))
                if ids.shape != (1,):
                    raise ValueError("exactly one declared reset per source and lead required")
                index = int(ids[0])
                start = int(states["reference_frame"][index])
                horizon = len(plan["joint"])
                origin = states["origin"][index]
                goal = states["goal_world"][index]
                if not np.array_equal(states["qpos"][index], original["native_qpos_before"][start]):
                    raise ValueError("reset state is not the actual declared source frame")
                first = None
                for repeat in (0, 1):
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
                    restore_native_policy_history(
                        actor,
                        frame=start,
                        previous_action=states["previous_action"][index],
                        phase_before=float(states["phase_before"][index]),
                    )
                    data = mujoco.MjData(model)
                    data.qpos[:] = states["qpos"][index]
                    data.qvel[:] = states["qvel"][index]
                    data.time = start * 0.02
                    if start:
                        data.ctrl[ai] = target29(states["previous_action"][index], actor.params)
                    data.qacc_warmstart[:] = states["qacc_warmstart"][index]
                    mujoco.mj_forward(model, data)
                    # The recorded snapshot is AFTER its original mj_forward.
                    # Rebuild derived state, then reinstate its recorded solver seed.
                    data.qacc_warmstart[:] = states["qacc_warmstart"][index]
                    warnings = tuple(int(w.number) for w in data.warning)
                    rows = []
                    for frame in range(start, horizon):
                        if frame != start:
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
                        if calls > maximum_calls:
                            raise ValueError("neural budget exceeded")
                        row.update(
                            observation=np.concatenate(
                                (actor.last_observation, actor.last_reference)
                            ),
                            action=actor.last_action.copy(),
                            target_rad=target.copy(),
                        )
                        data.ctrl[ai] = target
                        qs = []
                        vs = []
                        force = []
                        nf = []
                        for _ in range(10):
                            mujoco.mj_step(model, data)
                            steps += 1
                            if (
                                steps > maximum_steps
                                or tuple(int(w.number) for w in data.warning) != warnings
                            ):
                                raise ValueError("native budget, warning or restart")
                            expected = np.clip(
                                kp * target
                                + bias[:, 0]
                                + bias[:, 1] * data.actuator_length[ai]
                                + bias[:, 2] * data.actuator_velocity[ai],
                                model.actuator_forcerange[ai, 0],
                                model.actuator_forcerange[ai, 1],
                            )
                            if not np.allclose(
                                data.actuator_force[ai], expected, atol=1e-8, rtol=0
                            ):
                                raise ValueError("actual PD law differs")
                            cq, cv = layout.canonical(data)
                            qs.append(cq)
                            vs.append(cv)
                            ff = np.zeros(2)
                            nonfoot = 0.0
                            for ci in range(data.ncon):
                                c = data.contact[ci]
                                x, y = (int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
                                other = y if x == layout.ball else x if y == layout.ball else -1
                                if other >= 0 and (model.body(other).name or "").startswith(
                                    "robot/"
                                ):
                                    wrench = np.zeros(6)
                                    mujoco.mj_contactForce(model, data, ci, wrench)
                                    mag = float(np.linalg.norm(wrench[:3]))
                                    if other in feet:
                                        ff[feet.index(other)] = max(ff[feet.index(other)], mag)
                                    else:
                                        nonfoot = max(nonfoot, mag)
                            force.append(ff)
                            nf.append(nonfoot)
                        row.update(
                            canonical_qpos_after=np.array(qs),
                            canonical_qvel_after=np.array(vs),
                            foot_force=np.array(force),
                            nonfoot_force=np.array(nf),
                        )
                        rows.append(row)
                    raw = {k: np.array([r[k] for r in rows]) for k in rows[0]}
                    if any(not np.isfinite(v).all() for v in raw.values()):
                        raise ValueError("nonfinite reset proof")
                    errors = {
                        k: float(np.max(np.abs(v - original[k][start:]))) for k, v in raw.items()
                    }
                    exact = {
                        k: bool(np.array_equal(v, original[k][start:])) for k, v in raw.items()
                    }
                    if first is None:
                        first = raw
                    elif any(not np.array_equal(v, first[k]) for k, v in raw.items()):
                        raise ValueError("independent fresh reset repeats differ")
                    folder = a.output / f"world{world}-lead{lead}-repeat{repeat}"
                    folder.mkdir()
                    np.savez_compressed(folder / "native.npz", **raw)
                    result = dict(
                        world=world,
                        source=name,
                        lead=lead,
                        repeat=repeat,
                        frame=start,
                        controls=horizon - start,
                        exact_fields=exact,
                        max_errors=errors,
                        passed=all(exact.values()),
                        raw_hash=digest(folder / "native.npz"),
                    )
                    write_once(folder / "result.json", result)
                    results.append(result)
                    rows = []
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_PRECONTACT_RESET_SUFFIX_EXACT_NOT_SKILL"
                if all(r["passed"] for r in results)
                else "NATIVE_PRECONTACT_RESET_SUFFIX_REJECTED",
                input_file_hashes=pins,
                source_head=a.expected_head,
                scope=a.scope,
                measured_reset_states=worlds * 4,
                results=results,
                native_steps=steps,
                neural_calls=calls,
                optimizer_updates=0,
                all_fresh_reset_repeats_exact=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            native_steps=steps,
            neural_calls=calls,
            completed_cases=len(results),
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise


if __name__ == "__main__":
    main()
