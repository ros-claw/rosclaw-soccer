"""Bounded SIM control-authority pilot: actual run -> own shadow R2 -> contact.

Not a learned-player or old exam claim. The fixed task-space teacher is used
only in its explicit training arm. No body/ball/clock reset within a world.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("plan-root", "velocity-audit-root", "velocity-qualification-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("plan-hash", "velocity-audit-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--recovery-root", type=Path)
    p.add_argument("--recovery-hash")
    p.add_argument(
        "--motor-reference", choices=("current201", "published_R2"), default="current201"
    )
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen new handoff experiment required")

    clean()
    if a.output.exists():
        raise ValueError("unused handoff output required")
    from rosclaw_soccer.rsi.paid_neighborhood_sources import load_neighborhood_plan

    _, plan, pins = load_neighborhood_plan(a.plan_root, a.plan_hash)
    if plan.get("plan_kind") is not None:
        raise ValueError("original research lineage required")
    ap = a.velocity_audit_root / "complete.json"
    if ap.is_symlink() or digest(ap) != a.velocity_audit_hash:
        raise ValueError("exact whole-world velocity transfer proof required")
    audit = json.loads(ap.read_text())
    if (
        audit["status"] != "VELOCITY_PAID_BODY_OWN98_WHOLE_WORLD_NATIVE_AUDIT_PASS"
        or not audit["independent_whole_world_replay"]
        or audit["own_native_steps"] != 48000
        or audit["own_neural_calls"] != 4800
    ):
        raise ValueError("qualified actual common-body locomotion required")
    source_transfer = next(
        Path(f)
        for f, h in audit["input_file_hashes"].items()
        if h == audit["source_hash"] and f.endswith("/complete.json")
    )
    if (
        json.loads(source_transfer.read_text())["unchanged_source_body_hash"]
        != plan["body_scene_hash"]
    ):
        raise ValueError("velocity transfer proof must cover this exact body")
    for f, h in audit["input_file_hashes"].items():
        if f in pins and pins[f] != h:
            raise ValueError("conflicting common-body source")
        pins[f] = h
    pins[str(ap)] = a.velocity_audit_hash
    qp = a.velocity_qualification_root / "complete.json"
    if pins.get(str(qp)) != digest(qp):
        raise ValueError("same qualified velocity actor required")
    qual = json.loads(qp.read_text())
    onnx = next(Path(f) for f in audit["input_file_hashes"] if f.endswith("/exported/policy.onnx"))
    cfg = next(Path(f) for f in audit["input_file_hashes"] if f.endswith("/params/deploy.yaml"))
    original = Path(plan["pipeline_root"]) / "learned-train20-exam/declaration.json"
    if pins.get(str(original)) != digest(original):
        raise ValueError("original unperturbed context required")
    case = json.loads(original.read_text())["cases"][4]
    spec = plan["specifications"]["models"]["learned_memory"]
    rp = Path(spec["root"]) / "complete.json"
    training = json.loads(rp.read_text())
    if digest(rp) != spec["hash"]:
        raise ValueError("frozen current201 reference motor required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    courses = [dict(backoff_m=1.6, speed_mps=0.8), dict(backoff_m=1.2, speed_mps=0.6)]
    arms = ["standing_current", "run_shadow_current", "run_shadow_taskspace_teacher"]
    native_budget, query_budget = 55080, 6000
    protocol = "paid.velocity_actual_shadow_then_reference80.v1"
    if a.motor_reference == "published_R2":
        arms = ["reference_from_start", "velocity_shadow_reference"]
        native_budget, query_budget = 36000, 4000
        protocol = "paid.velocity_actual_shadow_then_reference80.published_R2_v2"
    from rosclaw_soccer.rsi.paid_run_pilot_recovery import (
        contact_quality,
        recover_first_standing_world,
    )

    recovered = None
    reused_steps = 0
    if a.recovery_root is not None or a.recovery_hash is not None:
        if a.motor_reference != "current201":
            raise ValueError("never relabel current-motor physics as published R2")
        if (
            a.recovery_root is None
            or a.recovery_hash is None
            or a.output.resolve().is_relative_to(a.recovery_root.resolve())
        ):
            raise ValueError(
                "explicit separate recovery destination and exact source commitment required"
            )
        recovered = recover_first_standing_world(
            a.recovery_root, a.recovery_hash, case, courses, arms
        )
        for f, h in recovered[2].items():
            if f in pins and pins[f] != h:
                raise ValueError("conflicting recovered first-world provenance")
            pins[f] = h
        reused_steps = 3890
        check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            case=case,
            courses=courses,
            arms=arms,
            protocol=protocol,
            motor_reference=a.motor_reference,
            repeats=2,
            prepare_velocity_controls=50,
            run_controls=100,
            shadow_reference_controls=80,
            no_R2_proposal_executed_during_shadow=True,
            body_state_reset_after_start=False,
            maximum_native_steps=native_budget,
            maximum_full_neural_queries=query_budget,
            original_formal_H300_exam=False,
            optimizer_updates=0,
            reused_native_steps=reused_steps,
            recovery_root=str(a.recovery_root) if a.recovery_root else None,
            recovery_hash=a.recovery_hash,
            teacher="fixed_original_source4_taskspace_intent_only_in_explicit_teacher_arm",
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    ep = None
    physical = []
    counts = dict(velocity=0, R2_shadow=0, R2_applied=0, R2_stand=0, teacher_IK=0)
    if recovered is not None:
        counts.update(recovered[3])
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.physics.g1_velocity_observation import world_command_to_body
        from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_causal_context import CausalMotorContext
        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_own_state_teacher import OwnStateTeacherHandoff
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
        from rosclaw_soccer.rsi.paid_velocity_shadow import (
            OwnVelocityShadow,
            integrate_velocity_target,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [case])
        if scene != plan["body_scene_hash"]:
            raise ValueError("same unchanged torque body required")
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=rp,
            training_receipt_hash=spec["hash"],
            scene_hash=scene,
        )
        active_actor = actor.base if a.motor_reference == "published_R2" else actor
        records = []
        for ci, course in enumerate(courses):
            for arm in arms:
                previous = None
                for repeat in range(2):
                    folder = a.output / f"course{ci}-{arm}-repeat{repeat}"
                    folder.mkdir()
                    blobs = {}
                    if (
                        recovered is not None
                        and ci == 0
                        and arm == "standing_current"
                        and repeat == 0
                    ):
                        import shutil

                        outcome, old_blobs, _, _ = recovered
                        artifacts = {}
                        for name in old_blobs:
                            source = (
                                a.recovery_root
                                / "course0-standing_current-repeat0"
                                / (name + ".npz")
                            )
                            target = folder / (name + ".npz")
                            if target.exists():
                                raise ValueError("recovered artifact overwrite forbidden")
                            shutil.copyfile(source, target)
                            artifacts[name.replace("-", "_")] = dict(
                                raw_path=str(target.relative_to(a.output)), raw_hash=digest(target)
                            )
                        outcome = dict(outcome, artifacts=artifacts)
                        write_once(folder / "complete.json", outcome)
                        previous = old_blobs
                        records.append(outcome)
                        print(
                            json.dumps(
                                dict(
                                    recovered_world=0, new_native_steps=0, reused_native_steps=3890
                                )
                            ),
                            flush=True,
                        )
                        continue

                    def store(name, z, folder=folder, blobs=blobs):
                        path = folder / (name + ".npz")
                        if path.exists():
                            raise ValueError("raw overwrites forbidden")
                        np.savez_compressed(path, **z)
                        blobs[name] = z
                        return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

                    ep = bench.make(case)
                    data = ep.env.data
                    q, _ = ep.layout.canonical(data)
                    w, x, y, z = q[3:7]
                    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                    direction = np.array([np.cos(yaw), np.sin(yaw)])
                    # Declared initial fixture only; never write pose again.
                    data.qpos[ep.layout.rq : ep.layout.rq + 2] -= course["backoff_m"] * direction
                    mujoco.mj_forward(ep.env.model, data)
                    ep.initial = snapshot(data)
                    artifacts = dict(initial=store("initial", ep.initial))
                    physical = []
                    controller = []
                    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                    velocity_rows = []
                    shadow_queries = []
                    shadow = None
                    if arm not in ("standing_current", "reference_from_start"):
                        velocity = G1VelocityPolicy(
                            onnx,
                            cfg,
                            expected_model_hash=pins[str(onnx)],
                            expected_config_hash=pins[str(cfg)],
                        )
                        if velocity.contract_hash != qual["policy_contract_hash"]:
                            raise ValueError("same qualified98D actor required")
                        kp, kd = [
                            np.array(v, np.float64)
                            for v in (velocity.params.stiffness, velocity.params.damping)
                        ]
                        shadow = OwnVelocityShadow(ep, active_actor)
                        pulse = (
                            float(course["speed_mps"] * direction[0]),
                            float(course["speed_mps"] * direction[1]),
                            0.0,
                        )
                        for frame in range(150):
                            active_shadow = frame >= 70
                            if active_shadow:
                                context, proposal = shadow.observe()
                                counts["R2_shadow"] += 1
                            mujoco.mj_forward(ep.env.model, data)
                            q, v = ep.layout.canonical(data)
                            command = world_command_to_body(
                                q, pulse if frame >= 50 else (0.0, 0.0, 0.0)
                            )
                            target = velocity.propose(
                                q, v, ep.layout.angular_velocity_body(data), command, frame=frame
                            )
                            counts["velocity"] += 1
                            velocity_rows.append(
                                dict(
                                    observation=velocity.last_observation.copy(),
                                    action=velocity.last_action.copy(),
                                    target=target.copy(),
                                    command_body=command.copy(),
                                    phase=velocity.phase,
                                )
                            )
                            applied = integrate_velocity_target(ep, target, kp, kd)
                            physical.append(applied.row)
                            controller.append(0)
                            if active_shadow:
                                shadow_queries.append(
                                    dict(
                                        observation=context.observation,
                                        proposal=proposal,
                                        h_before=context.h_before,
                                        c_before=context.c_before,
                                        h_after=context.h_after,
                                        c_after=context.c_after,
                                        actual_target_hg=target.copy(),
                                        proposal_executed=False,
                                        committed_for_suffix=not ep.done,
                                        reference_frame=ep.frame,
                                    )
                                )
                            if ep.done:
                                break
                            if active_shadow:
                                shadow.commit_velocity(applied)
                            else:
                                shadow.adopt_before_shadow(applied)
                        if velocity_rows:
                            artifacts["velocity_NN"] = store(
                                "velocity-NN",
                                {
                                    k: np.array([r[k] for r in velocity_rows])
                                    for k in velocity_rows[0]
                                },
                            )
                        if shadow_queries:
                            artifacts["shadow"] = store(
                                "R2-shadow",
                                {
                                    k: np.array([r[k] for r in shadow_queries])
                                    for k in shadow_queries[0]
                                },
                            )
                        if not ep.done:
                            state = shadow.suffix_state()
                    teacher = None
                    if arm == "run_shadow_taskspace_teacher" and not ep.done:
                        t = plan["specifications"]["teachers"]["4"]
                        intent = FootTaskSpaceIntent(
                            ep.env.model,
                            foot="left" if "left" in case["source"] else "right",
                            center=t["center"],
                            latent=np.array(t["latent"], np.float64),
                        )
                        teacher = OwnStateTeacherHandoff(
                            ep, actor.base, intent=intent, switch_control=80
                        )
                    neural = []
                    plans = []
                    while not ep.done and ep.frame < ep.clip.length:
                        obs = ep.observation()
                        before = [s.numpy().copy() for s in state]
                        with torch.no_grad():
                            mean, state = active_actor(
                                torch.from_numpy(obs).reshape(1, 1, 160), state
                            )
                        counts["R2_applied"] += 1
                        nominal = mean[0, 0].numpy().copy()
                        action = nominal.copy()
                        if teacher is not None:
                            context = CausalMotorContext(
                                id(ep),
                                ep.frame,
                                obs,
                                before[0],
                                before[1],
                                state[0].numpy(),
                                state[1].numpy(),
                            )
                            before_plan = snapshot(data)
                            action, tp = teacher(ep, nominal.copy(), context)
                            context.assert_current(ep)
                            if any(
                                not np.array_equal(v, snapshot(data)[k])
                                for k, v in before_plan.items()
                            ):
                                raise ValueError("teacher changed physical state")
                            plans.append(tp)
                            counts["teacher_IK"] += 1
                        neural.append(
                            dict(
                                observation=obs,
                                proposal=nominal,
                                applied_action=action.copy(),
                                h_before=before[0],
                                c_before=before[1],
                                h_after=state[0].numpy().copy(),
                                c_after=state[1].numpy().copy(),
                                reference_frame=ep.frame,
                            )
                        )
                        physical.append(ep.step(action))
                        controller.append(2 if teacher is not None else 1)
                        if ep.frame == ep.clip.length and not ep.done:
                            ep.admit_reference_recovery()
                    kick_end_native = len(physical) * 10
                    if neural:
                        artifacts["R2_applied"] = store(
                            "R2-applied", {k: np.array([r[k] for r in neural]) for k in neural[0]}
                        )
                    if plans:
                        artifacts["teacher_plans"] = store(
                            "teacher-plans", {k: np.array([r[k] for r in plans]) for k in plans[0]}
                        )
                    stand = None
                    if ep.termination_reason == "RECOVERY_ADMITTED":

                        def inference():
                            counts["R2_stand"] += 1

                        task = NeuralStandingTask(
                            ep,
                            actor.base,
                            state=state,
                            memory_mode="carry",
                            count_inference=inference,
                        )
                        sr, stand = task.collect(100)
                        artifacts["stand"] = store("stand", sr)
                        # Only actual physical rows; no neural/ref tail padding.
                        for i in range(len(sr["qpos"])):
                            physical.append(
                                {
                                    k: sr[k][i]
                                    for k in ("qpos", "qvel", "forces", "torque", "body_failed")
                                }
                            )
                            controller.append(3)
                    raw = {
                        k: np.array([r[k] for r in physical])
                        for k in ("qpos", "qvel", "forces", "torque", "body_failed")
                    }
                    raw["controller"] = np.array(controller, np.int32)
                    artifacts["physical"] = store("physical", raw)
                    force = raw["forces"].reshape(-1, 3)
                    post = native_contact_window(
                        force=force[:, :2],
                        canonical_qpos=raw["qpos"].reshape(-1, 43),
                        canonical_qvel=raw["qvel"].reshape(-1, 41),
                        goal=ep.goal,
                        intended_foot=0 if "left" in case["source"] else 1,
                    )
                    result = dict(
                        body_failed=bool(raw["body_failed"].any()),
                        forbidden=bool((force[:, 2] > 1).any()),
                        postcontact=post,
                        nonfoot_impulse_Ns=float(force[:, 2].sum() * 0.002),
                    )
                    first = post["first_foot_contact_substep"]
                    first_mode = (
                        int(raw["controller"][int(first) // 10]) if first is not None else None
                    )
                    quality = contact_quality(result)
                    outcome = dict(
                        course=ci,
                        arm=arm,
                        repeat=repeat,
                        quality=quality,
                        healthy_success=bool(
                            quality and stand and stand["healthy"] and stand["stability_observed"]
                        ),
                        result=result,
                        stand=stand,
                        first_contact_controller=first_mode,
                        first_contact_before_kick_end=bool(
                            first is not None and int(first) < kick_end_native
                        ),
                        total_native_steps=len(physical) * 10,
                        actual_world_time_s=float(data.time),
                        R2_shadow_queries=len(shadow_queries),
                        R2_shadow_proposals_executed=False,
                        artifacts=artifacts,
                        official_exam_result=False,
                        learned_player_claim=False,
                        source_execution_head=a.expected_head,
                        recovered_without_new_physics=False,
                    )
                    write_once(folder / "complete.json", outcome)
                    if previous is not None:
                        for k in blobs:
                            identical(blobs[k], previous[k])
                    else:
                        previous = blobs
                    records.append(outcome)
                    print(
                        json.dumps(
                            {
                                k: outcome[k]
                                for k in (
                                    "course",
                                    "arm",
                                    "repeat",
                                    "quality",
                                    "healthy_success",
                                    "first_contact_controller",
                                    "total_native_steps",
                                )
                            }
                        ),
                        flush=True,
                    )
                    if (
                        bench.native_steps + reused_steps > native_budget
                        or sum(
                            counts[k] for k in ("velocity", "R2_shadow", "R2_applied", "R2_stand")
                        )
                        > query_budget
                    ):
                        raise ValueError("bounded pilot budget exceeded")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_VELOCITY_SHADOW_CONTROL_PILOT_CLOSED_NOT_LEARNED_QUALIFICATION",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                records=records,
                counts=counts,
                actual_native_steps=bench.native_steps + reused_steps,
                new_native_steps=bench.native_steps,
                reused_native_steps=reused_steps,
                optimizer_updates=0,
                whole_repeats_exact=True,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        partial = {}
        if ep is not None:
            path = a.output / "failed-live-state.npz"
            np.savez_compressed(path, **snapshot(ep.env.data))
            partial["live_state"] = dict(path=str(path), hash=digest(path))
            if physical:
                path = a.output / "failed-physical-prefix.npz"
                np.savez_compressed(
                    path,
                    **{
                        k: np.array([r[k] for r in physical])
                        for k in ("qpos", "qvel", "forces", "torque", "body_failed")
                    },
                )
                partial["physical_prefix"] = dict(path=str(path), hash=digest(path))
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                counts=counts,
                partial_artifacts=partial,
                actual_native_steps=(bench.native_steps if bench else 0) + reused_steps,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
