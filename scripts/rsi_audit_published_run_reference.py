"""Independent whole-world replay of the inherited-base run pilot.

Reuses the admitted observation codec, not producer integration/shadow/standing
controllers. Recomputes every neural call and integrates every native step.
The legacy producer called actor.base published_R2, but its weights were
previously distilled. Preserve that receipt and explicitly correct provenance.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, required=True, type=Path)
    for n in ("source-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if subprocess.check_output(["git", "status", "--porcelain"]) or (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean pinned auditor required")

    clean()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if a.output.exists() or rp.is_symlink() or dp.is_symlink() or digest(rp) != a.source_hash:
        raise ValueError("exact closed source and unused audit output required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    if (
        (a.root / "failed.json").exists()
        or r["declaration_hash"] != digest(dp)
        or r["status"] != "PAID_VELOCITY_SHADOW_CONTROL_PILOT_CLOSED_NOT_LEARNED_QUALIFICATION"
        or r["source_head"] != d["source_head"]
        or d["protocol"] != "paid.velocity_actual_shadow_then_reference80.published_R2_v2"
        or d["motor_reference"] != "published_R2"
        or d["courses"] != [dict(backoff_m=1.6, speed_mps=0.8), dict(backoff_m=1.2, speed_mps=0.6)]
        or d["arms"] != ["reference_from_start", "velocity_shadow_reference"]
        or d["repeats"] != 2
        or d["prepare_velocity_controls"] != 50
        or d["run_controls"] != 100
        or d["shadow_reference_controls"] != 80
        or r["reused_native_steps"] != 0
        or r["actual_native_steps"] != 33920
        or r["whole_repeats_exact"] is not True
        or r["input_file_hashes"] != d["input_file_hashes"]
        or len(r["records"]) != 8
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("exact published-R2 two-course comparison required")
    pins = dict(r["input_file_hashes"])
    for x in (rp, dp):
        pins[str(x)] = digest(x)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    steps, calls = 0, 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.physics.g1_velocity_observation import world_command_to_body
        from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality, stand_from_raw
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if mujoco.__version__ != "3.5.0" or mujoco.get_mjcb_control() is not None:
            raise ValueError("pinned callback-free native engine required")
        bench, policy, scene = open_paid_bench(pins, [d["case"]])
        candidates = []
        for path in pins:
            if (
                path.endswith("/complete.json")
                and "/rsi-paid-canonical-pipeline-20261022/learned-trained/" in path
            ):
                candidates.append(Path(path))
        if len(candidates) != 1:
            raise ValueError("unique source-bound canonical motor lineage required")
        training_path = candidates[0]
        training = json.loads(training_path.read_text())
        wrapper, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=training_path,
            training_receipt_hash=pins[str(training_path)],
            scene_hash=scene,
        )
        actor = wrapper.base
        pristine = PaidTrainableActor(policy, expected_hash=PUBLISHED_HASH).eval()
        changed_tensors = [
            k for k, v in actor.state_dict().items() if not torch.equal(v, pristine.state_dict()[k])
        ]
        if not changed_tensors:
            raise ValueError("legacy identity correction must bind the actual distilled motor")
        onnx = next(Path(x) for x in pins if x.endswith("/exported/policy.onnx"))
        cfg = next(Path(x) for x in pins if x.endswith("/params/deploy.yaml"))

        def equal(x, y):
            if not np.array_equal(x, y):
                raise ValueError("independent measured primitive differs")

        seen = set()
        previous = {}
        results = []
        for rec in r["records"]:
            ci, arm, repeat = [rec[k] for k in ("course", "arm", "repeat")]
            key = (ci, arm, repeat)
            if key in seen or ci not in (0, 1) or arm not in d["arms"] or repeat not in (0, 1):
                raise ValueError("exact distinct declared world set required")
            seen.add(key)
            blobs = {}
            for name, art in rec["artifacts"].items():
                x = a.root / art["raw_path"]
                if (
                    x.is_symlink()
                    or not x.resolve().is_relative_to(a.root.resolve())
                    or digest(x) != art["raw_hash"]
                ):
                    raise ValueError("confined committed raw artifact required")
                pins[str(x)] = art["raw_hash"]
                with np.load(x, allow_pickle=False) as z:
                    blobs[name] = {k: z[k].copy() for k in z.files}
            old = previous.get((ci, arm))
            if old is not None:
                if set(old) != set(blobs):
                    raise ValueError("whole repeat artifact set differs")
                for name in blobs:
                    if set(blobs[name]) != set(old[name]):
                        raise ValueError("repeat raw schema differs")
                    for k in blobs[name]:
                        equal(blobs[name][k], old[name][k])
            previous[(ci, arm)] = blobs
            ep = bench.make(d["case"])
            model, data = ep.env.model, ep.env.data
            q, _ = ep.layout.canonical(data)
            w, x, y, z = q[3:7]
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            course = d["courses"][ci]
            data.qpos[ep.layout.rq : ep.layout.rq + 2] -= course["backoff_m"] * direction
            mujoco.mj_forward(model, data)
            for k, v in snapshot(data).items():
                equal(v, blobs["initial"][k])
            physical = blobs["physical"]
            state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
            velocity = G1VelocityPolicy(
                onnx, cfg, expected_model_hash=pins[str(onnx)], expected_config_hash=pins[str(cfg)]
            )
            nvel = 150 if arm == "velocity_shadow_reference" else 0
            expected_modes = np.r_[
                np.zeros(nvel, np.int32),
                np.ones(ep.clip.length - (80 if nvel else 0), np.int32),
                np.full(100, 3, np.int32),
            ]
            equal(physical["controller"], expected_modes)

            def neural(obs, raw, i, prefix=""):
                nonlocal calls, state
                equal(obs, raw[prefix + "observation" if not prefix else "NN_obs"][i])
                for v, k in zip(state, ("h_before", "c_before"), strict=True):
                    equal(v.numpy(), raw[prefix + k][i])
                with torch.no_grad():
                    mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
                calls += 1
                action = mean[0, 0].numpy().copy()
                equal(action, raw["proposal" if not prefix else "NN_action"][i])
                for v, k in zip(state, ("h_after", "c_after"), strict=True):
                    equal(v.numpy(), raw[prefix + k][i])
                return action

            applied = 0
            for control, mode in enumerate(expected_modes):
                if mode == 0:
                    if control >= 70:
                        sr = blobs["shadow"]
                        equal(sr["reference_frame"][control - 70], ep.frame)
                        if (
                            sr["proposal_executed"][control - 70]
                            or not sr["committed_for_suffix"][control - 70]
                        ):
                            raise ValueError("shadow provenance differs")
                        neural(ep.observation(), sr, control - 70)
                    mujoco.mj_forward(model, data)
                    q, v = ep.layout.canonical(data)
                    pulse = tuple(float(course["speed_mps"] * x) for x in direction) + (0.0,)
                    command = world_command_to_body(q, pulse if control >= 50 else (0.0, 0.0, 0.0))
                    target = velocity.propose(
                        q, v, ep.layout.angular_velocity_body(data), command, frame=control
                    )
                    calls += 1
                    vr = blobs["velocity_NN"]
                    for k, value in dict(
                        observation=velocity.last_observation,
                        action=velocity.last_action,
                        target=target,
                        command_body=command,
                        phase=velocity.phase,
                    ).items():
                        equal(value, vr[k][control])
                    if control >= 70:
                        equal(target, sr["actual_target_hg"][control - 70])
                    kp, kd = [
                        np.array(x, np.float64)
                        for x in (velocity.params.stiffness, velocity.params.damping)
                    ]
                    action = np.empty(29, np.float32)
                    action[ep.order] = ((target - ep.offset) / ep.scale).astype(np.float32)
                elif mode == 1:
                    ar = blobs["R2_applied"]
                    equal(ar["reference_frame"][applied], ep.frame)
                    action = neural(ep.observation(), ar, applied)
                    equal(action, ar["applied_action"][applied])
                    applied += 1
                    target, kp, kd = ep.offset + action[ep.order] * ep.scale, ep.kp, ep.kd
                else:
                    mujoco.mj_forward(model, data)
                    obs = ep.build_observation(
                        ep.env,
                        None,
                        0,
                        ep.last_action,
                        ep.ref_pos.astype(np.float32),
                        np.zeros(29, np.float32),
                        np.zeros(3, np.float32),
                        ep.env.ball_pos,
                        ep.goal,
                        ep.offset,
                        obs_dim=None,
                    )
                    action = neural(obs, blobs["stand"], control - len(expected_modes) + 100, "NN_")
                    target, kp, kd = ep.offset + action[ep.order] * ep.scale, ep.kp, ep.kd
                bad = False
                for sub in range(10):
                    q, v = ep.layout.canonical(data)
                    torque = np.minimum(
                        ep.effort, np.maximum(-ep.effort, kp * (target - q[7:36]) - kd * v[6:35])
                    )
                    equal(torque, physical["torque"][control, sub])
                    data.ctrl[ep.layout.ai] = torque
                    clock = float(data.time)
                    mujoco.mj_step(model, data)
                    steps += 1
                    if (
                        abs(data.time - clock - 0.002) > 1e-10
                        or tuple(int(w.number) for w in data.warning) != ep.warning
                    ):
                        raise ValueError("native timing/warning differs")
                    q, v = ep.layout.canonical(data)
                    equal(q, physical["qpos"][control, sub])
                    equal(v, physical["qvel"][control, sub])
                    equal(data.actuator_force[ep.layout.ai], torque)
                    force = np.zeros(3)
                    for j, c in enumerate(data.contact):
                        b1, b2 = [int(model.geom_bodyid[g]) for g in c.geom]
                        other = b2 if b1 == ep.layout.ball else b1 if b2 == ep.layout.ball else -1
                        if other in ep.robot:
                            wrench = np.zeros(6)
                            mujoco.mj_contactForce(model, data, j, wrench)
                            role = ep.feet.index(other) if other in ep.feet else 2
                            force[role] = max(force[role], np.linalg.norm(wrench[:3]))
                    equal(force, physical["forces"][control, sub])
                    bad |= bool(q[2] < 0.65 or 1 - 2 * (q[4] ** 2 + q[5] ** 2) < 0.9)
                equal(bad, physical["body_failed"][control])
                if bad:
                    raise ValueError("declared full healthy comparison was censored")
                ep.last_action = action.copy()
                if mode == 1 or (mode == 0 and control >= 70):
                    ep.ref_pos = ep.clip.joint_pos[ep.frame].copy()
                    ep.ref_vel = ep.clip.joint_vel[ep.frame].copy()
                    ep.ref_ang = ep.clip.body_ang_vel_w[ep.frame, ep.torso_index].copy()
                    ep.frame += 1
                    ep.ready = False
            if abs(data.time - rec["actual_world_time_s"]) > 1e-10:
                raise ValueError("whole world terminal time differs")
            force = physical["forces"].reshape(-1, 3)
            post = native_contact_window(
                force=force[:, :2],
                canonical_qpos=physical["qpos"].reshape(-1, 43),
                canonical_qvel=physical["qvel"].reshape(-1, 41),
                goal=ep.goal,
                intended_foot=0 if "left" in d["case"]["source"] else 1,
            )
            result = dict(
                body_failed=False,
                forbidden=bool((force[:, 2] > 1).any()),
                nonfoot_impulse_Ns=float(force[:, 2].sum() * 0.002),
                postcontact=post,
            )
            own_stand = stand_from_raw(blobs["stand"])
            own_success = bool(
                contact_quality(result) and own_stand["healthy"] and own_stand["stability_observed"]
            )
            if (
                result != rec["result"]
                or contact_quality(result) != rec["quality"]
                or own_stand != rec["stand"]
                or own_success != rec["healthy_success"]
            ):
                raise ValueError("recomputed physical outcome differs")
            results.append(
                dict(
                    course=ci,
                    arm=arm,
                    repeat=repeat,
                    native_steps=len(expected_modes) * 10,
                    outcome=rec["healthy_success"],
                )
            )
            print(json.dumps(results[-1]), flush=True)
        if steps != r["actual_native_steps"] or calls != sum(
            r["counts"][k] for k in ("velocity", "R2_shadow", "R2_applied", "R2_stand")
        ):
            raise ValueError("all actual native and NN work must be accounted")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="INHERITED_BASE_RUN_WHOLE_NATIVE_OWN_NN_AUDIT_PASS",
                legacy_producer_motor_label="published_R2",
                actual_motor="previously_distilled_inherited_base_without_outer_residual",
                pristine_published_R2=False,
                base_tensors_different_from_published=changed_tensors,
                source_hash=a.source_hash,
                source_head=a.expected_head,
                input_file_hashes=pins,
                independent_whole_world_replay=True,
                observation_codec_shared=True,
                own_native_steps=steps,
                own_neural_calls=calls,
                results=results,
                football_qualification=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                own_native_steps=steps,
                own_neural_calls=calls,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
