"""TRAIN recovery from the rejected learner's actual private dual-state paths.

Frozen reference observer or known foot intent is queried on current real SIM
state. Assistance is smoothly shared; no old teacher prefix/h/c is imported.
All outcomes remain teacher-assisted data, not learner success or PPO samples.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--teacher-mode",
        choices=("reference", "measured_contact_center", "measured_contact_shin_clearance"),
        default="reference",
    )
    for n in ("exam-root", "audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("audit-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen actual-state recovery source required")

    clean()
    rp, ap, dp = (
        a.exam_root / "complete.json",
        a.audit_root / "complete.json",
        a.exam_root / "declaration.json",
    )
    if a.output.exists() or any(x.is_symlink() for x in (rp, ap, dp)) or digest(ap) != a.audit_hash:
        raise ValueError("unused recovery output and exact complete own-native witness required")
    r, ar, d = [json.loads(x.read_text()) for x in (rp, ap, dp)]
    if (
        ar["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or ar["source_hash"] != digest(rp)
        or ar["worlds"] != 44
        or ar["counts"]["native"] != r["actual_native_steps"]
        or not ar["independent_whole_world_replay"]
        or r["declaration_hash"] != digest(dp)
        or r["old_protected_retained"] is not False
        or ar["hardware_authorized"] is not False
        or ar["promotion_authorized"] is not False
    ):
        raise ValueError("actual rejected learned policy and all44 certified worlds required")
    pins = dict(ar["input_file_hashes"])
    for x in (rp, ap, dp):
        pins[str(x)] = digest(x)
    contact_inputs = {}
    if a.teacher_mode.startswith("measured_contact_"):
        for i in (4, 5, 20, 21):
            entry = next(
                x for x in ar["contact_event_artifacts"] if x["context"] == i and x["repeat"] == 0
            )
            path = a.audit_root / entry["path"]
            if path.is_symlink() or path.parent != a.audit_root or digest(path) != entry["hash"]:
                raise ValueError("exact audited contact geometry required")
            pins[str(path)] = digest(path)
            contact_inputs[i] = json.loads(path.read_text())["events"]

    def receipt(h):
        paths = [Path(x) for x, v in pins.items() if v == h and x.endswith("/complete.json")]
        if len(paths) != 1:
            raise ValueError("unique exact recovery lineage required")
        return paths[0], json.loads(paths[0].read_text())

    training_path, training = receipt(d["training_hash"])
    parent_path, parent_record = receipt(training["parent_lineage_hash"])
    _, data = receipt(training["data_hash"])
    teacher_path, teacher = receipt(data["teacher_hash"])
    td = json.loads((teacher_path.parent / "declaration.json").read_text())
    selected = [4, 5, 20, 21]
    baselines = [
        next(x for x in r["records"] if x["context"] == i and x["repeat"] == 0) for i in selected
    ]
    if any(x["healthy_quality"] for x in baselines):
        raise ValueError("predeclared actual failed contexts only")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.TRAIN.actual_failed_state_shared_authority_recovery.v1"
            if a.teacher_mode == "reference"
            else (
                "dual256.TRAIN.measured_contact_center_on_learner_mean.v2"
                if a.teacher_mode == "measured_contact_center"
                else "dual256.TRAIN.measured_contact_center_and_shin_clearance.v3"
            ),
            teacher_mode=a.teacher_mode,
            centering_geometry=(
                "actual solver radial minus horizontal goal-directed .11m radial; "
                "not force direction"
            ),
            centering_support_controls=24,
            shin_clearance_margin_m=0.015
            if a.teacher_mode == "measured_contact_shin_clearance"
            else None,
            source_head=a.expected_head,
            input_file_hashes=pins,
            exam_hash=digest(rp),
            audit_hash=a.audit_hash,
            training_hash=d["training_hash"],
            contexts=[d["contexts"][i] for i in selected],
            source_context_ids=selected,
            alphas=[0.25, 0.5, 0.75, 1.0],
            start_control=80,
            ramp_controls=8,
            ramp_out_before_standing=True,
            maximum_worlds=24,
            maximum_native_steps=96000,
            maximum_actor_decisions=9600,
            selected_repeats=2,
            learner_success_claim=False,
            online_PPO_actor_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    calls = 0
    intent_calls = 0
    worlds = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.contact_center_teacher import (
            current_body_offset,
            measured_centering,
        )
        from rosclaw_soccer.rsi.dual_motor_checkpoint import load_dual_for_sim_audit
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.own_state_motor_assistance import (
            assistance_share,
            blend_actual_motor,
        )
        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
        from rosclaw_soccer.rsi.run_teacher_objective import run_teacher_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [d["contexts"][i]["case"] for i in selected])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=parent_path,
            training_receipt_hash=training["parent_lineage_hash"],
            scene_hash=scene,
        )
        actor, _ = load_dual_for_sim_audit(
            parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=training_path.parent,
            expected_receipt_hash=d["training_hash"],
        )
        scale = actor.parent.get_buffer("action_scale").numpy().copy()
        records = []

        def plain(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            return v

        def collect(slot, alpha, tag):
            nonlocal calls, intent_calls, worlds
            if worlds >= 24:
                raise ValueError("bounded recovery-world budget exceeded")
            context = selected[slot]
            ctx = d["contexts"][context]
            case = ctx["case"]
            folder = a.output / tag
            folder.mkdir()
            ep = bench.make(case)
            if ctx["backoff_m"]:
                q, _ = ep.layout.canonical(ep.env.data)
                w, x, y, z = q[3:7]
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= ctx["backoff_m"] * np.array(
                    [np.cos(yaw), np.sin(yaw)]
                )
                mujoco.mj_forward(ep.env.model, ep.env.data)
            baseline = baselines[slot]
            with np.load(
                a.exam_root / baseline["artifacts"]["kick"]["path"], allow_pickle=False
            ) as z:
                expected = {
                    k: z[k].copy()
                    for k in (
                        "NN_obs",
                        "NN_action",
                        "NN_h_before",
                        "NN_c_before",
                        "qpos",
                        "qvel",
                        "forces",
                        "torque",
                    )
                }
            intent = None
            world_offset = center = None
            if a.teacher_mode.startswith("measured_contact_"):
                world_offset, center = measured_centering(contact_inputs[context], ep.goal)
                intent = FootTaskSpaceIntent(
                    ep.env.model, foot="left", center=center, latent=np.zeros((2, 6))
                )
            elif context >= 20:
                source = next(
                    x for x in teacher["records"] if x["tag"] == f"selected-c{context - 20}-repeat0"
                )
                intent = FootTaskSpaceIntent(
                    ep.env.model,
                    foot="left" if "left" in case["source"] else "right",
                    center=td["centers"][context - 20],
                    latent=np.array(source["latent"], np.float64),
                )
            private = actor.initial_state()
            initial = snapshot(ep.env.data)
            rows = []
            plans = []
            while not ep.done and ep.frame < ep.clip.length:
                frame = ep.frame
                obs = ep.observation()
                before = [v.numpy().copy() for v in private]
                with torch.no_grad():
                    mean, private, e = actor.forward_with_evidence(
                        torch.from_numpy(obs).reshape(1, 1, 160), private
                    )
                calls += 1
                nominal = mean[0, 0].numpy().copy()
                ref = e["reference_nominal"][0, 0].numpy().copy()
                reference = e["reference_mean"][0, 0].numpy().copy()
                if frame <= 80 and (
                    not np.array_equal(obs, expected["NN_obs"][frame])
                    or not np.array_equal(nominal, expected["NN_action"][frame])
                    or not np.array_equal(before[0], expected["NN_h_before"][frame])
                    or not np.array_equal(before[1], expected["NN_c_before"][frame])
                ):
                    raise ValueError(
                        "own replayed learner prefix changed; no borrowed state repair"
                    )
                share = assistance_share(frame, ep.clip.length, float(alpha))
                action = nominal.copy()
                if share > 0:
                    physical = snapshot(ep.env.data)
                    if a.teacher_mode.startswith("measured_contact_"):
                        rotation = ep.env.data.xmat[ep.env.pelvis_body_id].reshape(3, 3).copy()
                        offset = current_body_offset(world_offset, rotation, frame, center)
                        oracle, ik = intent.solve_offset(
                            ep,
                            nominal.copy(),
                            offset,
                            shin_clearance=a.teacher_mode == "measured_contact_shin_clearance",
                        )
                        intent_calls += 1
                    elif intent is None:
                        oracle, ik = reference, None
                    else:
                        oracle, ik = intent(ep, ref.copy())
                        intent_calls += 1
                    action, info = blend_actual_motor(nominal, oracle, ref, scale, share=share)
                    if any(
                        not np.array_equal(v, snapshot(ep.env.data)[k]) for k, v in physical.items()
                    ):
                        raise ValueError("oracle changed actual body state")
                    plans.append(
                        plain(
                            dict(
                                frame=frame,
                                share=share,
                                oracle_requested=oracle,
                                applied=action,
                                reference_nominal=ref,
                                reference_mean=reference,
                                information=info,
                                foot_plan=ik,
                            )
                        )
                    )
                row = ep.step(action)
                if frame < 80 and any(
                    not np.array_equal(row[k], expected[k][frame])
                    for k in ("qpos", "qvel", "forces", "torque")
                ):
                    raise ValueError("actual learner prefix physics differs from certified failure")
                row.update(
                    NN_obs=obs,
                    NN_nominal_action=nominal,
                    NN_action=action.copy(),
                    NN_h_before=before[0],
                    NN_c_before=before[1],
                    NN_h_after=private[0].numpy().copy(),
                    NN_c_after=private[1].numpy().copy(),
                    NN_reference_nominal=ref,
                    NN_reference_mean=reference,
                    NN_assistance_share=share,
                )
                rows.append(row)
                if ep.frame == ep.clip.length and not ep.done:
                    ep.admit_reference_recovery()
            kick = {k: np.array([x[k] for x in rows]) for k in rows[0]}
            kick.update({"initial_" + k: v for k, v in initial.items()})
            kick.update({"terminal_" + k: v for k, v in snapshot(ep.env.data).items()})
            stand = summary = None
            if ep.termination_reason == "RECOVERY_ADMITTED":

                def count():
                    nonlocal calls
                    calls += 1

                stand, summary = NeuralStandingTask(
                    ep,
                    actor,
                    state=private,
                    memory_mode="carry",
                    count_inference=count,
                    hidden_width=256,
                ).collect(100)
            blobs = dict(kick=kick)
            if stand is not None:
                blobs["stand"] = stand
            artifacts = {}
            for name, raw in blobs.items():
                path = folder / (name + ".npz")
                np.savez_compressed(path, **raw)
                artifacts[name] = dict(path=str(path.relative_to(a.output)), hash=digest(path))
            planpath = folder / "oracle-plans.json"
            write_once(planpath, dict(plans=plans))
            artifacts["plans"] = dict(
                path=str(planpath.relative_to(a.output)), hash=digest(planpath)
            )
            raw = {
                k: np.concatenate([kick[k], stand[k]]) if stand is not None else kick[k]
                for k in ("qpos", "qvel", "forces", "body_failed")
            }
            f = raw["forces"].reshape(-1, 3)
            post = native_contact_window(
                force=f[:, :2],
                canonical_qpos=raw["qpos"].reshape(-1, 43),
                canonical_qvel=raw["qvel"].reshape(-1, 41),
                goal=ep.goal,
                intended_foot=0 if "left" in case["source"] else 1,
            )
            result = dict(
                body_failed=bool(raw["body_failed"].any()),
                forbidden=bool((f[:, 2] > 1).any()),
                nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
                postcontact=post,
            )
            good = bool(
                summary
                and summary["healthy"]
                and summary["stability_observed"]
                and not result["body_failed"]
                and not result["forbidden"]
                and post["correct_foot_first"]
                and post["post_contact_quality_observed"]
                and post["goal_progress_1p2s_m"] >= 1
                and abs(post["goal_lateral_1p2s_m"]) <= 0.75
                and not post["other_foot_contact_in_window"]
                and post["maximum_pelvis_retreat_after_contact_m"] <= 0.05
            )
            outcome = dict(
                tag=tag,
                source_context=context,
                alpha=alpha,
                oracle_mode=a.teacher_mode
                if a.teacher_mode != "reference"
                else ("reference" if intent is None else "reference_with_known_intent"),
                result=result,
                stand=summary,
                healthy_quality=good,
                native_steps=len(raw["qpos"]) * 10,
                actor_decisions=len(rows) + (len(stand["qpos"]) if stand is not None else 0),
                assisted=alpha > 0,
                learner_success_claim=False,
                PPO_actor_rows=0,
                prefix_through79_exact=True,
                artifacts=artifacts,
            )
            write_once(folder / "complete.json", outcome)
            records.append(outcome)
            worlds += 1
            print(
                json.dumps(
                    dict(
                        world=worlds,
                        context=context,
                        alpha=alpha,
                        teacher_quality=good,
                        forbidden=result["forbidden"],
                        lateral=post["goal_lateral_1p2s_m"],
                    )
                ),
                flush=True,
            )
            if bench.native_steps > 96000 or calls > 9600:
                raise ValueError("bounded recovery native/NN budget exceeded")
            return outcome, blobs, plans

        selected_trials = []
        for slot in range(4):
            best = None
            for alpha in (0.25, 0.5, 0.75, 1.0):
                trial = collect(slot, alpha, f"context{selected[slot]}-alpha{alpha}")
                if best is None or run_teacher_rank(trial[0]) > run_teacher_rank(best[0]):
                    best = trial
            selected_trials.append(best)
            for repeat in range(2):
                actual = collect(
                    slot, best[0]["alpha"], f"selected-context{selected[slot]}-repeat{repeat}"
                )
                if set(actual[1]) != set(best[1]):
                    raise ValueError("selected whole recovery termination differs")
                for name in actual[1]:
                    identical(actual[1][name], best[1][name])
                if actual[2] != best[2]:
                    raise ValueError("selected actual oracle plans differ")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_MOTOR_ACTUAL_STATE_TEACHER_RECOVERY_CLOSED_PENDING_OWN_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                exam_hash=digest(rp),
                training_hash=d["training_hash"],
                records=records,
                selected_tags=[x[0]["tag"] for x in selected_trials],
                selected_teacher_successes=sum(x[0]["healthy_quality"] for x in selected_trials),
                actual_native_steps=bench.native_steps,
                actual_actor_decisions=calls,
                recurrent_branch_queries=2 * calls,
                actual_foot_intent_calls=intent_calls,
                worlds=worlds,
                whole_selected_repeats_exact=True,
                learner_success_claim=False,
                PPO_actor_rows=0,
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
                actual_native_steps=bench.native_steps if bench else 0,
                actual_actor_decisions=calls,
                worlds=worlds,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
