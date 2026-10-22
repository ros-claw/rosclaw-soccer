"""Own-history full native audit of every assisted TRAIN recovery, including failures.

This certifies execution, not teacher competence or learner success. A queued
audit may bind a frozen producer head before its final receipt exists.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("producer-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--source-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen audit source required")

    clean()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (rp, dp))
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("closed source and unused audit output required")
    source_hash = digest(rp)
    if a.source_hash is not None and source_hash != a.source_hash:
        raise ValueError("exact completed source receipt required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    mode = d.get("teacher_mode", "reference")
    protocols = {
        "reference": "dual256.TRAIN.actual_failed_state_shared_authority_recovery.v1",
        "measured_contact_center": "dual256.TRAIN.measured_contact_center_on_learner_mean.v2",
        "measured_contact_shin_clearance": (
            "dual256.TRAIN.measured_contact_center_and_shin_clearance.v3"
        ),
        "measured_contact_observed_shin": (
            "dual256.TRAIN.measured_contact_center_observed_shin_predictive.v4"
        ),
    }
    if (
        mode not in protocols
        or d["protocol"] != protocols[mode]
        or r["status"] != "DUAL_MOTOR_ACTUAL_STATE_TEACHER_RECOVERY_CLOSED_PENDING_OWN_AUDIT"
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or d["source_context_ids"] != [4, 5, 20, 21]
        or d["alphas"] != [0.25, 0.5, 0.75, 1.0]
        or d["start_control"] != 80
        or d["ramp_controls"] != 8
        or d["selected_repeats"] != 2
        or len(r["records"]) != 24
        or not r["whole_selected_repeats_exact"]
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "learner_success_claim")
        )
        or r["PPO_actor_rows"] != 0
        or d["online_PPO_actor_rows"] != 0
    ):
        raise ValueError("complete explicitly assisted24-world TRAIN recovery required")
    pins = dict(r["input_file_hashes"])
    for x in (rp, dp):
        pins[str(x)] = digest(x)

    def receipt(h):
        paths = [
            Path(x) for x, value in pins.items() if value == h and x.endswith("/complete.json")
        ]
        if len(paths) != 1:
            raise ValueError("unique source-bound receipt required")
        return paths[0], json.loads(paths[0].read_text())

    tp, training = receipt(r["training_hash"])
    pp, parent_record = receipt(training["parent_lineage_hash"])
    _, data = receipt(training["data_hash"])
    teacher_path, teacher = receipt(data["teacher_hash"])
    tdpath = teacher_path.parent / "declaration.json"
    if pins.get(str(tdpath)) != digest(tdpath):
        raise ValueError("source-bound known intent declaration required")
    td = json.loads(tdpath.read_text())
    exam_path, exam = receipt(r["exam_hash"])
    _, original_audit = receipt(d["audit_hash"])
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(native=0, neural=0, kinematic_intent=0)
    verified = []
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
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality, stand_from_raw
        from rosclaw_soccer.rsi.paid_run_teacher_audit import equal, replay_teacher_world
        from rosclaw_soccer.rsi.run_contact_geometry import summarize_contacts
        from rosclaw_soccer.rsi.run_teacher_objective import run_teacher_rank

        def plain(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            return v

        def load_npz(path):
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [x["case"] for x in d["contexts"]])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=pp,
            training_receipt_hash=training["parent_lineage_hash"],
            scene_hash=scene,
        )
        actor, _ = load_dual_for_sim_audit(
            parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=tp.parent,
            expected_receipt_hash=r["training_hash"],
        )
        scale = actor.parent.get_buffer("action_scale").numpy().copy()
        seen, selected_tags, winners = set(), [], []
        for context, ctx in zip(d["source_context_ids"], d["contexts"], strict=True):
            candidates = [x for x in r["records"] if x["source_context"] == context]
            if len(candidates) != 6:
                raise ValueError("all four trials and two selected repetitions required")
            trials = candidates[:4]
            if [x["alpha"] for x in trials] != d["alphas"]:
                raise ValueError("predeclared trial order required")
            best = max(trials, key=run_teacher_rank)
            selected_tags.append(best["tag"])
            winners.append(best)
            best_blobs = None
            original = next(
                x for x in exam["records"] if x["context"] == context and x["repeat"] == 0
            )
            oldpath = exam_path.parent / original["artifacts"]["kick"]["path"]
            if digest(oldpath) != original["artifacts"]["kick"]["hash"]:
                raise ValueError("original failed prefix exact binding required")
            old = load_npz(oldpath)
            for slot, rec in enumerate(candidates):
                expected_tag = (
                    f"context{context}-alpha{d['alphas'][slot]}"
                    if slot < 4
                    else f"selected-context{context}-repeat{slot - 4}"
                )
                if rec["tag"] != expected_tag or rec["tag"] in seen or rec["PPO_actor_rows"] != 0:
                    raise ValueError("distinct explicit trial identity required")
                seen.add(rec["tag"])
                blobs, saved_plans = {}, None
                for name, art in rec["artifacts"].items():
                    path = a.root / art["path"]
                    if (
                        path.is_symlink()
                        or not path.resolve().is_relative_to(a.root.resolve())
                        or digest(path) != art["hash"]
                    ):
                        raise ValueError("exact confined source primitives required")
                    pins[str(path)] = art["hash"]
                    if name == "plans":
                        saved_plans = json.loads(path.read_text())["plans"]
                    else:
                        blobs[name] = load_npz(path)
                if rec["tag"] == best["tag"]:
                    best_blobs = blobs
                if slot >= 4:
                    if (
                        rec["alpha"] != best["alpha"]
                        or best_blobs is None
                        or set(blobs) != set(best_blobs)
                    ):
                        raise ValueError("complete selected repetition required")
                    for name in blobs:
                        identical(blobs[name], best_blobs[name])
                kick, stand = blobs["kick"], blobs.get("stand")
                for k in (
                    "NN_obs",
                    "NN_nominal_action",
                    "NN_h_before",
                    "NN_c_before",
                    "qpos",
                    "qvel",
                    "forces",
                    "torque",
                ):
                    source_key = "NN_action" if k == "NN_nominal_action" else k
                    equal(kick[k][:80], old[source_key][:80], "actual failed prefix " + k)
                ep = bench.make(ctx["case"])
                if ctx["backoff_m"]:
                    q, _ = ep.layout.canonical(ep.env.data)
                    w, x, y, z = q[3:7]
                    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                    ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= ctx[
                        "backoff_m"
                    ] * np.array([np.cos(yaw), np.sin(yaw)])
                    mujoco.mj_forward(ep.env.model, ep.env.data)
                intent = None
                world_offset = center = None
                if mode.startswith("measured_contact_"):
                    geo = next(
                        x
                        for x in original_audit["contact_event_artifacts"]
                        if x["context"] == context and x["repeat"] == 0
                    )
                    paths = [Path(x) for x, h in pins.items() if h == geo["hash"]]
                    if len(paths) != 1:
                        raise ValueError("exact certified geometry source required")
                    world_offset, center = measured_centering(
                        json.loads(paths[0].read_text())["events"], ep.goal
                    )
                    intent = FootTaskSpaceIntent(
                        ep.env.model, foot="left", center=center, latent=np.zeros((2, 6))
                    )
                elif context >= 20:
                    tr = next(
                        x
                        for x in teacher["records"]
                        if x["tag"] == f"selected-c{context - 20}-repeat0"
                    )
                    intent = FootTaskSpaceIntent(
                        ep.env.model,
                        foot="left",
                        center=td["centers"][context - 20],
                        latent=np.array(tr["latent"], np.float64),
                    )
                plans = []

                def adapter(
                    actual,
                    nominal,
                    evidence,
                    *,
                    rec=rec,
                    kick=kick,
                    intent=intent,
                    world_offset=world_offset,
                    center=center,
                    plans=plans,
                ):
                    frame = actual.frame
                    ref = evidence["reference_nominal"][0, 0].numpy().copy()
                    reference = evidence["reference_mean"][0, 0].numpy().copy()
                    share = assistance_share(frame, actual.clip.length, float(rec["alpha"]))
                    equal(ref, kick["NN_reference_nominal"][frame], "own reference nominal")
                    equal(reference, kick["NN_reference_mean"][frame], "own frozen observer")
                    equal(share, kick["NN_assistance_share"][frame], "declared actual share")
                    if not share:
                        return nominal.copy(), 0
                    if mode.startswith("measured_contact_"):
                        rotation = (
                            actual.env.data.xmat[actual.env.pelvis_body_id].reshape(3, 3).copy()
                        )
                        oracle, ik = intent.solve_offset(
                            actual,
                            nominal.copy(),
                            current_body_offset(world_offset, rotation, frame, center),
                            shin_clearance=mode
                            in (
                                "measured_contact_shin_clearance",
                                "measured_contact_observed_shin",
                            ),
                            observed_shin_clearance=mode == "measured_contact_observed_shin",
                        )
                    elif intent is None:
                        oracle, ik = reference, None
                    else:
                        oracle, ik = intent(actual, ref.copy())
                    action, info = blend_actual_motor(nominal, oracle, ref, scale, share=share)
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
                    return action, int(intent is not None)

                events, before = [], dict(counts)
                replay_teacher_world(
                    ep,
                    actor,
                    kick,
                    stand,
                    latent=np.zeros((2, 6)),
                    center=105,
                    foot="left",
                    counts=counts,
                    hidden_width=256,
                    action_adapter=adapter,
                    contact_observer=events.append,
                )
                if (
                    plans != saved_plans
                    or counts["native"] - before["native"] != rec["native_steps"]
                    or counts["neural"] - before["neural"] != rec["actor_decisions"]
                ):
                    raise ValueError("all own plans/actions/actual work must match")
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
                    intended_foot=0,
                )
                result = dict(
                    body_failed=bool(raw["body_failed"].any()),
                    forbidden=bool((f[:, 2] > 1).any()),
                    nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
                    postcontact=post,
                )
                ss = stand_from_raw(stand) if stand is not None else None
                good = bool(
                    contact_quality(result) and ss and ss["healthy"] and ss["stability_observed"]
                )
                if result != rec["result"] or ss != rec["stand"] or good != rec["healthy_quality"]:
                    raise ValueError("whole task grade differs")
                path = a.output / (rec["tag"] + "-contacts.json")
                write_once(path, dict(events=events, summary=summarize_contacts(events, ep.goal)))
                verified.append(
                    dict(
                        tag=rec["tag"],
                        context=context,
                        quality=good,
                        contact_path=path.name,
                        contact_hash=digest(path),
                    )
                )
                print(
                    json.dumps(
                        dict(
                            world=len(verified),
                            tag=rec["tag"],
                            quality=good,
                            native=counts["native"],
                        )
                    ),
                    flush=True,
                )
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_actor_decisions"]
            or counts["kinematic_intent"] != r["actual_foot_intent_calls"]
            or selected_tags != r["selected_tags"]
            or sum(x["healthy_quality"] for x in winners) != r["selected_teacher_successes"]
        ):
            raise ValueError("aggregate work and selection accounting differs")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_MOTOR_ASSISTED24_OWN_NATIVE_AUDIT_PASS",
                source_hash=source_hash,
                source_head=a.expected_head,
                input_file_hashes=pins,
                counts=counts,
                worlds=24,
                results=verified,
                teacher_mode=mode,
                selected_teacher_successes=r["selected_teacher_successes"],
                independent_whole_world_replay=True,
                shared_observation_and_IK_implementation=True,
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
                counts=counts,
                worlds=len(verified),
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
