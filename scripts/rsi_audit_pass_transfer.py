"""Whole32 two-body pass transfer replay on own NN history/manual native PD.

Shared codecs, stroke rule and geometric role selector run on OWN measured
states. No producer advance, saved joint targets/actions or borrowed hidden
state are used. Matching negative outcomes is proof integrity, not success.
"""

import argparse
import json
from pathlib import Path


def verify_json_preflight(expected, recorded):
    """Compare the exact JSON wire contract, including tuple-valued dimensions.

    Dataclass inertia is a tuple in memory and a list in JSON; normalizing only
    this serialization representation does not relax any numeric/hash field.
    """
    if json.loads(json.dumps(expected, allow_nan=False)) != recorded:
        raise ValueError("own whole8 distinct initial physical conditions differ")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--teacher", nargs=4, action="append", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_pass_transfer import admit_teacher
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    rp, dp, pp, xp = [
        a.root / x for x in ("complete.json", "declaration.json", "preflight.json", "scene.xml")
    ]
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (rp, dp, pp, xp))
        or (a.root / "failed.json").exists()
        or digest(rp) != a.source_hash
        or len(a.teacher) != 4
    ):
        raise ValueError("unused audit and exact whole closed two-body transfer required")
    s, d, pf = [json.loads(x.read_text()) for x in (rp, dp, pp)]
    if (
        s["status"] != "PASS_TEACHER_TWO_BODY32_COLLECTED_PENDING_OWN_REPLAY"
        or d["protocol"] != "soccer.audited_pass_teacher_two_body_transfer32.v1"
        or any(r["source_head"] != a.producer_head for r in (s, d))
        or s["declaration_hash"] != digest(dp)
        or s["preflight_hash"] != digest(pp)
        or s["scene_hash"] != digest(xp)
        or len(s["records"]) != 32
        or len(d["courses"]) != 32
        or d["maximum_worlds"] != 32
        or d["max_controls_per_world"] != 400
        or d["maximum_native_steps"] != 128000
        or d["maximum_actor_calls"] != 25600
        or d["formal_pass_receive_controls"] != 300
        or d["required_A_neural_recovery_controls"] != 100
        or d["no_B_kick"] is not True
        or d["actual_applied_action_feedback"] is not True
        or any(
            r[k] is not False
            for r in (s, d)
            for k in ("goal_complete", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("exact unchanged32-world protocol and scoring boundary required")
    pins = dict(s["input_file_hashes"])
    teachers = [admit_teacher(Path(r), h, Path(o), oh, pins) for r, h, o, oh in a.teacher]
    if [r["context"] for r, _ in teachers] != [1, 4, 20, 21]:
        raise ValueError("all4 source teacher contexts required")
    courses = []
    for source, preflight in teachers:
        if [r["task"] for r in source["selections"]] != [0, 1]:
            raise ValueError("both source goals retained required")
        for choice in source["selections"]:
            for side in (-0.55, 0.55):
                for assisted in (False, True):
                    t = choice["task"]
                    courses.append(
                        dict(
                            context=source["context"],
                            task=t,
                            side=side,
                            assisted=assisted,
                            target=preflight["tasks"][t]["goal"],
                            center=preflight["center"],
                            foot=preflight["stroke_foot"],
                            latent=choice["latent"],
                            source_teacher_qualified=choice["passing_result"][
                                "precision_0p1m_moderate_speed_success"
                            ],
                        )
                    )
    if courses != d["courses"] or any(
        r["index"] != i or r["course"] != courses[i] for i, r in enumerate(s["records"])
    ):
        raise ValueError("whole teacher/control course bank differs from audited inputs")
    for path in (rp, dp, pp, xp):
        pins[str(path)] = digest(path)
    for i, row in enumerate(s["records"]):
        if set(row["artifacts"]) != {"initial", "physical", "A", "B"}:
            raise ValueError("all initial/native/two-neural artifacts required")
        for art in row["artifacts"].values():
            path = a.root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (a.root / f"trial{i}").resolve()
                or digest(path) != art["hash"]
            ):
                raise ValueError("all exact confined native and NN artifacts required")
            pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=a.source_hash,
            input_file_hashes=pins,
            independent_native_integrator=True,
            own_neural_history=True,
            shared_codecs_and_teacher_rule_on_own_states=True,
            source_actions_executed=False,
            borrowed_hidden_states=False,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    try:
        from types import SimpleNamespace

        import mujoco
        import numpy as np

        from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
        from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import stand_from_raw
        from rosclaw_soccer.rsi.pass_transfer_outcome import pass_transfer_outcome
        from rosclaw_soccer.rsi.relay_audit_clock import RelayAuditClock
        from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
        from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
        from rosclaw_soccer.sim.contracts import hash_json
        from rosclaw_soccer.training.native_pass_courses import NativePassCourse
        from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
        from scripts.rsi_stroke_search_common import open_motor, stroke_action

        base_hash = d["baseline_training_hash"]
        old = [
            json.loads(Path(path).read_text())
            for path in pins
            if path.endswith("/native-audit/complete.json")
        ]
        old = [r for r in old if r.get("training_hash") == base_hash]
        if len(old) != 1:
            raise ValueError("original consumed native baseline context witness required")
        native_path, native = receipt(pins, old[0]["source_hash"])
        context_path = native_path.parent / "declaration.json"
        if digest(context_path) != native["declaration_hash"]:
            raise ValueError("original context declaration commitment differs")
        contexts = json.loads(context_path.read_text())["contexts"]
        if d["contexts"] != {str(i): contexts[i] for i in (1, 4, 20, 21)}:
            raise ValueError("original physical courses must be independently reconstructed")
        bench, base = open_motor(pins, base_hash, {i: contexts[i] for i in (1, 4, 20, 21)})
        c = d["candidate"]
        if c["seed"] != 65023 or c["kind"] != "contact-consolidated-v2":
            raise ValueError("same frozen retained motor required")
        _, tr = receipt(pins, c["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=tr["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
        if xp.read_text() != setup.xml:
            raise ValueError("own compiled shared scene differs")
        typed = []
        for i in (1, 4, 20, 21):
            for side in (-0.55, 0.55):
                _, w, _, _, _ = setup.prepare(i, side)
                typed.append(
                    NativePassCourse(
                        i,
                        4.0,
                        side,
                        hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
                    )
                )
        expected_pf = dict(
            initial_conditions=8,
            worlds=32,
            course_binding=preflight_receiving_courses(tuple(typed)),
            native_steps=0,
            ball_dimensions=setup.ball_dimensions,
        )
        verify_json_preflight(expected_pf, pf)

        def equal(actual, saved, index):
            if set(actual) != set(saved) or any(
                not np.array_equal(v, saved[k][index]) for k, v in actual.items()
            ):
                raise ValueError(f"own neural/native trace differs at control{index}")

        def load(claim, key):
            with np.load(a.root / claim["artifacts"][key]["path"], allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        prior_initial = None
        for index, course in enumerate(courses):
            claim = s["records"][index]
            ep, initial, forward, home, _ = setup.prepare(course["context"], course["side"])
            w = RelayAuditClock(initial)
            raw, na, nb, ini = [load(claim, k) for k in ("physical", "A", "B", "initial")]
            if (
                set(ini) != {"qpos", "qvel"}
                or not np.array_equal(ini["qpos"], w.data.qpos)
                or not np.array_equal(ini["qvel"], w.data.qvel)
            ):
                raise ValueError("own pre-physics initial state differs")
            if index % 2 and any(not np.array_equal(v, prior_initial[k]) for k, v in ini.items()):
                raise ValueError("paired whole initial states differ")
            prior_initial = ini
            goal = np.array(course["target"])
            anchor = goal[:2] + 0.25 * forward + 0.12 * np.array([-forward[1], forward[0]])
            passer = SharedNativeNeuralPlayer(
                w,
                "a/",
                actor,
                clip=ep.clip,
                torso_index=ep.torso_index,
                build_observation=ep.build_observation,
                goal=goal,
                offset=ep.offset,
                scale=ep.scale,
                order=ep.order,
            )
            velocity = ContinuousVelocityStream(
                setup.velocity_path,
                setup.velocity_config,
                expected_model_hash=pins[str(setup.velocity_path)],
                expected_config_hash=pins[str(setup.velocity_config)],
                horizon_controls=400,
            )
            ready = OnlineReceivingBoundary(w, "b/")
            physical = []
            launch, stand_controls = False, 0
            for step in range(ep.clip.length + 100):
                mujoco.mj_forward(w.model, w.data)
                if passer.phase == "MOTION_COMPLETE":
                    passer.begin_standing()
                standing = passer.phase == "STANDING"
                target_a, ta = passer.propose(with_reference_evidence=True)
                action, delta, clipped = ta["NN_action"].copy(), np.zeros(29), False
                if course["assisted"] and not standing:
                    action, delta, clipped = stroke_action(
                        SimpleNamespace(frame=passer.frame, order=ep.order, scale=ep.scale),
                        action,
                        ta["NN_reference_nominal"],
                        course["center"],
                        np.array(course["latent"]),
                        foot=course["foot"],
                    )
                    target_a = ep.offset + action[ep.order] * ep.scale
                ta.update(
                    control=np.array(step),
                    executed_action=action.copy(),
                    teacher_delta_HG=delta,
                    teacher_authority_clipped=np.array(clipped),
                    standing=np.array(standing),
                )
                equal(ta, na, step)
                q, v = w.layouts["b/"].canonical(w.data)
                qw, x, y, z = q[3:7]
                yaw = float(np.arctan2(2 * (qw * z + x * y), 1 - 2 * (y * y + z * z)))
                intent = receiving_intent(
                    root_xy=q[:2],
                    yaw=yaw,
                    ball_xy=q[36:38],
                    ball_velocity_xy=v[35:37],
                    home_xy=home,
                    pass_direction=forward,
                    launch_seen=launch,
                    ready_anchor_xy=anchor,
                )
                launch = intent.launch_seen
                command = np.array(intent.command_body, np.float32)
                target_b = velocity.propose(
                    q, v, w.layouts["b/"].angular_velocity_body(w.data), command, frame=step
                )
                equal(
                    dict(
                        obs=velocity.last_observation.copy(),
                        action=velocity.last_action.copy(),
                        command=command,
                        phase=np.array(velocity.phase),
                        control=np.array(step),
                    ),
                    nb,
                    step,
                )
                row = w.advance({"a/": target_a, "b/": target_b})
                counts["native_steps"] += 10
                counts["actor_calls"] += 2
                passer.commit(executed_action=action)
                ready.update()
                stand_controls += int(standing)
                own = {k: row[k] for k in ("qpos", "qvel", "torque", "ball_forces")} | dict(
                    body_failed=np.array([row["body_failed"][p] for p in ("a/", "b/")]),
                    target_a=target_a.copy(),
                    target_b=target_b.copy(),
                    time_before=np.array(row["time_before"]),
                )
                equal(own, raw, step)
                physical.append(own)
                if w.failed:
                    break
            if any(len(v) != w.control_steps for r in (raw, na, nb) for v in r.values()):
                raise ValueError("all own controls and complete source rows required")
            ownraw = {k: np.stack([r[k] for r in physical]) for k in physical[0]}
            al = w.layouts["a/"]
            qids = [*range(al.rq, al.rq + 7), *al.qi, *range(al.bq, al.bq + 7)]
            vids = [*range(al.rv, al.rv + 6), *al.vi, *range(al.bv, al.bv + 6)]
            recovery = None
            if stand_controls:
                recovery = stand_from_raw(
                    dict(
                        qpos=ownraw["qpos"][-stand_controls:, :, qids],
                        qvel=ownraw["qvel"][-stand_controls:, :, vids],
                        body_failed=ownraw["body_failed"][-stand_controls:, 0],
                    )
                )
            measure = pass_transfer_outcome(
                ownraw["qpos"][:, :, al.bq : al.bq + 3].reshape(-1, 3),
                ownraw["qvel"][:, :, al.bv : al.bv + 3].reshape(-1, 3),
                ownraw["ball_forces"].reshape(-1, 2, 3),
                goal,
                whole_body_healthy=not w.failed,
                passer_recovery_observed=bool(
                    stand_controls == 100
                    and recovery
                    and recovery["healthy"]
                    and recovery["stability_observed"]
                ),
                receiver_first_ready_control=ready.first_ready_control,
                receiver_excluded_contact=ready.excluded_contact,
            )
            result = dict(
                index=index,
                course=course,
                controls=w.control_steps,
                native_steps=w.native_steps,
                actor_calls=2 * w.control_steps,
                stand_controls=stand_controls,
                A_recovery=recovery,
                outcome=measure,
                termination="BODY_FAILURE" if w.failed else "FULL_MOTION_PLUS100NN_RECOVERY",
                neural_actor_optimizer_updates=0,
            )
            if result != {k: v for k, v in claim.items() if k != "artifacts"}:
                raise ValueError("own recovery/receiving/delivery/whole counters differ")
            records.append(result)
            write_once(a.output / f"trial{index}.json", result)
            print(json.dumps(result), flush=True)
        if counts != s["counts"]:
            raise ValueError("whole32 actual native and NN budget differs")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PASS_TEACHER_TWO_BODY32_WHOLE_OWN_NATIVE_REPLAY_MATCH",
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                input_file_hashes=pins,
                counts=counts,
                records=records,
                goal_complete=False,
                team_game_verified=False,
                neural_actor_optimizer_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="PASS_TRANSFER_AUDIT_FAILED_PRESERVE_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                records=records,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
