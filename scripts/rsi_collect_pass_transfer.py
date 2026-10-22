"""Actual two-body transfer of audited learned pass teachers; SIM research.

All4 original TRAIN contexts,2 goals,2 B initial sides and BOTH unassisted
and selected-teacher arms are mandatory. Failed single-body goals remain.
One continuous shared world per trial; no ball reset or source action replay.
Formal pass/receive deadline remains300 controls.100 NEW neural A recovery
controls require a separately declared continuous B stream, not legacy H300.
"""

import argparse
import json
from pathlib import Path


def admit_teacher(root, expected, own_root, own_expected, pins):
    from scripts.rsi_native_context_recovery_common import digest

    paths = [root / "complete.json", own_root / "complete.json"]
    if (
        any(x.is_symlink() for x in paths)
        or any((r / "failed.json").exists() for r in (root, own_root))
        or [digest(x) for x in paths] != [expected, own_expected]
    ):
        raise ValueError("exact closed independently replayed response refinement required")
    s, own = [json.loads(x.read_text()) for x in paths]
    if (
        s["status"] != "LEARNED_PASS_RESPONSE_REFINEMENT_COLLECTED_PENDING_OWN_AND_TRANSFER"
        or own["status"] != "LEARNED_PASS_RESPONSE_ALL_MODELS_AND_WHOLE_NATIVE_REPLAY_MATCH"
        or own["producer_hash"] != expected
        or own["context"] != s["context"]
        or own["counts"]
        != dict(
            native=s["counts"]["native_steps"],
            neural=s["counts"]["actor_calls"],
            kinematic_intent=0,
        )
        or own["selections"] != s["selections"]
        or len(own["records"]) != len(s["records"])
        or len(s["selections"]) != 2
        or any(
            r[k] is not False
            for r in (s, own)
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("whole own replay, both goals and all negative outcomes required")
    for actual, proven in zip(s["records"], own["records"], strict=True):
        if any(
            actual[k] != proven[k] for k in ("tag", "task", "passing_result", "actual_response")
        ):
            raise ValueError("source teacher outcome is not own-reconstructed")
    for path, key in (
        (root / "declaration.json", "declaration_hash"),
        (root / "plan.json", "plan_hash"),
        (root / "preflight.json", "preflight_hash"),
    ):
        if path.is_symlink() or digest(path) != s[key]:
            raise ValueError("bound teacher metadata required")
        pins[str(path)] = s[key]
    for record in s["records"]:
        for art in record["artifacts"].values():
            path = root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (root / record["tag"]).resolve()
                or digest(path) != art["hash"]
            ):
                raise ValueError("all confined physical teacher trajectories required")
            pins[str(path)] = art["hash"]
    pins.update(own["input_file_hashes"])
    pins.update({str(p): h for p, h in zip(paths, (expected, own_expected), strict=True)})
    return s, json.loads((root / "preflight.json").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("frontier-root", "velocity-root", "velocity-audit-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "frontier-head",
        "frontier-hash",
        "velocity-hash",
        "velocity-audit-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--teacher",
        nargs=4,
        action="append",
        required=True,
        metavar=("ROOT", "HASH", "OWN_ROOT", "OWN_HASH"),
    )
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists() or len(a.teacher) != 4:
        raise ValueError("unused two-body transfer root and all four teacher contexts required")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if [c["seed"] for c in candidates] != [65023]:
        raise ValueError("same retained neural actor65023 required")
    teachers = [admit_teacher(Path(r), h, Path(o), oh, pins) for r, h, o, oh in a.teacher]
    if [s["context"] for s, _ in teachers] != [1, 4, 20, 21] or any(
        [r["task"] for r in s["selections"]] != [0, 1] for s, _ in teachers
    ):
        raise ValueError("all original contexts in declared order required")
    velocity_reports = []
    for root, h in (
        (a.velocity_root, a.velocity_hash),
        (a.velocity_audit_root, a.velocity_audit_hash),
    ):
        path = root / "complete.json"
        if path.is_symlink() or (root / "failed.json").exists() or digest(path) != h:
            raise ValueError("closed long-horizon velocity qualification required")
        report = json.loads(path.read_text())
        velocity_reports.append(report)
        pins.update(report["input_file_hashes"])
        pins[str(path)] = h
    vs, vo = velocity_reports
    if (
        vs["status"] != "CONTINUOUS_VELOCITY60S_COLLECTED_NOT_TEAM_GAME"
        or vo["status"] != "CONTINUOUS_VELOCITY60S_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or vo["source_hash"] != a.velocity_hash
        or vo["counts"] != vs["counts"]
        or any([r["context"] for r in v["records"]] != [1, 4, 20, 21] for v in velocity_reports)
        or any(not r["healthy_full60s"] for v in velocity_reports for r in v["records"])
    ):
        raise ValueError(
            "whole4 continuous native/own qualification, not legacy H300 bypass, required"
        )
    source_pins(pins)
    check_pins(pins)
    courses = []
    for s, pf in teachers:
        for selection in s["selections"]:
            task = selection["task"]
            for side in (-0.55, 0.55):
                for assisted in (False, True):
                    courses.append(
                        dict(
                            context=s["context"],
                            task=task,
                            side=side,
                            assisted=assisted,
                            target=pf["tasks"][task]["goal"],
                            center=pf["center"],
                            foot=pf["stroke_foot"],
                            latent=selection["latent"],
                            source_teacher_qualified=selection["passing_result"][
                                "precision_0p1m_moderate_speed_success"
                            ],
                        )
                    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.audited_pass_teacher_two_body_transfer32.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            courses=courses,
            candidate=candidates[0],
            baseline_training_hash=base_hash,
            contexts={str(i): contexts[i] for i in (1, 4, 20, 21)},
            maximum_worlds=32,
            max_controls_per_world=400,
            maximum_native_steps=128000,
            maximum_actor_calls=25600,
            formal_pass_receive_controls=300,
            required_A_neural_recovery_controls=100,
            source_failed_goals_retained=True,
            no_B_kick=True,
            no_ball_reposition=True,
            actual_applied_action_feedback=True,
            role_selector_is_geometry_teacher=True,
            teacher_rows_are_PPO_samples=False,
            neural_actor_optimizer_updates=0,
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
        from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
        from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
        from rosclaw_soccer.sim.contracts import hash_json
        from rosclaw_soccer.training.native_pass_courses import NativePassCourse
        from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
        from scripts.rsi_stroke_search_common import open_motor, stroke_action

        bench, base = open_motor(pins, base_hash, {i: contexts[i] for i in (1, 4, 20, 21)})
        c = candidates[0]
        _, tr = receipt(pins, c["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=tr["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
        typed = []
        for i in (1, 4, 20, 21):
            for side in (-0.55, 0.55):
                ep, w, _, _, _ = setup.prepare(i, side)
                if w.native_steps or not 1 <= ep.clip.length <= 300:
                    raise ValueError("all physical initial states admitted before stepping")
                typed.append(
                    NativePassCourse(
                        i,
                        4.0,
                        side,
                        hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
                    )
                )
        binding = preflight_receiving_courses(tuple(typed))
        write_once(
            a.output / "preflight.json",
            dict(
                initial_conditions=8,
                worlds=32,
                course_binding=binding,
                native_steps=0,
                ball_dimensions=setup.ball_dimensions,
            ),
        )
        (a.output / "scene.xml").write_text(setup.xml)

        def store(folder, name, rows):
            path = folder / (name + ".npz")
            np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
            return dict(path=str(path.relative_to(a.output)), hash=digest(path))

        for index, course in enumerate(courses):
            ep, w, forward, home, _ = setup.prepare(course["context"], course["side"])
            initial_q, initial_v = w.data.qpos.copy(), w.data.qvel.copy()
            folder = a.output / f"trial{index}"
            folder.mkdir()
            init_path = folder / "initial.npz"
            np.savez_compressed(init_path, qpos=initial_q, qvel=initial_v)
            if index % 2:
                with np.load(
                    a.output / f"trial{index - 1}" / "initial.npz", allow_pickle=False
                ) as z:
                    if not np.array_equal(initial_q, z["qpos"]) or not np.array_equal(
                        initial_v, z["qvel"]
                    ):
                        raise ValueError(
                            "paired teacher/control arms must start in identical physics"
                        )
            goal = np.array(course["target"])
            lateral = np.array([-forward[1], forward[0]])
            anchor = goal[:2] + 0.25 * forward + 0.12 * lateral
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
            physical, nn_a, nn_b = [], [], []
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
                tb = dict(
                    obs=velocity.last_observation.copy(),
                    action=velocity.last_action.copy(),
                    command=command,
                    phase=np.array(velocity.phase),
                    control=np.array(step),
                )
                row = w.advance({"a/": target_a, "b/": target_b})
                counts["native_steps"] += 10
                counts["actor_calls"] += 2
                passer.commit(executed_action=action)
                ready.update()
                stand_controls += int(standing)
                ta.update(
                    control=np.array(step),
                    executed_action=action.copy(),
                    teacher_delta_HG=delta,
                    teacher_authority_clipped=np.array(clipped),
                    standing=np.array(standing),
                )
                nn_a.append(ta)
                nn_b.append(tb)
                physical.append(
                    {k: row[k] for k in ("qpos", "qvel", "torque", "ball_forces")}
                    | dict(
                        body_failed=np.array([row["body_failed"][p] for p in ("a/", "b/")]),
                        target_a=target_a.copy(),
                        target_b=target_b.copy(),
                        time_before=np.array(row["time_before"]),
                    )
                )
                if w.failed:
                    break
            raw = {k: np.stack([r[k] for r in physical]) for k in physical[0]}
            # Persist every charged native and inference row BEFORE scoring;
            # a scoring defect must not destroy an actual physical execution.
            artifacts = {
                "initial": dict(path=str(init_path.relative_to(a.output)), hash=digest(init_path)),
                "physical": store(folder, "physical", physical),
                "A": store(folder, "A", nn_a),
                "B": store(folder, "B", nn_b),
            }
            al = w.layouts["a/"]
            qids = [*range(al.rq, al.rq + 7), *al.qi, *range(al.bq, al.bq + 7)]
            vids = [*range(al.rv, al.rv + 6), *al.vi, *range(al.bv, al.bv + 6)]
            recovery = None
            if stand_controls:
                recovery = stand_from_raw(
                    dict(
                        qpos=raw["qpos"][-stand_controls:, :, qids],
                        qvel=raw["qvel"][-stand_controls:, :, vids],
                        body_failed=raw["body_failed"][-stand_controls:, 0],
                    )
                )
            measure = pass_transfer_outcome(
                raw["qpos"][:, :, al.bq : al.bq + 3].reshape(-1, 3),
                raw["qvel"][:, :, al.bv : al.bv + 3].reshape(-1, 3),
                raw["ball_forces"].reshape(-1, 2, 3),
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
                artifacts=artifacts,
                neural_actor_optimizer_updates=0,
            )
            records.append(result)
            write_once(folder / "complete.json", result)
            print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)
        if len(records) != 32 or counts["native_steps"] > 128000 or counts["actor_calls"] > 25600:
            raise ValueError("whole finite32-world paired bank required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PASS_TEACHER_TWO_BODY32_COLLECTED_PENDING_OWN_REPLAY",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                preflight_hash=digest(a.output / "preflight.json"),
                scene_hash=digest(a.output / "scene.xml"),
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
                status="PASS_TRANSFER_FAILED_PRESERVE_NO_RETRY",
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
