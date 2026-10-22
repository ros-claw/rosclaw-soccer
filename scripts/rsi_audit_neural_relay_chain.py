"""Own-history whole32 relay replay with an independent native PD clock.

Reuses observation/skill wrappers and geometric selector on OWN physics.
Does not execute saved actions, borrow RNN history or call producer advance.
Negative outcomes are mandatory too. Audit equality is NOT football success.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-head", "source-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    rp, dp = a.source_root / "complete.json", a.source_root / "declaration.json"
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (rp, dp))
        or (a.source_root / "failed.json").exists()
        or digest(rp) != a.source_hash
    ):
        raise ValueError("unused own replay and exact completed source required")
    s, dec = [json.loads(x.read_text()) for x in (rp, dp)]
    if (
        s["status"] != "NEURAL_RELAY32_COLLECTED_PENDING_OWN_REPLAY"
        or s["source_head"] != a.source_head
        or s["declaration_hash"] != digest(dp)
        or dec["protocol"] != "soccer.neural_relay_chain32.v1"
        or len(s["records"]) != 32
        or len(dec["courses"]) != 32
        or dec["optimizer_updates"] != 0
        or any(
            s[k] is not False
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("all32 exact unpromoted neural relay trials required")
    pins = dict(s["input_file_hashes"])
    pins[str(rp)], pins[str(dp)] = a.source_hash, digest(dp)
    for i, claim in enumerate(s["records"]):
        if claim["index"] != i or claim["course"] != dec["courses"][i]:
            raise ValueError("full preregistered order required")
        artifacts = list(claim["artifacts"].values())
        if claim["handoff"] is not None:
            artifacts.append(claim["handoff"]["boundary"])
        for artifact in artifacts:
            if artifact is None:
                continue
            path = a.source_root / artifact["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (a.source_root / f"trial{i}").resolve()
                or digest(path) != artifact["hash"]
            ):
                raise ValueError("all exact confined relay raw artifacts required")
            pins[str(path)] = artifact["hash"]
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
            shared_neural_codecs_and_role_selector=True,
            source_actions_executed=False,
            borrowed_memory=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native_steps=0, actual_actor_calls=0, shadow_actor_calls=0)
    records = []
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
        from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
        from rosclaw_soccer.rsi.receiving_handoff_opportunity import body_contact_kind
        from rosclaw_soccer.rsi.relay_audit_clock import RelayAuditClock
        from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters
        from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
        from rosclaw_soccer.rsi.shared_relay_setup import (
            SharedRelaySetup,
            motion_segment,
            receiving_target_for_player,
        )
        from rosclaw_soccer.rsi.shared_shadow_motor import SharedShadowMotor
        from scripts.rsi_stroke_search_common import open_motor

        base = dec["baseline_training_hash"]
        audits = [
            json.loads(Path(path).read_text())
            for path in pins
            if path.endswith("/native-audit/complete.json")
        ]
        audits = [row for row in audits if row.get("training_hash") == base]
        if len(audits) != 1:
            raise ValueError("unique consumed native baseline context witness required")
        native_path, native = receipt(pins, audits[0]["source_hash"])
        context_path = native_path.parent / "declaration.json"
        if digest(context_path) != native["declaration_hash"]:
            raise ValueError("original context declaration mismatch")
        contexts = json.loads(context_path.read_text())["contexts"]
        bench, baseline = open_motor(pins, base, {i: contexts[i] for i in (1, 4, 20, 21)})
        candidate = dec["candidate"]
        _, training = receipt(pins, candidate["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            baseline.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(candidate["training_root"]),
            expected_receipt_hash=candidate["training_hash"],
            checkpoint_kind=candidate["kind"],
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
        if (a.source_root / "scene.xml").read_text() != setup.xml:
            raise ValueError("scene reconstruction mismatch")

        def load(claim, name):
            artifact = claim["artifacts"][name]
            if artifact is None:
                return {}
            with np.load(a.source_root / artifact["path"], allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def equal_trace(actual, saved, index):
            if set(actual) != set(saved):
                raise ValueError("neural trace keys differ")
            for k, value in actual.items():
                if not np.array_equal(value, saved[k][index]):
                    raise ValueError(f"own neural trace differs {k}:{index}")

        for claim in s["records"]:
            ci, course = claim["index"], claim["course"]
            ep, initialized, forward, home, anchor = setup.prepare(
                course["context"], course["lateral_m"]
            )
            w = RelayAuditClock(initialized)
            raw = load(claim, "physics")
            initial = load(claim, "initial")
            if not np.array_equal(initial["qpos"], w.data.qpos) or not np.array_equal(
                initial["qvel"], w.data.qvel
            ):
                raise ValueError("own original physical initialization differs")
            nn = {
                name: load(claim, name)
                for name in (
                    "passer_nn",
                    "receiver_velocity",
                    "receiver_shadow",
                    "receiver_return_nn",
                )
            }
            segment = motion_segment(ep.clip, course["arm"]["start"])
            kwargs = dict(
                torso_index=ep.torso_index,
                build_observation=ep.build_observation,
                offset=ep.offset,
                scale=ep.scale,
                order=ep.order,
            )
            passer = SharedNativeNeuralPlayer(w, "a/", actor, clip=ep.clip, goal=ep.goal, **kwargs)
            shadow = SharedShadowMotor(
                w,
                "b/",
                actor,
                clip=segment,
                goal=receiving_target_for_player(w, "a/"),
                **kwargs,
            )
            velocity = setup.velocity()
            readiness = OnlineReceivingBoundary(w, "b/")
            receiver = None
            ready = launch = False
            handoff_control = None
            goal = origin = None
            excluded = set()
            last_target = None
            largest_jump = 0.0
            physical = []
            vi = bi = 0
            for step in range(len(raw["qpos"])):
                if receiver is None and ready:
                    boundary = w.switch_pd_profile(
                        "b/", NativePDParameters(ep.kp, ep.kd, ep.effort)
                    )
                    h = claim["handoff"]
                    if h is None or h["control"] != step:
                        raise ValueError("own past-only readiness boundary differs")
                    with np.load(a.source_root / h["boundary"]["path"], allow_pickle=False) as z:
                        for side in ("before", "after"):
                            for k, value in boundary[side].items():
                                if not np.array_equal(value, z[side + "_" + k]):
                                    raise ValueError("own unchanged physical handoff differs")
                        for key, value in dict(
                            shadow_h=shadow.state[0].numpy(),
                            shadow_c=shadow.state[1].numpy(),
                            previous_executed_action=shadow.previous_action,
                        ).items():
                            if not np.array_equal(value, z[key]):
                                raise ValueError("own shadow carry history differs")
                    goal = receiving_target_for_player(w, "a/")
                    layout = w.layouts["b/"]
                    origin = w.data.qpos[layout.bq : layout.bq + 3].copy()
                    if goal.tolist() != h["return_goal"] or origin.tolist() != h["kick_origin"]:
                        raise ValueError("own current return task differs")
                    receiver = shadow.handoff(
                        goal=goal, carry_memory=course["arm"]["memory"] == "own_shadow_carry"
                    )
                    handoff_control = step
                if receiver is None and step >= 300:
                    raise ValueError("source exceeded original velocity H300")
                mujoco.mj_forward(w.model, w.data)
                if passer.phase == "MOTION_COMPLETE":
                    passer.begin_standing()
                target_a, trace = passer.propose()
                trace["control"] = np.array(step)
                equal_trace(trace, nn["passer_nn"], step)
                if receiver is None:
                    trace_s = shadow.observe()
                    trace_s["control"] = np.array(step)
                    equal_trace(trace_s, nn["receiver_shadow"], vi)
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
                    equal_trace(
                        dict(
                            control=np.array(step),
                            obs=velocity.last_observation.copy(),
                            action=velocity.last_action.copy(),
                            command=command,
                            phase=np.array(velocity.phase),
                        ),
                        nn["receiver_velocity"],
                        vi,
                    )
                    counts["shadow_actor_calls"] += 1
                    vi += 1
                else:
                    if receiver.phase == "MOTION_COMPLETE":
                        receiver.begin_standing()
                    target_b, trace_b = receiver.propose()
                    trace_b["control"] = np.array(step)
                    equal_trace(trace_b, nn["receiver_return_nn"], bi)
                    bi += 1
                row = w.advance({"a/": target_a, "b/": target_b})
                counts["native_steps"] += 10
                counts["actual_actor_calls"] += 2
                for key in ("qpos", "qvel", "torque", "ball_forces"):
                    if not np.array_equal(row[key], raw[key][step]):
                        raise ValueError(f"own native trajectory differs {ci}:{step}:{key}")
                if (
                    not np.array_equal(target_b, raw["target_b"][step])
                    or row["time_before"] != raw["time_before"][step]
                    or not np.array_equal(
                        list(row["body_failed"].values()), raw["body_failed"][step]
                    )
                ):
                    raise ValueError("own control, time or health differs")
                passer.commit()
                if receiver is None:
                    shadow.commit_actual()
                    ready = readiness.update()
                else:
                    receiver.commit()
                if last_target is not None:
                    largest_jump = max(largest_jump, float(np.abs(target_b - last_target).max()))
                last_target = target_b.copy()
                for prefix, contacts in w.last_ball_contact_bodies.items():
                    excluded.update(
                        name
                        for name in contacts
                        if body_contact_kind(name, prefix=prefix) == "excluded_or_unknown"
                    )
                physical.append(row)
                if w.failed and step != len(raw["qpos"]) - 1:
                    raise ValueError("source continued after body failure")
            termination = (
                "BODY_FAILURE"
                if w.failed
                else "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300"
                if receiver is None
                else "WORLD_HORIZON_COMPLETE"
            )
            if (
                termination != claim["termination"]
                or w.control_steps != claim["controls"]
                or w.native_steps != claim["native_steps"]
                or (not w.failed and w.control_steps != (300 if receiver is None else 600))
            ):
                raise ValueError("own complete outcome horizon differs")
            if (
                (receiver is None and ready and not w.failed)
                or len(nn["passer_nn"]["control"]) != w.control_steps
                or len(nn["receiver_velocity"].get("control", [])) != vi
                or len(nn["receiver_shadow"].get("control", [])) != vi
                or len(nn["receiver_return_nn"].get("control", [])) != bi
            ):
                raise ValueError("complete neural horizons or final online transition differ")
            result = dict(
                handoff_executed=handoff_control is not None,
                outgoing_foot_strike=False,
                minimum_return_goal_error_3d_m=None,
                return_0p1m_and_healthy=False,
                maximum_B_target_change_rad=largest_jump,
                excluded_contact_bodies=sorted(excluded),
            )
            if handoff_control is not None:
                post = physical[handoff_control:]
                if post:
                    layout = w.layouts["b/"]
                    q = np.stack([r["qpos"] for r in post]).reshape(-1, w.model.nq)
                    v = np.stack([r["qvel"] for r in post]).reshape(-1, w.model.nv)
                    f = np.stack([r["ball_forces"] for r in post]).reshape(-1, 2, 3)
                    direction = goal[:2] - origin[:2]
                    direction /= np.linalg.norm(direction)
                    strike = (f[:, 1, :2].max(axis=1) > 10) & (
                        v[:, layout.bv : layout.bv + 2] @ direction >= 1
                    )
                    points = q[:, layout.bq : layout.bq + 3]
                    error = float(np.linalg.norm(points - goal, axis=1).min())
                    result["outgoing_foot_strike"] = bool(strike.any())
                    if strike.any():
                        error = float(
                            np.linalg.norm(
                                points[int(np.flatnonzero(strike)[0]) :] - goal, axis=1
                            ).min()
                        )
                        result["return_0p1m_and_healthy"] = bool(
                            error <= 0.1 and not w.failed and not excluded
                        )
                    result["minimum_return_goal_error_3d_m"] = error
            if result != claim["result"]:
                raise ValueError("own return outcome differs")
            records.append(dict(index=ci, course=course, termination=termination, result=result))
            write_once(a.output / f"trial{ci}.json", records[-1])
            print(json.dumps(dict(index=ci, counts=counts, result=result)), flush=True)
        if counts != s["counts"]:
            raise ValueError("own complete physical/neural counts differ")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="ALL32_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH",
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                input_file_hashes=pins,
                counts=counts,
                records=records,
                independent_native_integrator=True,
                shared_neural_codecs_and_role_selector=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="NEURAL_RELAY_OWN_REPLAY_FAILED_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                completed_records=records,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
