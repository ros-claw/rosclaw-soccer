"""Reusable SIM transfer collector; never used as the audit integrator.

The caller owns immutable source admission, whole-bank preflight and budget.
Normal terminal trials persist all charged native/NN rows before scoring.
No simulator or model is allocated by importing this module.
"""

import json
from types import SimpleNamespace

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
from rosclaw_soccer.rsi.paid_run_pilot_recovery import stand_from_raw
from rosclaw_soccer.rsi.pass_transfer_outcome import pass_transfer_outcome
from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_stroke_search_common import stroke_action


def collect_trial(setup, actor, course, *, output, index, counts):
    def store(folder, name, rows):
        path = folder / (name + ".npz")
        np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
        return dict(path=str(path.relative_to(output)), hash=digest(path))

    ep, w, forward, home, _ = setup.prepare(
        course["context"],
        course["side"],
        ball_planar_offset_world=None
        if "ball_planar_offset_world" not in course
        else np.array(course["ball_planar_offset_world"], dtype=np.float64),
    )
    initial_q, initial_v = w.data.qpos.copy(), w.data.qvel.copy()
    folder = output / f"trial{index}"
    folder.mkdir()
    init_path = folder / "initial.npz"
    np.savez_compressed(init_path, qpos=initial_q, qvel=initial_v)
    if index % 2:
        with np.load(output / f"trial{index - 1}" / "initial.npz", allow_pickle=False) as z:
            if not np.array_equal(initial_q, z["qpos"]) or not np.array_equal(initial_v, z["qvel"]):
                raise ValueError("paired teacher/control arms must start in identical physics")
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
        expected_model_hash=setup.pins[str(setup.velocity_path)],
        expected_config_hash=setup.pins[str(setup.velocity_config)],
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
        "initial": dict(path=str(init_path.relative_to(output)), hash=digest(init_path)),
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
    write_once(folder / "complete.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)
    return result
