"""Verify live shared SIM controls through early receiver contact."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import observation98, target29
from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
from rosclaw_soccer.rsi.receiver_contextual_selector import selector_class
from rosclaw_soccer.rsi.receiver_counterfactual_residual import receiver_counterfactual_residual
from rosclaw_soccer.rsi.receiver_transition_blend import receiver_transition_action
from rosclaw_soccer.rsi.relay_restored_receiving_role import restored_receiving_role
from rosclaw_soccer.rsi.restored_motion_reference import restored_motion_reference
from rosclaw_soccer.rsi.restored_scratch_world import restored_scratch_world
from rosclaw_soccer.rsi.restored_shadow_motor import restored_shadow_motor
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters
from rosclaw_soccer.rsi.shared_native_neural_player import PlayerObservationView
from rosclaw_soccer.rsi.shared_relay_setup import (
    SharedRelaySetup,
    motion_segment,
    receiving_target_for_player,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_context_recovery_common import check_pins
from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_stroke_search_common import open_motor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "snapshots", "frontier", "role-proof", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "audit-hash",
        "snapshot-hash",
        "frontier-hash",
        "frontier-head",
        "role-proof-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--probe-leg-rad", type=float, nargs=12)
    p.add_argument("--selector-receipt", type=Path)
    p.add_argument("--selector-receipt-hash")
    p.add_argument("--nondefault", action="store_true")
    p.add_argument("--dynamic-boundary", action="store_true")
    p.add_argument("--with-shadow", action="store_true")
    p.add_argument("--return-handoff", action="store_true")
    p.add_argument("--smooth-transition", action="store_true")
    p.add_argument("--smooth-transition-duration", type=int, default=20)
    p.add_argument("--expanded-velocity-proof", type=Path)
    p.add_argument("--expanded-velocity-proof-hash")
    a = p.parse_args()
    probe = None if a.probe_leg_rad is None else np.asarray(a.probe_leg_rad, dtype=np.float64)
    if probe is not None and (not np.isfinite(probe).all() or np.abs(probe).max() > 0.15):
        raise ValueError("finite bounded 12-leg SIM probe required")
    clean(a.expected_head)
    if (
        a.nondefault != (a.expanded_velocity_proof is not None)
        or a.nondefault != (a.expanded_velocity_proof_hash is not None)
        or (a.nondefault and (probe is not None or a.selector_receipt is not None))
    ):
        raise ValueError("expanded zero-only proof requires exact expanded receiver inference")
    if a.dynamic_boundary and not a.nondefault:
        raise ValueError("dynamic boundary qualification requires expanded source bank")
    if a.with_shadow and not a.nondefault:
        raise ValueError("shadow qualification requires expanded source bank")
    if a.return_handoff and not (a.dynamic_boundary and a.with_shadow):
        raise ValueError("return replay requires actual dynamic readiness and own shadow history")
    if a.smooth_transition and not a.return_handoff:
        raise ValueError("smooth candidate requires the full own return handoff")
    if not 1 <= a.smooth_transition_duration <= 100 or (
        not a.smooth_transition and a.smooth_transition_duration != 20
    ):
        raise ValueError("explicit bounded transition duration requires smooth candidate")
    if a.nondefault:
        if (
            a.expanded_velocity_proof.is_symlink()
            or digest(a.expanded_velocity_proof) != a.expanded_velocity_proof_hash
        ):
            raise ValueError("exact expanded velocity proof required")
        expanded_proof = json.loads(a.expanded_velocity_proof.read_text())
        if (
            expanded_proof.get("status")
            != "RECEIVER_NONDEFAULT_SNAPSHOT_OWN_VELOCITY_INFERENCE_MATCH_NOT_POLICY_RESUME"
            or expanded_proof.get("source_hash") != a.source_hash
            or expanded_proof.get("audit_hash") != a.audit_hash
            or expanded_proof.get("snapshot_hash") != a.snapshot_hash
            or len(expanded_proof.get("rows", [])) != 24
            or any(
                not all(
                    r.get(k) is True
                    for k in ("observation_exact", "action_exact", "phase_exact", "target_exact")
                )
                for r in expanded_proof["rows"]
            )
            or any(
                expanded_proof.get(k) is not False
                for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
            )
        ):
            raise ValueError("all exact unpromoted expanded receiver inferences required")
    selector_weights = None
    if (a.selector_receipt is None) != (a.selector_receipt_hash is None) or (
        a.selector_receipt is not None and probe is not None
    ):
        raise ValueError("one explicitly bound selector or one fixed probe required")
    if a.selector_receipt is not None:
        if a.selector_receipt.is_symlink() or digest(a.selector_receipt) != a.selector_receipt_hash:
            raise ValueError("exact selector training receipt required")
        selector_receipt = json.loads(a.selector_receipt.read_text())
        weights_path = a.selector_receipt.parent / "weights.npz"
        if (
            selector_receipt.get("status")
            != "RECEIVER_CONTEXT_SELECTOR_TRAINED_DEV_ONLY_NOT_PHYSICALLY_VALIDATED"
            or any(
                selector_receipt.get(k) is not False
                for k in ("goal_complete", "hardware_authorized", "promotion_authorized")
            )
            or weights_path.is_symlink()
            or digest(weights_path) != selector_receipt.get("weights_hash")
        ):
            raise ValueError("unpromoted exact selector weight artifact required")
        with np.load(weights_path, allow_pickle=False) as archive:
            selector_weights = {key: archive[key].copy() for key in archive.files}
        selector_class(np.zeros(98, dtype=np.float32), selector_weights)
    root, audit, snapshot_receipt = (
        a.source / "source",
        a.source / "audit/complete.json",
        a.snapshots / "complete.json",
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or snapshot_receipt.is_symlink()
        or digest(snapshot_receipt) != a.snapshot_hash
        or a.role_proof.is_symlink()
        or digest(a.role_proof) != a.role_proof_hash
    ):
        raise ValueError("exact completed snapshots and unused external output required")
    source, _, sh = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    snaps = json.loads(snapshot_receipt.read_text())
    if (
        sh != a.source_hash
        or snaps.get("status")
        != (
            "RELAY_NONDEFAULT_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
            if a.nondefault
            else "RELAY_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
        )
        or snaps.get("source_hash") != sh
        or snaps.get("audit_hash") != a.audit_hash
        or len(snaps.get("rows", [])) != (24 if a.nondefault else 8)
        or snaps.get("saved_torque_is_not_policy") is not True
        or any(snaps.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
    ):
        raise ValueError("all exact unpromoted state snapshots required")
    if a.nondefault:
        snapshot_ids = {
            (r["index"], r["snapshot_hash"], r["receiver_lateral_offset_m"]) for r in snaps["rows"]
        }
        proof_ids = {
            (r["index"], r["snapshot_hash"], r["receiver_lateral_offset_m"])
            for r in expanded_proof["rows"]
        }
        if len(snapshot_ids) != 24 or snapshot_ids != proof_ids:
            raise ValueError("expanded receiver proof must cover exactly these snapshots")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier, a.frontier_head, a.frontier_hash
    )
    if [c["seed"] for c in candidates] != [65023] or any(
        source["input_file_hashes"].get(k) != v for k, v in pins.items()
    ):
        raise ValueError("exact original frozen passer model and lineage required")
    check_pins(pins)
    indices = (2, 3, 5, 7)
    bench, base = open_motor(pins, base_hash, {i: contexts[i] for i in indices})
    candidate = candidates[0]
    _, training = receipt(pins, candidate["training_hash"])
    actor, _ = load_dual_candidate_for_sim_audit(
        base.parent,
        parent_lineage_hash=training["parent_lineage_hash"],
        root=Path(candidate["training_root"]),
        expected_receipt_hash=candidate["training_hash"],
        checkpoint_kind=candidate["kind"],
    )
    setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor, context_indices=indices)
    velocity_policy = setup.velocity()
    scene = root / "scene.xml"
    if digest(scene) != snaps["scene_hash"] or scene.read_text() != setup.xml:
        raise ValueError("unchanged original shared model required")
    layout = NativeVelocityLayout(
        setup.model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    receiver_layout = NativeVelocityLayout(
        setup.model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    by_index = {r["index"]: r for r in source["records"]}
    rows = []
    probe_trajectories = []
    end_control = 300 if a.dynamic_boundary else 160 if a.nondefault else 178
    for item in snaps["rows"]:
        row = by_index[item["index"]]
        role_offset = row["course"]["arm"]["receiver_lateral_offset_m"]
        if (
            role_offset not in ((-0.35, -0.12, 0.35) if a.nondefault else (0.12,))
            or (a.nondefault and item.get("receiver_lateral_offset_m") != role_offset)
            or row["course"]["context"] != item["context"]
            or row["course"]["lateral_m"] != item["side"]
            or item["replayed_steps"] != 1400
            or item["restored_validation_steps"] != 10
        ):
            raise ValueError("exact observed original context and boundary required")
        snap = a.snapshots / item["snapshot"]
        trace, physical = bind(root, row, "passer_nn"), bind(root, row, "physics")
        initial, velocity = bind(root, row, "initial"), bind(root, row, "receiver_velocity")
        shadow_path = bind(root, row, "receiver_shadow") if a.with_shadow else None
        if (
            snap.is_symlink()
            or snap.resolve().parent != a.snapshots.resolve()
            or digest(snap) != item["snapshot_hash"]
            or trace is None
            or physical is None
            or initial is None
            or velocity is None
            or (a.with_shadow and shadow_path is None)
        ):
            raise ValueError("confined snapshot and exact private source histories required")
        data = mujoco.MjData(setup.model)
        with np.load(snap, allow_pickle=False) as archive:
            integration_state = archive["state"].copy()
            mujoco.mj_setState(
                setup.model, data, integration_state, mujoco.mjtState.mjSTATE_INTEGRATION
            )
        mujoco.mj_forward(setup.model, data)
        view = PlayerObservationView(SimpleNamespace(data=data, layouts={"a/": layout}), "a/")
        ep = setup.episodes[item["context"]]
        if not a.dynamic_boundary and (
            ep.clip.length < end_control
            or (row["handoff"] is not None and row["handoff"]["control"] < end_control)
        ):
            raise ValueError("contact proof must stop before actual motion or receiver handoff")
        with (
            np.load(trace, allow_pickle=False) as t,
            np.load(physical, allow_pickle=False) as physics,
            np.load(initial, allow_pickle=False) as initial_data,
            np.load(velocity, allow_pickle=False) as velocity_data,
            np.load(shadow_path, allow_pickle=False)
            if a.with_shadow
            else nullcontext(None) as shadow_data,
        ):
            if (
                len(t["NN_obs"]) <= 140
                or not np.array_equal(t["control"][:141], np.arange(141))
                or not np.array_equal(data.qpos, physics["qpos"][139, -1])
                or not np.array_equal(data.qvel, physics["qvel"][139, -1])
                or not np.array_equal(t["NN_h_after"][139], t["NN_h_before"][140])
                or not np.array_equal(t["NN_c_after"][139], t["NN_c_before"][140])
                or not np.array_equal(t["NN_obs"][140, 125:154], t["NN_action"][139])
                or not np.array_equal(
                    velocity_data["obs"][140, 69:98], velocity_data["action"][139]
                )
                or not np.array_equal(
                    target29(velocity_data["action"][139], velocity_policy.params),
                    physics["target_b"][139],
                )
            ):
                raise ValueError("actual own recurrent and physical boundary required")
            obs = ep.build_observation(
                view,
                None,
                0,
                t["NN_action"][139].copy(),
                ep.clip.joint_pos[139].copy(),
                ep.clip.joint_vel[139].copy(),
                ep.clip.body_ang_vel_w[139, ep.torso_index].copy(),
                view.ball_pos,
                ep.goal,
                ep.offset,
                obs_dim=None,
            )
            with torch.no_grad():
                out, (h, c) = actor(
                    torch.from_numpy(obs).reshape(1, 1, 160),
                    (
                        torch.from_numpy(t["NN_h_before"][140].copy()),
                        torch.from_numpy(t["NN_c_before"][140].copy()),
                    ),
                )
            if (
                not np.array_equal(obs, t["NN_obs"][140])
                or not np.array_equal(out[0, 0].numpy(), t["NN_action"][140])
                or not np.array_equal(h.numpy(), t["NN_h_after"][140])
                or not np.array_equal(c.numpy(), t["NN_c_after"][140])
            ):
                raise ValueError("restored passer own observation or recurrent inference differs")
            past_ball_vel = np.concatenate(
                (
                    initial_data["qvel"][None, layout.bv : layout.bv + 2],
                    physics["qvel"][:139, -1, layout.bv : layout.bv + 2],
                ),
                axis=0,
            )
            receiver_q, receiver_v = receiver_layout.canonical(data)
            role = restored_receiving_role(
                initial_ball=initial_data["qpos"][layout.bq : layout.bq + 2].copy(),
                initial_home=initial_data["qpos"][
                    receiver_layout.rq : receiver_layout.rq + 2
                ].copy(),
                pass_goal=ep.goal,
                current_q=receiver_q,
                current_v=receiver_v,
                past_ball_velocities=past_ball_vel,
                lateral_offset_m=role_offset,
            )
            if not np.array_equal(
                np.asarray(role.command_body, dtype=np.float32), velocity_data["command"][140]
            ):
                raise ValueError("receiver own causal role decision differs at restored boundary")
            receiver_obs, receiver_phase = observation98(
                receiver_q,
                receiver_v,
                receiver_layout.angular_velocity_body(data),
                np.asarray(role.command_body, dtype=np.float32),
                velocity_data["action"][139].copy(),
                velocity_data["phase"][139],
                velocity_policy.params,
            )
            receiver_action = velocity_policy.session.run(["actions"], {"obs": receiver_obs[None]})[
                0
            ][0]
            receiver_target = target29(receiver_action, velocity_policy.params)
            passer_target = ep.offset + out[0, 0].numpy()[ep.order] * ep.scale
            if (
                not np.array_equal(receiver_obs, velocity_data["obs"][140])
                or not np.array_equal(receiver_action, velocity_data["action"][140])
                or receiver_phase != velocity_data["phase"][140]
                or not np.array_equal(receiver_target, physics["target_b"][140])
            ):
                raise ValueError("receiver own inference differs at restored boundary")
            selection = None
            case_probe = probe
            if selector_weights is not None:
                selection = selector_class(receiver_obs, selector_weights)
                knee = (0.0, 0.1, -0.1)[selection]
                case_probe = np.asarray(
                    [0.0, 0.0, 0.0, knee, -0.5 * knee, 0.0] * 2, dtype=np.float64
                )
            validate_original = case_probe is None or not np.any(case_probe)
            vp = velocity_policy.params
            scratch = restored_scratch_world(
                setup.model,
                integration_state,
                players={
                    "a/": NativePDParameters(ep.kp, ep.kd, ep.effort),
                    "b/": NativePDParameters(
                        np.asarray(vp.stiffness), np.asarray(vp.damping), ep.effort
                    ),
                },
                ball_body_name="soccer_ball",
                control=140,
                prior_targets={
                    "a/": ep.offset + t["NN_action"][139][ep.order] * ep.scale,
                    "b/": physics["target_b"][139].copy(),
                },
            )
            shadow = None
            if a.with_shadow:
                if (
                    len(shadow_data["control"]) <= 140
                    or not np.array_equal(shadow_data["control"][:141], np.arange(141))
                    or not np.array_equal(shadow_data["h_before"][140], shadow_data["h_after"][139])
                    or not np.array_equal(shadow_data["c_before"][140], shadow_data["c_after"][139])
                ):
                    raise ValueError("complete own prior shadow recurrent history required")
                initial_world = SimpleNamespace(
                    data=SimpleNamespace(qpos=initial_data["qpos"], qvel=initial_data["qvel"]),
                    layouts={"a/": layout},
                )
                shadow = restored_shadow_motor(
                    scratch,
                    "b/",
                    actor,
                    clip=motion_segment(ep.clip, row["course"]["arm"]["start"]),
                    torso_index=ep.torso_index,
                    build_observation=ep.build_observation,
                    goal=receiving_target_for_player(initial_world, "a/"),
                    offset=ep.offset,
                    scale=ep.scale,
                    order=ep.order,
                    previous_action=shadow_data["obs"][140, 125:154].copy(),
                    hidden=shadow_data["h_before"][140].copy(),
                    cell=shadow_data["c_before"][140].copy(),
                    observed_controls=140,
                )
                shadow_trace = shadow.observe()
                if any(
                    not np.array_equal(shadow_trace[k], shadow_data[k][140]) for k in shadow_trace
                ):
                    raise ValueError("restored own shadow observation or inference differs at 140")
            readiness = OnlineReceivingBoundary(scratch, "b/") if a.dynamic_boundary else None
            resumed = scratch.advance({"a/": passer_target, "b/": receiver_target})
            if shadow is not None:
                shadow.commit_actual()
            for key in ("qpos", "qvel", "torque", "ball_forces"):
                if not np.array_equal(resumed[key], physics[key][140]):
                    raise ValueError("restored coupled native substeps differ: " + key)
            if (resumed["ball_forces"][:, 1, :] > 1.0).any():
                raise ValueError("first restored boundary must precede receiver contact")
            first_contact = None
            first_groups: list[str] = []
            ready = readiness.update() if readiness is not None else False
            physical_samples = [resumed]
            executed_actions = [receiver_action.copy()]
            passer_actions = [out[0, 0].numpy().copy()]
            receiver_observations = [receiver_obs.copy()]
            receiver_commands = [np.asarray(role.command_body, dtype=np.float32)]
            # Continue from each actor's *own newly generated* state. Recorded
            # future traces are audit targets only, never policy inputs.
            previous_action = out[0, 0].numpy().copy()
            previous_receiver_action = receiver_action.copy()
            previous_receiver_phase = receiver_phase
            past_ball_vel = np.concatenate((past_ball_vel, receiver_v[None, 35:37]), axis=0)
            for frame in range(141, end_control):
                if ready:
                    break
                mujoco.mj_forward(setup.model, scratch.data)
                view = PlayerObservationView(scratch, "a/")
                if shadow is not None:
                    shadow_trace = shadow.observe()
                    if any(
                        not np.array_equal(shadow_trace[k], shadow_data[k][frame])
                        for k in shadow_trace
                    ):
                        raise ValueError(f"own actual shadow recurrence differs at {frame}")
                if a.dynamic_boundary:
                    ref_pos, ref_vel, ref_ang = restored_motion_reference(
                        ep.clip, ep.torso_index, frame
                    )
                else:
                    ref_pos = ep.clip.joint_pos[frame - 1].copy()
                    ref_vel = ep.clip.joint_vel[frame - 1].copy()
                    ref_ang = ep.clip.body_ang_vel_w[frame - 1, ep.torso_index].copy()
                obs = ep.build_observation(
                    view,
                    None,
                    0,
                    previous_action,
                    ref_pos,
                    ref_vel,
                    ref_ang,
                    view.ball_pos,
                    ep.goal,
                    ep.offset,
                    obs_dim=None,
                )
                with torch.no_grad():
                    out, (h, c) = actor(torch.from_numpy(obs).reshape(1, 1, 160), (h, c))
                current_action = out[0, 0].numpy().copy()
                receiver_q, receiver_v = receiver_layout.canonical(scratch.data)
                role = restored_receiving_role(
                    initial_ball=initial_data["qpos"][layout.bq : layout.bq + 2].copy(),
                    initial_home=initial_data["qpos"][
                        receiver_layout.rq : receiver_layout.rq + 2
                    ].copy(),
                    pass_goal=ep.goal,
                    current_q=receiver_q,
                    current_v=receiver_v,
                    past_ball_velocities=past_ball_vel,
                    lateral_offset_m=role_offset,
                )
                receiver_obs, receiver_phase = observation98(
                    receiver_q,
                    receiver_v,
                    receiver_layout.angular_velocity_body(scratch.data),
                    np.asarray(role.command_body, dtype=np.float32),
                    previous_receiver_action,
                    previous_receiver_phase,
                    velocity_policy.params,
                )
                receiver_action = velocity_policy.session.run(
                    ["actions"], {"obs": receiver_obs[None]}
                )[0][0]
                if validate_original and (
                    not np.array_equal(obs, t["NN_obs"][frame])
                    or not np.array_equal(current_action, t["NN_action"][frame])
                    or not np.array_equal(h.numpy(), t["NN_h_after"][frame])
                    or not np.array_equal(c.numpy(), t["NN_c_after"][frame])
                    or not np.array_equal(receiver_obs, velocity_data["obs"][frame])
                    or not np.array_equal(receiver_action, velocity_data["action"][frame])
                    or receiver_phase != velocity_data["phase"][frame]
                ):
                    raise ValueError(f"restored own live policy history differs at {frame}")
                if case_probe is not None:
                    amplitude = float(np.sin(np.pi * (frame - 140) / 38.0) ** 2)
                    receiver_action, receiver_target = receiver_counterfactual_residual(
                        receiver_action, case_probe, velocity_policy.params, amplitude=amplitude
                    )
                else:
                    receiver_target = target29(receiver_action, velocity_policy.params)
                resumed = scratch.advance(
                    {
                        "a/": ep.offset + current_action[ep.order] * ep.scale,
                        "b/": receiver_target,
                    }
                )
                if shadow is not None:
                    shadow.commit_actual()
                physical_samples.append(resumed)
                executed_actions.append(receiver_action.copy())
                passer_actions.append(current_action.copy())
                receiver_observations.append(receiver_obs.copy())
                receiver_commands.append(np.asarray(role.command_body, dtype=np.float32))
                for key in ("qpos", "qvel", "torque", "ball_forces"):
                    if validate_original and not np.array_equal(resumed[key], physics[key][frame]):
                        raise ValueError(f"restored coupled native {key} differs at {frame}")
                witnesses = resumed["ball_forces"][:, 1, :]
                hit = np.flatnonzero((witnesses > 1.0).any(axis=1))
                if first_contact is None and len(hit):
                    substep = int(hit[0])
                    first_contact = (frame, substep)
                    first_groups = [
                        name
                        for name, force in zip(
                            ("left_foot", "right_foot", "nonfoot"),
                            witnesses[substep],
                            strict=True,
                        )
                        if force > 1.0
                    ]
                previous_action = current_action
                previous_receiver_action = receiver_action.copy()
                previous_receiver_phase = receiver_phase
                past_ball_vel = np.concatenate((past_ball_vel, receiver_v[None, 35:37]), axis=0)
                if readiness is not None:
                    ready = readiness.update()
                if scratch.failed:
                    if validate_original and not a.dynamic_boundary:
                        raise ValueError("baseline body failure before fixed comparison horizon")
                    break
            if a.dynamic_boundary:
                source_handoff = None if row["handoff"] is None else row["handoff"]["control"]
                if (
                    (ready and scratch.control_steps != source_handoff)
                    or (not ready and source_handoff is not None)
                    or (not ready and scratch.failed != (row["termination"] == "BODY_FAILURE"))
                    or (not ready and not scratch.failed and scratch.control_steps != 300)
                ):
                    raise ValueError("own causal readiness or source termination differs")
            return_controls = 0
            return_actions = []
            return_observations = []
            return_targets = []
            if a.return_handoff and ready:
                return_path = bind(root, row, "receiver_return_nn")
                if return_path is None or shadow is None:
                    raise ValueError("own recorded return actor and live shadow required")
                with np.load(return_path, allow_pickle=False) as return_data:
                    goal = receiving_target_for_player(scratch, "a/")
                    if not np.array_equal(goal, np.asarray(row["handoff"]["return_goal"])):
                        raise ValueError("actual own return goal must match original boundary")
                    scratch.switch_pd_profile("b/", NativePDParameters(ep.kp, ep.kd, ep.effort))
                    receiver = shadow.handoff(goal=goal, carry_memory=True)
                    start = scratch.control_steps
                    transition_start = receiver.last_action.copy()
                    for frame in range(start, 600):
                        mujoco.mj_forward(setup.model, scratch.data)
                        view = PlayerObservationView(scratch, "a/")
                        ref_pos, ref_vel, ref_ang = restored_motion_reference(
                            ep.clip, ep.torso_index, frame
                        )
                        obs = ep.build_observation(
                            view,
                            None,
                            0,
                            previous_action,
                            ref_pos,
                            ref_vel,
                            ref_ang,
                            view.ball_pos,
                            ep.goal,
                            ep.offset,
                            obs_dim=None,
                        )
                        with torch.no_grad():
                            out, (h, c) = actor(torch.from_numpy(obs).reshape(1, 1, 160), (h, c))
                        current_action = out[0, 0].numpy().copy()
                        if not a.smooth_transition and (
                            not np.array_equal(obs, t["NN_obs"][frame])
                            or not np.array_equal(current_action, t["NN_action"][frame])
                            or not np.array_equal(h.numpy(), t["NN_h_after"][frame])
                            or not np.array_equal(c.numpy(), t["NN_c_after"][frame])
                        ):
                            raise ValueError(
                                f"passer own return-phase recurrence differs at {frame}"
                            )
                        if receiver.phase == "MOTION_COMPLETE":
                            receiver.begin_standing()
                        relative = frame - start
                        smoothing = a.smooth_transition and relative < a.smooth_transition_duration
                        target_b, trace_b = receiver.propose(with_reference_evidence=smoothing)
                        executed_b = trace_b["NN_action"]
                        if smoothing:
                            executed_b = receiver_transition_action(
                                transition_start,
                                executed_b,
                                trace_b["NN_reference_nominal"],
                                ep.scale,
                                ep.order,
                                control_index=relative,
                                duration=a.smooth_transition_duration,
                            )
                            target_b = ep.offset + executed_b[ep.order] * ep.scale
                        if not a.smooth_transition and (
                            int(return_data["control"][relative]) != frame
                            or any(
                                not np.array_equal(trace_b[k], return_data[k][relative])
                                for k in trace_b
                            )
                        ):
                            raise ValueError(f"receiver own return policy differs at {frame}")
                        resumed = scratch.advance(
                            {"a/": ep.offset + current_action[ep.order] * ep.scale, "b/": target_b}
                        )
                        for key in ("qpos", "qvel", "torque", "ball_forces"):
                            if not a.smooth_transition and not np.array_equal(
                                resumed[key], physics[key][frame]
                            ):
                                raise ValueError(f"return-phase native {key} differs at {frame}")
                        receiver.commit(executed_action=executed_b if smoothing else None)
                        physical_samples.append(resumed)
                        return_actions.append(executed_b.copy())
                        return_observations.append(trace_b["NN_obs"].copy())
                        return_targets.append(target_b.copy())
                        return_controls += 1
                        previous_action = current_action
                        if scratch.failed:
                            break
                    if not a.smooth_transition and (
                        scratch.failed != (row["termination"] == "BODY_FAILURE")
                        or scratch.control_steps != len(physics["qpos"])
                    ):
                        raise ValueError("actual return-phase termination differs")
                if digest(return_path) != row["artifacts"]["receiver_return_nn"]["hash"]:
                    raise ValueError("original return actor trace changed")
        if (
            digest(snap) != item["snapshot_hash"]
            or digest(trace) != row["artifacts"]["passer_nn"]["hash"]
            or digest(physical) != row["artifacts"]["physics"]["hash"]
            or digest(initial) != row["artifacts"]["initial"]["hash"]
            or digest(velocity) != row["artifacts"]["receiver_velocity"]["hash"]
            or (
                a.with_shadow and digest(shadow_path) != row["artifacts"]["receiver_shadow"]["hash"]
            )
        ):
            raise ValueError("original physical or neural input changed")
        rows.append(
            dict(
                index=item["index"],
                context=item["context"],
                side=item["side"],
                receiver_lateral_offset_m=role_offset,
                observation_exact=validate_original,
                action_exact=validate_original,
                recurrent_after_exact=validate_original,
                receiver_role_command_exact=True,
                receiver_neural_inference_exact=validate_original,
                coupled_native_substeps_exact=validate_original and not a.smooth_transition,
                consecutive_controls=len(physical_samples),
                first_contact_frame_and_substep=first_contact,
                first_contact_groups=first_groups,
                body_failure=bool(scratch.failed),
                receiving_ready=bool(ready),
                last_control=scratch.control_steps,
                return_controls=return_controls,
                receiver_velocity_controls=len(executed_actions),
                selector_class=selection,
                own_shadow_verified=a.with_shadow,
            )
        )
        if probe is not None or selector_weights is not None or a.dynamic_boundary:
            probe_trajectories.append(
                {
                    **{
                        key: np.stack([sample[key] for sample in physical_samples])
                        for key in ("qpos", "qvel", "torque", "ball_forces")
                    },
                    "executed_receiver_action": np.stack(executed_actions),
                    "passer_action": np.stack(passer_actions),
                    "receiver_observation": np.stack(receiver_observations),
                    "receiver_command": np.stack(receiver_commands),
                    "receiver_return_action": np.asarray(return_actions, dtype=np.float32).reshape(
                        -1, 29
                    ),
                    "receiver_return_observation": np.asarray(
                        return_observations, dtype=np.float32
                    ).reshape(-1, 160),
                    "receiver_return_target": np.asarray(return_targets, dtype=np.float64).reshape(
                        -1, 29
                    ),
                }
            )
    if (
        len(rows) != (24 if a.nondefault else 8)
        or len({(r["context"], r["side"], r["receiver_lateral_offset_m"]) for r in rows})
        != (24 if a.nondefault else 8)
        or digest(snapshot_receipt) != a.snapshot_hash
        or digest(root / "complete.json") != sh
        or digest(audit) != a.audit_hash
        or digest(scene) != snaps["scene_hash"]
        or digest(a.role_proof) != a.role_proof_hash
        or (a.nondefault and digest(a.expanded_velocity_proof) != a.expanded_velocity_proof_hash)
        or (
            a.selector_receipt is not None
            and (
                digest(a.selector_receipt) != a.selector_receipt_hash
                or digest(weights_path) != selector_receipt["weights_hash"]
            )
        )
    ):
        raise ValueError("all exact unchanged original development cases required")
    clean(a.expected_head)
    check_pins(pins)
    a.output.mkdir(parents=True)
    if probe is not None or selector_weights is not None or a.dynamic_boundary:
        for row, trajectory in zip(rows, probe_trajectories, strict=True):
            path = a.output / f"trial-{row['index']}.npz"
            np.savez_compressed(path, **trajectory)
            row["physical_trajectory"] = path.name
            row["physical_trajectory_hash"] = digest(path)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "RELAY_NONDEFAULT_SMOOTH_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS"
                if a.smooth_transition
                else "RELAY_NONDEFAULT_CAUSAL_RETURN_CHAIN_MATCH_NOT_FOOTBALL_SUCCESS"
                if a.return_handoff
                else "RELAY_NONDEFAULT_CAUSAL_RECEIVING_BOUNDARY_MATCH_NOT_FOOTBALL_SUCCESS"
                if a.dynamic_boundary
                else "RELAY_NONDEFAULT_DUAL_NEURAL_20_CONTROLS_MATCH_NOT_FOOTBALL_SUCCESS"
                if a.nondefault
                else "RELAY_CONTEXT_SELECTOR_PHYSICAL_PROBE_NOT_FOOTBALL_SUCCESS"
                if selector_weights is not None
                else "RELAY_LEG_RESIDUAL_ZERO_OWN_REPLAY_NOT_FOOTBALL_SUCCESS"
                if probe is not None and not np.any(probe)
                else "RELAY_LEG_RESIDUAL_PHYSICAL_PROBE_NOT_FOOTBALL_SUCCESS"
                if probe is not None
                else "RELAY_RESTORED_OWN_NEURAL_THROUGH_CONTACT_MATCH_NOT_FOOTBALL_SUCCESS"
            ),
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            snapshot_hash=a.snapshot_hash,
            frontier_hash=a.frontier_hash,
            role_proof_hash=a.role_proof_hash,
            expanded_velocity_proof_hash=a.expanded_velocity_proof_hash,
            rows=rows,
            prior_own_action_and_recurrent_state_used=True,
            future_controls_used=False,
            receiver_role_selector_verified_at_boundary=True,
            causal_receiving_boundary_reconstructed=a.dynamic_boundary,
            own_return_handoff_reconstructed=a.return_handoff and not a.smooth_transition,
            smooth_transition=a.smooth_transition,
            smooth_transition_duration=a.smooth_transition_duration
            if a.smooth_transition
            else None,
            own_shadow_recurrence_verified=a.with_shadow,
            coupled_physics_continuation_verified=all(
                r["coupled_native_substeps_exact"] for r in rows
            ),
            native_steps=sum(row["consecutive_controls"] for row in rows) * 10,
            probe_leg_rad=None if probe is None else probe.tolist(),
            selector_receipt_hash=a.selector_receipt_hash,
            probe_action_is_actual_history=probe is not None or selector_weights is not None,
            optimizer_updates=0,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
