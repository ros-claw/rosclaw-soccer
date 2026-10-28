"""Recompute the passer's own recurrent inference at restored SIM boundaries."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.relay_restored_receiving_role import restored_receiving_role
from rosclaw_soccer.rsi.shared_native_neural_player import PlayerObservationView
from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_context_recovery_common import check_pins
from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_stroke_search_common import open_motor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "snapshots", "frontier", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "audit-hash",
        "snapshot-hash",
        "frontier-hash",
        "frontier-head",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
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
        or snaps.get("status") != "RELAY_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
        or snaps.get("source_hash") != sh
        or snaps.get("audit_hash") != a.audit_hash
        or len(snaps.get("rows", [])) != 8
        or snaps.get("saved_torque_is_not_policy") is not True
        or any(snaps.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
    ):
        raise ValueError("all eight exact unpromoted state snapshots required")
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
    for item in snaps["rows"]:
        row = by_index[item["index"]]
        if (
            row["course"]["arm"]["receiver_lateral_offset_m"] != 0.12
            or row["course"]["context"] != item["context"]
            or row["course"]["lateral_m"] != item["side"]
            or item["replayed_steps"] != 1400
            or item["restored_validation_steps"] != 10
        ):
            raise ValueError("exact observed original context and boundary required")
        snap = a.snapshots / item["snapshot"]
        trace, physical = bind(root, row, "passer_nn"), bind(root, row, "physics")
        initial, velocity = bind(root, row, "initial"), bind(root, row, "receiver_velocity")
        if (
            snap.is_symlink()
            or snap.resolve().parent != a.snapshots.resolve()
            or digest(snap) != item["snapshot_hash"]
            or trace is None
            or physical is None
            or initial is None
            or velocity is None
        ):
            raise ValueError("confined snapshot and exact private source histories required")
        data = mujoco.MjData(setup.model)
        with np.load(snap, allow_pickle=False) as archive:
            mujoco.mj_setState(
                setup.model, data, archive["state"], mujoco.mjtState.mjSTATE_INTEGRATION
            )
        mujoco.mj_forward(setup.model, data)
        view = PlayerObservationView(SimpleNamespace(data=data, layouts={"a/": layout}), "a/")
        ep = setup.episodes[item["context"]]
        with (
            np.load(trace, allow_pickle=False) as t,
            np.load(physical, allow_pickle=False) as physics,
            np.load(initial, allow_pickle=False) as initial_data,
            np.load(velocity, allow_pickle=False) as velocity_data,
        ):
            if (
                len(t["NN_obs"]) <= 140
                or not np.array_equal(t["control"][:141], np.arange(141))
                or not np.array_equal(data.qpos, physics["qpos"][139, -1])
                or not np.array_equal(data.qvel, physics["qvel"][139, -1])
                or not np.array_equal(t["NN_h_after"][139], t["NN_h_before"][140])
                or not np.array_equal(t["NN_c_after"][139], t["NN_c_before"][140])
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
            )
            if not np.array_equal(
                np.asarray(role.command_body, dtype=np.float32), velocity_data["command"][140]
            ):
                raise ValueError("receiver own causal role decision differs at restored boundary")
        if (
            digest(snap) != item["snapshot_hash"]
            or digest(trace) != row["artifacts"]["passer_nn"]["hash"]
            or digest(physical) != row["artifacts"]["physics"]["hash"]
            or digest(initial) != row["artifacts"]["initial"]["hash"]
            or digest(velocity) != row["artifacts"]["receiver_velocity"]["hash"]
        ):
            raise ValueError("original physical or neural input changed")
        rows.append(
            dict(
                index=item["index"],
                context=item["context"],
                side=item["side"],
                observation_exact=True,
                action_exact=True,
                recurrent_after_exact=True,
                receiver_role_command_exact=True,
            )
        )
    if (
        len(rows) != 8
        or len({(r["context"], r["side"]) for r in rows}) != 8
        or digest(snapshot_receipt) != a.snapshot_hash
        or digest(root / "complete.json") != sh
        or digest(audit) != a.audit_hash
        or digest(scene) != snaps["scene_hash"]
    ):
        raise ValueError("all exact unchanged original eight cases required")
    clean(a.expected_head)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PASSER_SNAPSHOT_OWN_RECURRENT_INFERENCE_MATCH_NOT_POLICY_RESUME",
            source_head=a.expected_head,
            source_hash=sh,
            audit_hash=a.audit_hash,
            snapshot_hash=a.snapshot_hash,
            frontier_hash=a.frontier_hash,
            rows=rows,
            prior_own_action_and_recurrent_state_used=True,
            future_controls_used=False,
            receiver_role_selector_verified_at_boundary=True,
            coupled_physics_continuation_verified=False,
            native_steps=0,
            optimizer_updates=0,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
