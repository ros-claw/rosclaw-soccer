"""Resume receiver actor on cloned shared snapshots; passer still held."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy, observation98, target29
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_contact_neural_lookahead import neural_lookahead
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "snapshots", "actor", "config", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    for key in (
        "source-hash",
        "audit-hash",
        "snapshot-hash",
        "actor-hash",
        "config-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    root, audit = a.source / "source", a.source / "audit/complete.json"
    snap_receipt = a.snapshots / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or digest(snap_receipt) != a.snapshot_hash
        or digest(a.actor) != a.actor_hash
        or digest(a.config) != a.config_hash
    ):
        raise ValueError("unused output and exact pinned inputs required")
    source, _, source_hash = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    snaps = json.loads(snap_receipt.read_text())
    if (
        source_hash != a.source_hash
        or snaps["source_hash"] != source_hash
        or snaps["audit_hash"] != a.audit_hash
        or snaps["promotion_authorized"] is not False
        or snaps["saved_torque_is_not_policy"] is not True
    ):
        raise ValueError("exact audited snapshot lineage required")
    if (
        source["input_file_hashes"].get(str(a.actor)) != a.actor_hash
        or source["input_file_hashes"].get(str(a.config)) != a.config_hash
    ):
        raise ValueError("exact source actor and configuration required")
    scene = root / "scene.xml"
    if digest(scene) != snaps["scene_hash"]:
        raise ValueError("exact shared native scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    actor = G1VelocityPolicy(
        a.actor, a.config, expected_model_hash=a.actor_hash, expected_config_hash=a.config_hash
    )
    pd = NativePDParameters(
        np.asarray(actor.params.stiffness),
        np.asarray(actor.params.damping),
        0.85 * np.asarray(G1_HARD_TORQUE_LIMITS),
    )
    la = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    lb = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    source_rows = {r["index"]: r for r in source["records"]}
    snapshot_rows = {r["index"]: r for r in snaps["rows"]}
    ids = (0, 4, 8, 12)
    corrections = np.zeros((9, 29))
    for i, joint in enumerate((0, 3, 6, 9)):
        corrections[1 + 2 * i, joint], corrections[2 + 2 * i, joint] = 0.08, -0.08
    results = []
    for index in ids:
        item, original = snapshot_rows[index], source_rows[index]
        snapshot = a.snapshots / item["snapshot"]
        velocity, physical = (
            bind(root, original, "receiver_velocity"),
            bind(root, original, "physics"),
        )
        if digest(snapshot) != item["snapshot_hash"] or velocity is None or physical is None:
            raise ValueError("exact previous receiver history and physical trace required")
        d = mujoco.MjData(model)
        with np.load(snapshot, allow_pickle=False) as s:
            mujoco.mj_setState(model, d, s["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, d)
        with (
            np.load(velocity, allow_pickle=False) as h,
            np.load(physical, allow_pickle=False) as trace,
        ):
            if not np.array_equal(d.qpos, trace["qpos"][139, -1]) or not np.array_equal(
                d.qvel, trace["qvel"][139, -1]
            ):
                raise ValueError("exact observed initial state required")
            last_action = h["action"][139].copy()
            phase = np.float32(h["phase"][139])
            # Original next command used only for a one-step identity audit;
            # actual branch uses the last already-observed command instead.
            q, v = lb.canonical(d)
            obs, next_phase = observation98(
                q,
                v,
                lb.angular_velocity_body(d),
                h["command"][140],
                last_action,
                phase,
                actor.params,
            )
            action = actor.session.run(["actions"], {"obs": obs[None]})[0][0]
            if (
                not np.array_equal(obs, h["obs"][140])
                or not np.array_equal(action, h["action"][140])
                or next_phase != h["phase"][140]
                or not np.array_equal(target29(action, actor.params), trace["target_b"][140])
            ):
                raise ValueError("receiver one-step neural resume identity failed")
            command = h["command"][139].copy()
        candidates = neural_lookahead(
            model,
            d,
            players={"a/": pd, "b/": pd},
            ball_body_name="soccer_ball",
            receiver="b/",
            passer_target=d.qpos[la.qi].copy(),
            policy=actor,
            last_action=last_action,
            phase=phase,
            command=command,
            corrections=corrections,
            horizon_controls=30,
        )
        results.append(
            dict(
                index=index,
                snapshot_hash=item["snapshot_hash"],
                identity_one_step=True,
                held_command=command.tolist(),
                candidates=candidates,
            )
        )
        if digest(snapshot) != item["snapshot_hash"]:
            raise ValueError("snapshot changed during branch")
    if (
        digest(snap_receipt) != a.snapshot_hash
        or digest(a.actor) != a.actor_hash
        or digest(a.config) != a.config_hash
        or digest(scene) != snaps["scene_hash"]
    ):
        raise ValueError("input changed during branches")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="SHARED_RECEIVER_NEURAL_FORECAST_NOT_POLICY",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            snapshot_hash=a.snapshot_hash,
            actor_hash=a.actor_hash,
            config_hash=a.config_hash,
            indices=list(ids),
            results=results,
            correction_vectors=corrections.tolist(),
            receiver_history="observed prior raw action and phase; branch-local reinference",
            receiver_command="last observed held; original next command for identity check only",
            passer_control="initial measured posture held, not resumed original policy",
            horizon_controls=30,
            no_future_controls_in_forecast=True,
            optimizer_updates=0,
            independent_fresh=False,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    print(
        json.dumps({"cases": len(results), "branches": sum(len(r["candidates"]) for r in results)})
    )


if __name__ == "__main__":
    main()
