"""Export matched current-state SIM features from independently verified snapshots.

Original next-step states are used only to validate the rotating-frame velocity,
never as model input. No outcome labels, motor commands or policy are emitted.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiver_ball_observation import receiver_ball_observation
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "snapshots", "velocity-proof", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in (
        "source-hash",
        "audit-hash",
        "snapshot-hash",
        "velocity-proof-hash",
        "expected-head",
    ):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    snapshot_path = args.snapshots / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(
            p.is_symlink() or digest(p) != h
            for p, h in (
                (snapshot_path, args.snapshot_hash),
                (args.velocity_proof, args.velocity_proof_hash),
            )
        )
    ):
        raise ValueError("pinned snapshots, inference proof and new external output required")
    root, audit = args.source / "source", args.source / "audit/complete.json"
    source, _, source_hash = report(
        root,
        audit,
        args.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    snapshots, proof = (json.loads(p.read_text()) for p in (snapshot_path, args.velocity_proof))
    if (
        source_hash != args.source_hash
        or snapshots.get("source_hash") != source_hash
        or snapshots.get("audit_hash") != args.audit_hash
        or proof.get("snapshot_hash") != args.snapshot_hash
        or proof.get("source_hash") != source_hash
        or proof.get("audit_hash") != args.audit_hash
        or snapshots.get("status")
        != "RELAY_NONDEFAULT_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
        or proof.get("status")
        != "RECEIVER_NONDEFAULT_SNAPSHOT_OWN_VELOCITY_INFERENCE_MATCH_NOT_POLICY_RESUME"
        or any(
            d.get(k) is not False
            for d in (snapshots, proof)
            for k in (
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
    ):
        raise ValueError("exact independently validated development lineage required")
    snap_rows = {r["index"]: r for r in snapshots["rows"]}
    proof_rows = {r["index"]: r for r in proof["rows"]}
    source_rows = {r["index"]: r for r in source["records"]}
    if (
        len(snapshots["rows"]) != 24
        or len(proof["rows"]) != 24
        or len(snap_rows) != 24
        or set(snap_rows) != set(proof_rows)
    ):
        raise ValueError("complete unique matched snapshot and inference rows required")
    scene = root / "scene.xml"
    if scene.is_symlink() or digest(scene) != snapshots["scene_hash"]:
        raise ValueError("exact source-bound native scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for index in sorted(snap_rows):
        item, proof_row, original = snap_rows[index], proof_rows[index], source_rows[index]
        if (
            item["snapshot_hash"] != proof_row["snapshot_hash"]
            or item["context"] != proof_row["context"]
            or item["side"] != proof_row["side"]
            or item["receiver_lateral_offset_m"] != proof_row["receiver_lateral_offset_m"]
            or any(
                proof_row.get(k) is not True
                for k in ("observation_exact", "action_exact", "phase_exact", "target_exact")
            )
        ):
            raise ValueError("matched independently verified own observation required")
        snap = args.snapshots / item["snapshot"]
        physical, velocity = (
            bind(root, original, "physics"),
            bind(root, original, "receiver_velocity"),
        )
        if (
            snap.is_symlink()
            or snap.resolve().parent != args.snapshots.resolve()
            or digest(snap) != item["snapshot_hash"]
            or physical is None
            or velocity is None
        ):
            raise ValueError("confined snapshot and original own histories required")
        data = mujoco.MjData(model)
        with np.load(snap, allow_pickle=False) as saved:
            mujoco.mj_setState(model, data, saved["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        layout.angular_velocity_body(data)
        q0, v0 = layout.canonical(data)
        feature = receiver_ball_observation(q0, v0)
        with (
            np.load(physical, allow_pickle=False) as raw,
            np.load(velocity, allow_pickle=False) as actor,
        ):
            if not np.array_equal(data.qpos, raw["qpos"][139, -1]) or not np.array_equal(
                data.qvel, raw["qvel"][139, -1]
            ):
                raise ValueError("snapshot must equal original previous control")
            observation = actor["obs"][140].copy()
            if (
                observation.shape != (98,)
                or observation.dtype != np.float32
                or not np.isfinite(observation).all()
            ):
                raise ValueError("exact finite own pre-intervention observation required")
            data.qpos[:], data.qvel[:] = raw["qpos"][140, 0], raw["qvel"][140, 0]
        mujoco.mj_forward(model, data)
        q1, v1 = layout.canonical(data)
        numerical = (receiver_ball_observation(q1, v1)[:3] - feature[:3]) / model.opt.timestep
        if (
            digest(snap) != item["snapshot_hash"]
            or digest(physical) != original["artifacts"]["physics"]["hash"]
            or digest(velocity) != original["artifacts"]["receiver_velocity"]["hash"]
        ):
            raise ValueError("source changed during feature extraction")
        rows.append(
            dict(
                index=index,
                context=item["context"],
                side=item["side"],
                receiver_lateral_offset_m=item["receiver_lateral_offset_m"],
                observation=observation.tolist(),
                ball_feature=feature.tolist(),
                finite_difference_velocity_error_mps=(numerical - feature[3:]).tolist(),
            )
        )
    if (
        digest(snapshot_path) != args.snapshot_hash
        or digest(args.velocity_proof) != args.velocity_proof_hash
        or digest(root / "complete.json") != source_hash
        or digest(audit) != args.audit_hash
        or digest(scene) != snapshots["scene_hash"]
    ):
        raise ValueError("lineage changed during extraction")
    clean(args.expected_head)
    errors = np.asarray([r["finite_difference_velocity_error_mps"] for r in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="CAUSAL_RECEIVER_FEATURES_DEV_ONLY_NOT_TRAINED",
            source_head=args.expected_head,
            source_hash=source_hash,
            audit_hash=args.audit_hash,
            snapshot_hash=args.snapshot_hash,
            velocity_proof_hash=args.velocity_proof_hash,
            scene_hash=snapshots["scene_hash"],
            max_error_norm_mps=float(np.linalg.norm(errors, axis=1).max()),
            rows=rows,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
