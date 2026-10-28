"""Match train-only motion contexts to audited two-body precontact states."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receive_reference_matching import match_error, yaw_local
from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import report


def eligible_geometry_events(geometry: dict) -> set[tuple]:
    return {
        (row["capture"], row["event_frame"], row["foot"])
        for row in geometry["samples"]
        if abs(row["samples"][1]["foot_surface_gap_m"]) <= 0.01
        and row["samples"][1]["ball_floor_clearance_m"] >= -0.01
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset-root", "screen", "source", "audit", "snapshots", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "screen-hash",
        "source-hash",
        "audit-hash",
        "snapshot-hash",
        "scene-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--geometry", type=Path)
    p.add_argument("--geometry-hash")
    a = p.parse_args()
    clean(a.expected_head)
    snapshot_receipt = a.snapshots / "complete.json"
    scene = a.source / "scene.xml"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(x.is_symlink() for x in (a.screen, snapshot_receipt, scene))
        or digest(a.screen) != a.screen_hash
        or digest(snapshot_receipt) != a.snapshot_hash
        or digest(scene) != a.scene_hash
    ):
        raise ValueError("exact source receipts, scene and unused external output required")
    screen, snapshots = (json.loads(x.read_text()) for x in (a.screen, snapshot_receipt))
    if (a.geometry is None) != (a.geometry_hash is None):
        raise ValueError("geometry receipt and hash must be supplied together")
    allowed = None
    if a.geometry is not None:
        if a.geometry.is_symlink() or digest(a.geometry) != a.geometry_hash:
            raise ValueError("exact training geometry receipt required")
        geometry = json.loads(a.geometry.read_text())
        if (
            geometry.get("status") != "OMNICONTACT_REFERENCE_FOOT_GEOMETRY_NOT_PHYSICS"
            or geometry.get("screen_hash") != a.screen_hash
            or geometry.get("scene_hash") != a.scene_hash
            or geometry.get("promotion_authorized") is not False
        ):
            raise ValueError("source-bound unpromoted geometry receipt required")
        allowed = eligible_geometry_events(geometry)
        if not allowed:
            raise ValueError("no source events satisfy declared static onset geometry")
    source, _, source_hash = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if (
        source_hash != a.source_hash
        or screen.get("status") != "OMNICONTACT_TRAIN_RECEIVING_REFERENCE_SCREEN_NOT_PHYSICS"
        or screen.get("validation_or_test_motion_opened") is not False
        or screen.get("rejected_inputs") != []
        or snapshots.get("status")
        != "RELAY_NONDEFAULT_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
        or snapshots.get("source_hash") != a.source_hash
        or snapshots.get("audit_hash") != a.audit_hash
        or snapshots.get("scene_hash") != a.scene_hash
        or len(snapshots.get("rows", [])) != 24
    ):
        raise ValueError("audited training screen and exact 24 physical snapshots required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    lo = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    state = mujoco.MjData(model)
    actual = []
    for row in snapshots["rows"]:
        source_row = source["records"][row["index"]]
        path = a.snapshots / row["snapshot"]
        if (
            path.parent != a.snapshots
            or path.is_symlink()
            or digest(path) != row["snapshot_hash"]
            or source_row["artifacts"]["physics"]["hash"] != row["original_physics_hash"]
        ):
            raise ValueError("exact source-bound snapshot required")
        with np.load(path, allow_pickle=False) as data:
            saved = data["state"].copy()
        if saved.shape != (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),):
            raise ValueError("complete native integration snapshot required")
        mujoco.mj_setState(model, state, saved, mujoco.mjtState.mjSTATE_INTEGRATION)
        q, v = lo.canonical(state)
        actual.append(
            (
                row,
                dict(
                    ball_relative=yaw_local(q[3:7], q[36:39] - q[:3]),
                    velocity_relative=yaw_local(q[3:7], v[35:38] - v[:3]),
                    joints=q[7:36].copy(),
                    root_height=float(q[2]),
                ),
            )
        )
    root = a.dataset_root.resolve()
    candidates, pins = [], []
    for row in screen["rows"]:
        if not row["events"]:
            continue
        path = root / row["source_relative_path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root / "npz" / "soccer")
            or digest(path) != row["source_hash"]
        ):
            raise ValueError("unchanged confined training motion required")
        with np.load(path, allow_pickle=False) as z:
            fps = int(z["fps"][0])
            for event in row["events"]:
                if (
                    allowed is not None
                    and (row["capture"], event["frame"], event["foot"]) not in allowed
                ):
                    continue
                frame = event["frame"] - round(0.5 * fps)
                if frame < 0:
                    continue
                ball = np.asarray(z["object_pos_w"][frame], dtype=np.float64).copy()
                ball[2] += 0.06  # Explicit same reference-only lift as prior pilot.
                if ball[2] < 0.10:
                    continue
                root_position = z["body_pos_w"][frame, 0]
                quat = z["body_quat_w"][frame, 0]
                feature = dict(
                    ball_relative=yaw_local(quat, ball - root_position),
                    velocity_relative=yaw_local(
                        quat, z["object_lin_vel_w"][frame] - z["body_lin_vel_w"][frame, 0]
                    ),
                    joints=np.asarray(z["joint_pos"][frame, list(LAB_TO_MUJOCO)], dtype=np.float64),
                    root_height=float(root_position[2]),
                )
                candidates.append((row, event, frame, feature))
        if digest(path) != row["source_hash"]:
            raise ValueError("training motion changed during matching")
        pins.append((path, row["source_hash"]))
    if not candidates:
        raise ValueError("no admissible training reference states")
    rows = []
    for snapshot, features in actual:
        scored = [(match_error(features, candidate[3]), candidate) for candidate in candidates]
        scored.sort(key=lambda x: (x[0]["score"], x[1][0]["capture"], x[1][2], x[1][1]["foot"]))
        error, (motion, event, frame, _) = scored[0]
        rows.append(
            dict(
                snapshot_index=snapshot["index"],
                context=snapshot["context"],
                side=snapshot["side"],
                snapshot_hash=snapshot["snapshot_hash"],
                reference_capture=motion["capture"],
                reference_hash=motion["source_hash"],
                reference_path=motion["source_relative_path"],
                reference_frame=frame,
                reference_event_frame=event["frame"],
                reference_foot=event["foot"],
                **error,
            )
        )
    if (
        digest(a.screen) != a.screen_hash
        or digest(snapshot_receipt) != a.snapshot_hash
        or digest(scene) != a.scene_hash
        or digest(a.source / "complete.json") != a.source_hash
        or digest(a.audit) != a.audit_hash
        or (a.geometry is not None and digest(a.geometry) != a.geometry_hash)
        or any(digest(path) != h for path, h in pins)
    ):
        raise ValueError("inputs changed during matching")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="TRAIN_REFERENCE_TO_ACTUAL_TWO_BODY_STATE_MATCH_NOT_PHYSICS",
            source_head=a.expected_head,
            screen_hash=a.screen_hash,
            geometry_hash=a.geometry_hash,
            onset_surface_abs_gap_max_m=None if allowed is None else 0.01,
            onset_ball_floor_min_m=None if allowed is None else -0.01,
            onset_geometry_is_before_reference_ball_lift=True,
            post_lift_contact_geometry_verified=False,
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            snapshot_hash=a.snapshot_hash,
            scene_hash=a.scene_hash,
            eligible_reference_states=len(candidates),
            actual_states=len(rows),
            geometrically_compatible=sum(r["geometry_state_compatible"] for r in rows),
            rows=rows,
            reference_lookback_s=0.5,
            reference_only_ball_height_lift_m=0.06,
            reference_event_is_retrospective=True,
            actual_snapshot_reset_or_step=False,
            validation_or_test_motion_opened=False,
            noncommercial_source=True,
            independent_fresh=False,
            commercial_publicity_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
