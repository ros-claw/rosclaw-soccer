"""Read back all static sequence frames without optimization or integration."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--prior-review", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean
    from scripts.rsi_probe_reference_support_ik import CASES, MODEL_HASH

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if a.output.exists() or complete.is_symlink() or digest(complete) != a.source_hash:
        raise ValueError("unused audit and exact static sequence source required")
    source = json.loads(complete.read_text())
    if (
        source["status"] != "PROVISIONAL_SUPPORT_SEQUENCE_STATIC_RECORDED_NOT_PHYSICS"
        or digest(a.prior) != source["source_hash"]
        or digest(a.prior_review) != source["review_hash"]
        or source["native_steps"] != 0
        or source["neural_optimizer_updates"] != 0
        or [(r["source_file"], r["source_hash"]) for r in source["records"]] != list(CASES)
    ):
        raise ValueError("exact reviewed source lineage and two cases required")
    prior = json.loads(a.prior.read_text())
    if digest(a.scene) != prior["scene_hash"]:
        raise ValueError("exact prior canonical scene required")
    paths = [(a.dataset / name, h) for name, h in CASES]
    if any(path.is_symlink() or digest(path) != h for path, h in paths):
        raise ValueError("exact external source data required")
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.reference_contact_diagnostic import (
        provisional_prior_support,
        reference_ankle_positions,
    )
    from rosclaw_soccer.rsi.reference_sequence_gate import assess_reference_sequence
    from rosclaw_soccer.rsi.reference_support_ik_audit import audit_support_candidate
    from rosclaw_soccer.rsi.restricted_motion_pickle import read_motion
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(a.scene.read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != MODEL_HASH:
        raise ValueError("exact canonical body required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    ankle_ids = [model.body(f"a/{side}_ankle_roll_link").id for side in ("left", "right")]
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            source_prior_hash=source["source_hash"],
            source_review_hash=source["review_hash"],
            scene_hash=prior["scene_hash"],
            source_data=CASES,
            model_hash=MODEL_HASH,
            maximum_frames=26,
            optimizer_replayed=False,
            source_mapping_support_rule_and_sequence_gate_math_shared=True,
            independently_requeried_candidate_geomDistance=True,
            no_missing_frame_padding=True,
            native_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records = []
    current = None
    try:
        for case, (path, expected), bounds in zip(
            source["records"], paths, ((142, 155), (100, 113)), strict=True
        ):
            fields = read_motion(path, expected_sha256=expected)["fields"]
            if int(fields["fps"]) != 30 or not np.isin(fields["contact_mask"], [0, 1]).all():
                raise ValueError("binary source labels and declared 30Hz required")
            raw_labels = fields["contact_mask"].astype(bool)
            ref_ankles = reference_ankle_positions(model, fields, expected_model_hash=MODEL_HASH)
            mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
            indices = list(range(*bounds))
            if case["frames"] != indices or [r["frame"] for r in case["rows"]] != indices:
                raise ValueError("all predeclared ordered frames required")
            n = len(indices)
            roots = np.full((n, 7), np.nan)
            joints = np.full((n, 29), np.nan)
            ankles = np.full((n, 2, 3), np.nan)
            nonfoot = np.full(n, np.nan)
            masks, valid, boundary = np.zeros((n, 2), bool), np.zeros(n, bool), np.zeros(n, bool)
            data = mujoco.MjData(model)
            data.qpos[layout.bq : layout.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
            checked = []
            for slot, row in enumerate(case["rows"]):
                frame = row["frame"]
                current = dict(source_file=path.name, frame=frame)
                decision = provisional_prior_support(ref_ankles, raw_labels, frame=frame, fps=30.0)
                serial = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in decision.items()
                }
                if (
                    serial != row["provisional_support"]
                    or row["original_labels"] != raw_labels[frame].tolist()
                ):
                    raise ValueError("source provisional mask or original labels differ")
                masks[slot] = decision["provisional_mask"]
                measured = None
                if decision["abstained"]:
                    if (
                        row["event"] != "ABSTAINED"
                        or row["result"] is not None
                        or row["independent_geometry"] is not None
                    ):
                        raise ValueError("abstention must not have a fabricated candidate")
                else:
                    fit = row["result"]
                    if row["event"] != "FITTED" or type(fit) is not dict:
                        raise ValueError("non-abstaining source must have recorded fit")
                    original_root = np.r_[
                        fields["root_trans_offset"][frame], mapped["root_rot_wxyz"][frame]
                    ].astype(np.float64)
                    original_joints = mapped["dof29"][frame].astype(np.float64)
                    target_root = np.array(fit["root_pose_wxyz"], np.float64)
                    target_joints = np.array(fit["joints_hg29"], np.float64)
                    measured = audit_support_candidate(
                        model,
                        original_root,
                        original_joints,
                        target_root,
                        target_joints,
                        masks[slot],
                        expected_model_hash=MODEL_HASH,
                        max_claimed_ankle_xy_change_m=0.02,
                    )
                    if (
                        measured != row["independent_geometry"]
                        or measured["reference_geometry_feasible"]
                        != fit["reference_geometry_feasible"]
                        or not np.allclose(
                            measured["candidate_foot_gaps_m"],
                            fit["candidate_foot_gaps_m"],
                            rtol=0,
                            atol=1e-12,
                        )
                    ):
                        raise ValueError("independent per-frame measured geometry differs")
                    boundary[slot] = (
                        fit["joint_trust_boundary_active"] or fit["root_trust_boundary_active"]
                    )
                    if (
                        fit["reference_geometry_feasible"]
                        and measured["candidate_minimum_nonfoot_floor_distance_m"] >= -0.002
                    ):
                        valid[slot] = True
                        roots[slot], joints[slot] = target_root, target_joints
                        nonfoot[slot] = measured["candidate_minimum_nonfoot_floor_distance_m"]
                        data.qpos[layout.rq : layout.rq + 7], data.qpos[layout.qi] = (
                            target_root,
                            target_joints,
                        )
                        mujoco.mj_kinematics(model, data)
                        ankles[slot] = data.xpos[ankle_ids]
                checked.append(
                    dict(
                        frame=frame,
                        abstained=decision["abstained"],
                        candidate_valid=bool(valid[slot]),
                        measured=measured,
                    )
                )
            metrics = assess_reference_sequence(
                roots, joints, ankles, masks, valid, boundary, nonfoot, declared_fps=30.0
            )
            if metrics != case["metrics"]:
                raise ValueError("source sequence gate metrics do not reproduce")
            result = dict(
                source_file=path.name,
                source_hash=expected,
                frames=indices,
                metrics=metrics,
                checked_frames=checked,
            )
            records.append(result)
            write_once(a.output / (path.stem + ".json"), result)
            print(
                json.dumps(
                    dict(
                        source=path.name,
                        valid=metrics["valid_frames"],
                        gate=metrics["reference_sequence_gate_passed"],
                    )
                ),
                flush=True,
            )
        if (
            digest(a.prior) != source["source_hash"]
            or digest(a.prior_review) != source["review_hash"]
            or digest(a.scene) != prior["scene_hash"]
            or any(digest(path) != h for path, h in paths)
        ):
            raise ValueError("pinned inputs changed during static audit")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PROVISIONAL_SUPPORT_SEQUENCE_READBACK_EXACT_NOT_PHYSICS",
                source_head=a.expected_head,
                source_hash=a.source_hash,
                records=records,
                source_mapping_support_rule_and_sequence_gate_math_shared=True,
                optimizer_replayed=False,
                native_steps=0,
                neural_optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="PROVISIONAL_SEQUENCE_READBACK_FAILED_CLOSED",
                source_head=a.expected_head,
                completed_cases=len(records),
                current=current,
                error_type=type(error).__name__,
                reason=str(error),
                native_steps=0,
                neural_optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
