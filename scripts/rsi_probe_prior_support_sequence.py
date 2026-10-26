"""Audit static continuity around two consumed support transitions."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
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
    if (
        a.output.exists()
        or a.source.is_symlink()
        or a.review.is_symlink()
        or digest(a.source) != a.source_hash
        or digest(a.review) != a.review_hash
    ):
        raise ValueError("unused output and exact source/review required")
    source, review = json.loads(a.source.read_text()), json.loads(a.review.read_text())
    if (
        source["status"] != "PROVISIONAL_PRIOR_SUPPORT_TWO_STATIC_POSES_RECORDED_NOT_PHYSICS"
        or review["status"] != "PROVISIONAL_SUPPORT_INDEPENDENT_DISTANCE_READBACK_EXACT_NOT_PHYSICS"
        or review["source_hash"] != a.source_hash
        or [(r["source_file"], r["source_hash"], r["frame"]) for r in source["records"]]
        != [(CASES[0][0], CASES[0][1], 148), (CASES[1][0], CASES[1][1], 106)]
        or digest(a.scene) != source["scene_hash"]
    ):
        raise ValueError("exact reviewed two transition anchors and pinned scene required")
    paths = [(a.dataset / name, h) for name, h in CASES]
    if any(path.is_symlink() or digest(path) != h for path, h in paths):
        raise ValueError("exact source motion references required")
    import mujoco
    import numpy as np
    import scipy

    from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.reference_contact_diagnostic import (
        provisional_prior_support,
        reference_ankle_positions,
    )
    from rosclaw_soccer.rsi.reference_sequence_gate import assess_reference_sequence
    from rosclaw_soccer.rsi.reference_support_ik import fit_reference_support_pose
    from rosclaw_soccer.rsi.reference_support_ik_audit import audit_support_candidate
    from rosclaw_soccer.rsi.restricted_motion_pickle import read_motion
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(a.scene.read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != MODEL_HASH:
        raise ValueError("exact canonical SIM body required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    ankle_ids = [model.body(f"a/{side}_ankle_roll_link").id for side in ("left", "right")]
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            review_hash=a.review_hash,
            scene_hash=source["scene_hash"],
            source_data=CASES,
            model_hash=MODEL_HASH,
            windows=[list(range(142, 155)), list(range(100, 113))],
            fps_from_source=30.0,
            per_frame_budget=2000,
            maximum_residual_evaluations=52000,
            maximum_frames=26,
            ankle_xy_source_pose_limit_m=0.02,
            foot_gap_limit_m=0.01,
            foot_penetration_limit_m=0.002,
            root_speed_limit_m_s=0.3,
            joint_speed_limit_rad_s=3.0,
            support_ankle_speed_limit_m_s=0.3,
            nonfoot_floor_penetration_limit_m=0.002,
            trust_boundary_not_accepted=True,
            invalid_or_abstained_frames_never_padded=True,
            scipy_version=scipy.__version__,
            retrospectively_selected_development_windows=True,
            labels_and_mapping_unverified=True,
            native_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records, total_evaluations = [], 0
    current = None
    try:
        for old, (path, expected) in zip(source["records"], paths, strict=True):
            fields = read_motion(path, expected_sha256=expected)["fields"]
            if int(fields["fps"]) != 30 or not np.isin(fields["contact_mask"], [0, 1]).all():
                raise ValueError("source-declared 30Hz and binary labels required")
            raw_labels = fields["contact_mask"].astype(bool)
            ref_ankles = reference_ankle_positions(model, fields, expected_model_hash=MODEL_HASH)
            mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
            indices = list(range(old["frame"] - 6, old["frame"] + 7))
            n = len(indices)
            roots = np.full((n, 7), np.nan)
            joints = np.full((n, 29), np.nan)
            ankles = np.full((n, 2, 3), np.nan)
            nonfoot = np.full(n, np.nan)
            masks = np.zeros((n, 2), bool)
            valid = np.zeros(n, bool)
            boundary = np.zeros(n, bool)
            data = mujoco.MjData(model)
            data.qpos[layout.bq : layout.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
            rows = []
            for slot, frame in enumerate(indices):
                current = dict(source_file=path.name, frame=frame)
                support = provisional_prior_support(ref_ankles, raw_labels, frame=frame, fps=30.0)
                masks[slot] = support["provisional_mask"]
                support_json = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in support.items()
                }
                row = dict(
                    frame=frame,
                    original_labels=raw_labels[frame].tolist(),
                    provisional_support=support_json,
                    event="ABSTAINED" if support["abstained"] else "FITTED",
                    result=None,
                    independent_geometry=None,
                )
                if not support["abstained"]:
                    original_root = np.r_[
                        fields["root_trans_offset"][frame], mapped["root_rot_wxyz"][frame]
                    ].astype(np.float64)
                    original_joints = mapped["dof29"][frame].astype(np.float64)
                    fit = fit_reference_support_pose(
                        model,
                        original_root,
                        original_joints,
                        masks[slot],
                        expected_model_hash=MODEL_HASH,
                        max_ankle_xy_change_m=0.02,
                        maximum_evaluations=2000,
                    )
                    total_evaluations += fit["solver_evaluations"]
                    audited = audit_support_candidate(
                        model,
                        original_root,
                        original_joints,
                        fit["root_pose_wxyz"],
                        fit["joints_hg29"],
                        masks[slot],
                        expected_model_hash=MODEL_HASH,
                        max_claimed_ankle_xy_change_m=0.02,
                    )
                    if (
                        not np.allclose(
                            audited["candidate_foot_gaps_m"],
                            fit["candidate_foot_gaps_m"],
                            rtol=0,
                            atol=1e-12,
                        )
                        or audited["reference_geometry_feasible"]
                        != fit["reference_geometry_feasible"]
                    ):
                        raise ValueError("independent per-frame geometric readback differs")
                    row["result"] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()
                    }
                    row["independent_geometry"] = audited
                    boundary[slot] = (
                        fit["joint_trust_boundary_active"] or fit["root_trust_boundary_active"]
                    )
                    if (
                        fit["reference_geometry_feasible"]
                        and audited["candidate_minimum_nonfoot_floor_distance_m"] >= -0.002
                    ):
                        valid[slot] = True
                        roots[slot], joints[slot] = fit["root_pose_wxyz"], fit["joints_hg29"]
                        nonfoot[slot] = audited["candidate_minimum_nonfoot_floor_distance_m"]
                        data.qpos[layout.rq : layout.rq + 7], data.qpos[layout.qi] = (
                            roots[slot],
                            joints[slot],
                        )
                        mujoco.mj_kinematics(model, data)
                        ankles[slot] = data.xpos[ankle_ids]
                rows.append(row)
                write_once(a.output / f"{path.stem}-{frame:05d}.json", row)
            metrics = assess_reference_sequence(
                roots, joints, ankles, masks, valid, boundary, nonfoot, declared_fps=30.0
            )
            case = dict(
                source_file=path.name,
                source_hash=expected,
                frames=indices,
                metrics=metrics,
                rows=rows,
            )
            records.append(case)
            write_once(a.output / (path.stem + "-sequence.json"), case)
            print(json.dumps(dict(source=path.name, metrics=metrics)), flush=True)
        if (
            total_evaluations > 52000
            or digest(a.source) != a.source_hash
            or digest(a.review) != a.review_hash
        ):
            raise ValueError("bounded calls and unchanged source/review required")
        if digest(a.scene) != source["scene_hash"] or any(digest(path) != h for path, h in paths):
            raise ValueError("scene or source data changed during research")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PROVISIONAL_SUPPORT_SEQUENCE_STATIC_RECORDED_NOT_PHYSICS",
                source_head=a.expected_head,
                source_hash=a.source_hash,
                review_hash=a.review_hash,
                records=records,
                residual_evaluations=total_evaluations,
                source_labels_changed=False,
                native_steps=0,
                neural_optimizer_updates=0,
                physical_execution_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="PROVISIONAL_SEQUENCE_FAILED_CLOSED",
                source_head=a.expected_head,
                completed_cases=len(records),
                current=current,
                residual_evaluations=total_evaluations,
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
