"""Independently recompute support candidate geometry, with no optimizer or steps."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--provisional", action="store_true")
    p.add_argument("--control", type=Path)
    p.add_argument("--control-review", type=Path)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean
    from scripts.rsi_probe_reference_support_ik import CASES, MODEL_HASH

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if a.output.exists() or complete.is_symlink() or digest(complete) != a.source_hash:
        raise ValueError("unused independent audit and exact static source required")
    source = json.loads(complete.read_text())
    if (
        source["status"]
        != (
            "PROVISIONAL_PRIOR_SUPPORT_TWO_STATIC_POSES_RECORDED_NOT_PHYSICS"
            if a.provisional
            else "REFERENCE_SUPPORT_IK_TWO_STATIC_POSES_RECORDED_NOT_PHYSICS"
        )
        or [(r["source_file"], r["source_hash"]) for r in source["records"]] != list(CASES)
        or source["model_hash"] != MODEL_HASH
        or source["native_steps"] != 0
        or source["neural_optimizer_updates"] != 0
        or a.scene.is_symlink()
        or digest(a.scene) != source["scene_hash"]
    ):
        raise ValueError("exact two-source static fit and scene required")
    if a.provisional:
        if (
            a.control is None
            or a.control_review is None
            or a.control.is_symlink()
            or a.control_review.is_symlink()
            or digest(a.control) != source["control_hash"]
            or digest(a.control_review) != source["control_review_hash"]
        ):
            raise ValueError("exact provisional negative control and independent review required")
        control = json.loads(a.control.read_text())
        previous_review = json.loads(a.control_review.read_text())
        if (
            previous_review["status"]
            != "REFERENCE_SUPPORT_IK_INDEPENDENT_DISTANCE_READBACK_EXACT_NOT_PHYSICS"
            or previous_review["source_hash"] != source["control_hash"]
            or any(row["result"]["reference_geometry_feasible"] for row in control["records"])
        ):
            raise ValueError("independently audited negative control required")
    paths = [(a.dataset / name, expected) for name, expected in CASES]
    if any(path.is_symlink() or digest(path) != expected for path, expected in paths):
        raise ValueError("exact source data required")
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
    from rosclaw_soccer.rsi.reference_contact_diagnostic import (
        provisional_prior_support,
        reference_ankle_positions,
    )
    from rosclaw_soccer.rsi.reference_support_geometry import support_distances
    from rosclaw_soccer.rsi.reference_support_ik_audit import audit_support_candidate
    from rosclaw_soccer.rsi.restricted_motion_pickle import read_motion
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(a.scene.read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != MODEL_HASH:
        raise ValueError("exact native body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            source_scene_hash=source["scene_hash"],
            source_data=CASES,
            model_hash=MODEL_HASH,
            selection_and_mapping_math_shared=True,
            candidate_gap_reader="mujoco.mj_geomDistance",
            no_optimizer_replayed=True,
            provisional_support_rule_shared=a.provisional,
            native_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    rows = []
    for source_row, (path, expected) in zip(source["records"], paths, strict=True):
        fields = read_motion(path, expected_sha256=expected)["fields"]
        gaps, labels = support_distances(model, fields, expected_model_hash=MODEL_HASH)
        if not labels.all(axis=1).any():
            raise ValueError("source claimed double support required")
        scores = np.where(labels.all(axis=1), np.abs(gaps[:, 0] - gaps[:, 1]), -1.0)
        frame = int(np.argmax(scores))
        if frame != source_row["frame"] or (
            not a.provisional
            and float(scores[frame]) != source_row["maximum_double_support_gap_difference_m"]
        ):
            raise ValueError("source-selected static frame does not reproduce")
        chosen = labels[frame]
        if a.provisional:
            if (
                source_row["original_claimed_contacts"] != labels[frame].astype(float).tolist()
                or source_row["negative_control_geometry_feasible"] is not False
            ):
                raise ValueError("original claimed labels and failed control must be preserved")
            ankles = reference_ankle_positions(model, fields, expected_model_hash=MODEL_HASH)
            support = provisional_prior_support(
                ankles, labels, frame=frame, fps=float(fields["fps"])
            )
            serial = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in support.items()}
            if serial != source_row["provisional_support"]:
                raise ValueError("provisional support decision does not reproduce")
            chosen = support["provisional_mask"]
            if support["abstained"]:
                if source_row["result"] is not None:
                    raise ValueError("abstention cannot produce a reference candidate")
                row = dict(source_file=path.name, source_hash=expected, frame=frame, abstained=True)
                rows.append(row)
                write_once(a.output / (path.stem + ".json"), row)
                continue
        mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
        source_root = np.r_[
            fields["root_trans_offset"][frame], mapped["root_rot_wxyz"][frame]
        ].astype(np.float64)
        source_joints = mapped["dof29"][frame].astype(np.float64)
        result = source_row["result"]
        measured = audit_support_candidate(
            model,
            source_root,
            source_joints,
            np.array(result["root_pose_wxyz"], np.float64),
            np.array(result["joints_hg29"], np.float64),
            chosen,
            expected_model_hash=MODEL_HASH,
            max_claimed_ankle_xy_change_m=result.get("claimed_ankle_xy_limit_m"),
        )
        if (
            not np.allclose(
                measured["source_foot_gaps_m"], result["original_foot_gaps_m"], rtol=0, atol=1e-12
            )
            or not np.allclose(
                measured["candidate_foot_gaps_m"],
                result["candidate_foot_gaps_m"],
                rtol=0,
                atol=1e-12,
            )
            or measured["reference_geometry_feasible"] != result["reference_geometry_feasible"]
            or measured["leg_trust_boundary_active"] != result["joint_trust_boundary_active"]
            or (
                result.get("claimed_ankle_xy_limit_m") is not None
                and not np.isclose(
                    measured["max_claimed_ankle_xy_displacement_m"],
                    result["maximum_claimed_ankle_xy_change_m"],
                    rtol=0,
                    atol=1e-12,
                )
            )
        ):
            raise ValueError("independent geometry or numeric trust-bound decision differs")
        row = dict(
            source_file=path.name,
            source_hash=expected,
            frame=frame,
            independently_measured=measured,
        )
        rows.append(row)
        write_once(a.output / (path.stem + ".json"), row)
        print(json.dumps(row), flush=True)
    if digest(a.scene) != source["scene_hash"] or any(digest(path) != h for path, h in paths):
        raise ValueError("source data or scene changed during audit")
    if a.provisional and (
        digest(a.control) != source["control_hash"]
        or digest(a.control_review) != source["control_review_hash"]
    ):
        raise ValueError("negative control changed during audit")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "PROVISIONAL_SUPPORT_INDEPENDENT_DISTANCE_READBACK_EXACT_NOT_PHYSICS"
                if a.provisional
                else "REFERENCE_SUPPORT_IK_INDEPENDENT_DISTANCE_READBACK_EXACT_NOT_PHYSICS"
            ),
            source_head=a.expected_head,
            source_hash=a.source_hash,
            records=rows,
            selection_and_mapping_math_shared=True,
            provisional_support_rule_shared=a.provisional,
            optimizer_replayed=False,
            native_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
