"""Test causal provisional-support subset on two consumed static source frames."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--control-hash", required=True)
    p.add_argument("--control-review", type=Path, required=True)
    p.add_argument("--control-review-hash", required=True)
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
        or a.control.is_symlink()
        or a.control_review.is_symlink()
        or digest(a.control) != a.control_hash
        or digest(a.control_review) != a.control_review_hash
    ):
        raise ValueError("unused output and exact source/control audit required")
    control, review = json.loads(a.control.read_text()), json.loads(a.control_review.read_text())
    if (
        control["status"] != "REFERENCE_SUPPORT_IK_TWO_STATIC_POSES_RECORDED_NOT_PHYSICS"
        or review["status"]
        != "REFERENCE_SUPPORT_IK_INDEPENDENT_DISTANCE_READBACK_EXACT_NOT_PHYSICS"
        or review["source_hash"] != a.control_hash
        or [(r["source_file"], r["source_hash"]) for r in control["records"]] != list(CASES)
        or any(r["result"]["reference_geometry_feasible"] for r in control["records"])
        or digest(a.scene) != control["scene_hash"]
    ):
        raise ValueError("independently checked original two-support negative control required")
    paths = [(a.dataset / name, expected) for name, expected in CASES]
    if any(path.is_symlink() or digest(path) != expected for path, expected in paths):
        raise ValueError("exact source references required")
    import mujoco
    import numpy as np
    import scipy

    from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
    from rosclaw_soccer.rsi.reference_contact_diagnostic import (
        provisional_prior_support,
        reference_ankle_positions,
    )
    from rosclaw_soccer.rsi.reference_support_ik import fit_reference_support_pose
    from rosclaw_soccer.rsi.restricted_motion_pickle import read_motion
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(a.scene.read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != MODEL_HASH:
        raise ValueError("exact canonical native body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            control_hash=a.control_hash,
            control_review_hash=a.control_review_hash,
            source_data=CASES,
            scene_hash=control["scene_hash"],
            model_hash=MODEL_HASH,
            fixed_development_frames=[r["frame"] for r in control["records"]],
            frame_selection_used_whole_source_and_is_not_causal=True,
            support_rule=(
                "current and two prior labels plus backward ankle speed <=0.3m/s; "
                "abstain if empty"
            ),
            speed_threshold_m_s=0.3,
            source_labels_changed=False,
            support_truth_not_proven=True,
            stance_xy_limit_m=0.02,
            all_feet_penetration_limit_m=0.002,
            root_delta_limit_m=0.15,
            leg_delta_limit_rad=0.2,
            max_evaluations_per_frame=2000,
            scipy_version=scipy.__version__,
            static_reference_only=True,
            native_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records = []
    try:
        for old, (path, expected) in zip(control["records"], paths, strict=True):
            fields = read_motion(path, expected_sha256=expected)["fields"]
            frame = old["frame"]
            ankles = reference_ankle_positions(model, fields, expected_model_hash=MODEL_HASH)
            mask = fields["contact_mask"]
            if not np.isin(mask, [0, 1]).all():
                raise ValueError("binary source labels required; no implicit relabeling")
            support = provisional_prior_support(
                ankles, mask.astype(bool), frame=frame, fps=float(fields["fps"])
            )
            mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
            root = np.r_[fields["root_trans_offset"][frame], mapped["root_rot_wxyz"][frame]].astype(
                np.float64
            )
            joints = mapped["dof29"][frame].astype(np.float64)
            serial_support = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in support.items()
            }
            write_once(
                a.output / (path.stem + "-before.json"),
                dict(
                    source_hash=expected,
                    frame=frame,
                    original_claimed_contacts=mask[frame].tolist(),
                    provisional_support=serial_support,
                    root_pose_wxyz=root.tolist(),
                    joints_hg29=joints.tolist(),
                ),
            )
            result = None
            if not support["abstained"]:
                fitted = fit_reference_support_pose(
                    model,
                    root,
                    joints,
                    support["provisional_mask"],
                    expected_model_hash=MODEL_HASH,
                    max_ankle_xy_change_m=0.02,
                    maximum_evaluations=2000,
                )
                result = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fitted.items()
                }
            row = dict(
                source_file=path.name,
                source_hash=expected,
                frame=frame,
                original_claimed_contacts=mask[frame].tolist(),
                provisional_support=serial_support,
                negative_control_geometry_feasible=old["result"]["reference_geometry_feasible"],
                result=result,
                source_labels_not_changed=True,
            )
            records.append(row)
            write_once(a.output / (path.stem + ".json"), row)
            print(
                json.dumps(
                    dict(
                        source=path.name,
                        frame=frame,
                        mask=serial_support["provisional_mask"],
                        feasible=None if result is None else result["reference_geometry_feasible"],
                    )
                ),
                flush=True,
            )
        if (
            digest(a.control) != a.control_hash
            or digest(a.control_review) != a.control_review_hash
            or digest(a.scene) != control["scene_hash"]
            or any(digest(path) != expected for path, expected in paths)
        ):
            raise ValueError("pinned source or control changed during static fit")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PROVISIONAL_PRIOR_SUPPORT_TWO_STATIC_POSES_RECORDED_NOT_PHYSICS",
                source_head=a.expected_head,
                control_hash=a.control_hash,
                control_review_hash=a.control_review_hash,
                scene_hash=control["scene_hash"],
                model_hash=MODEL_HASH,
                records=records,
                source_labels_changed=False,
                frame_selection_not_causal=True,
                independent_review_pending=True,
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
                status="PROVISIONAL_SUPPORT_FAILED_CLOSED",
                source_head=a.expected_head,
                completed_cases=len(records),
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
