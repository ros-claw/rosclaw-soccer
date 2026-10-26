"""Qualify two source-frame support IK hypotheses, never actuating a robot."""

import argparse
import json
from pathlib import Path

CASES = (
    (
        "B22_-_side_step_left_stageii.pkl",
        "sha256:17f192db3a375db28ad7b2ba2bbcecd643764b10350743d21f37d6c5979fc785",
    ),
    (
        "B23_-_side_step_right_stageii.pkl",
        "sha256:5d3b3d86d029a3f7550426de29f37e9315f408e5a94a50bb1a658d932220e945",
    ),
)
MODEL_HASH = "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--scene-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists() or a.scene.is_symlink() or digest(a.scene) != a.scene_hash:
        raise ValueError("unused output and exact declared scene required")
    paths = [(a.dataset / name, expected) for name, expected in CASES]
    if any(path.is_symlink() or digest(path) != expected for path, expected in paths):
        raise ValueError("both exact source references required")
    import mujoco
    import numpy as np
    import scipy

    from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
    from rosclaw_soccer.rsi.reference_support_geometry import support_distances
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
            scene_path=str(a.scene),
            scene_hash=a.scene_hash,
            compiled_model_hash=MODEL_HASH,
            dataset=str(a.dataset),
            cases=CASES,
            frame_selection="maximum absolute two-foot geometric gap among claimed double supports",
            maximum_root_z_change_m=0.15,
            maximum_leg_change_rad=0.2,
            evaluation_budget_per_pose=2000,
            contact_tolerance_m=0.01,
            penetration_tolerance_m=0.002,
            root_minimum_m=0.65,
            source_labels_and_23_to_29_mapping_assumed=True,
            optimizer="scipy.optimize.least_squares",
            scipy_version=scipy.__version__,
            single_frame_static_only=True,
            native_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records = []
    try:
        for path, expected in paths:
            fields = read_motion(path, expected_sha256=expected)["fields"]
            distances, labels = support_distances(model, fields, expected_model_hash=MODEL_HASH)
            eligible = labels.all(axis=1)
            if not eligible.any():
                raise ValueError("at least one claimed double-support frame required")
            scores = np.where(eligible, np.abs(distances[:, 0] - distances[:, 1]), -1.0)
            frame = int(np.argmax(scores))
            mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
            root = np.r_[fields["root_trans_offset"][frame], mapped["root_rot_wxyz"][frame]].astype(
                np.float64
            )
            joints = mapped["dof29"][frame].astype(np.float64)
            selected = labels[frame].copy()
            write_once(
                a.output / (path.stem + "-before.json"),
                dict(
                    source_hash=expected,
                    frame=frame,
                    claimed_contacts=selected.tolist(),
                    root_pose_wxyz=root.tolist(),
                    joints_hg29=joints.tolist(),
                    maximum_double_support_gap_difference_m=float(scores[frame]),
                ),
            )
            result = fit_reference_support_pose(
                model,
                root,
                joints,
                selected,
                expected_model_hash=MODEL_HASH,
                maximum_evaluations=2000,
            )
            serial = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
            row = dict(
                source_file=path.name,
                source_hash=expected,
                frame=frame,
                maximum_double_support_gap_difference_m=float(scores[frame]),
                result=serial,
            )
            records.append(row)
            write_once(a.output / (path.stem + ".json"), row)
            print(
                json.dumps(
                    dict(
                        source=path.name,
                        frame=frame,
                        feasible=result["reference_geometry_feasible"],
                        max_leg_change_rad=result["maximum_leg_change_rad"],
                    )
                ),
                flush=True,
            )
        if digest(a.scene) != a.scene_hash or any(
            digest(path) != expected for path, expected in paths
        ):
            raise ValueError("scene or source reference changed during static fit")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="REFERENCE_SUPPORT_IK_TWO_STATIC_POSES_RECORDED_NOT_PHYSICS",
                source_head=a.expected_head,
                scene_hash=a.scene_hash,
                model_hash=MODEL_HASH,
                records=records,
                scipy_version=scipy.__version__,
                source_contact_labels_assumed=True,
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
                status="REFERENCE_SUPPORT_IK_FAILED_CLOSED",
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
