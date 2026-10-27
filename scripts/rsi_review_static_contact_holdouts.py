"""Independent readback of all four fixed, preregistered SIM holdouts.

No new integration or independent execution: validate hashes, physical force
labels, common prefixes and frame-end geometry from retained private futures.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

ROOT = Path("/code/rosclaw/soccer_native_evidence")
CASES = {
    "yminus-vminus": "aa52c5a955311c7d7e7cef6d91574011fa25acac99cda22b2b6bafb90887e952",
    "yminus-vplus": "2e0f2d49f5994721785c00cf283116612b342bfa727f07c8f54b41d15b5b2774",
    "yplus-vminus": "14a07d357ab2ec11df573aca47450139ade75807f5509e010b7ad69db83b6b3f",
    "yplus-vplus": "295880f29e327943af7fb37a9f74a4ff8d80fa2349a62a693718dbf95afe1d1b",
}
TRAINING_CASES = {
    "yminus-vminus": "20265a84fb10503bf964e6abb6eaa78ac573622a775dbe4e6a7e8a43e154b4f0",
    "yminus-vplus": "11d5b80fce212b3525eab76dbccd8ea535bf621d713f201115c6e705e17f5f76",
    "yplus-vminus": "065a32238ba31763f933ee865cf1028f94370271dfebd983c79082616a2a735b",
    "yplus-vplus": "92f7e2eae3b09400c08d7e493900d339d7139cbb668ba915285b0fa0d52dc8f5",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    clean(args.expected_head)
    if args.output.exists():
        raise ValueError("unused output required")
    scene = ROOT / "rsi-paid-torch-native-full16-20261021/cached-standard/declared-scene.xml"
    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact compiled model required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    scratch = mujoco.MjData(model)
    ball_geom = int(np.flatnonzero(model.geom_bodyid == lo.ball)[0])
    feet = {
        side: [model.geom(f"a/{side}_foot{i}_collision").id for i in range(1, 8)]
        for side in ("left", "right")
    }
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            cases=TRAINING_CASES if args.training else CASES,
            training_split=args.training,
            source_head=args.expected_head,
            raw_hashes_verified=True,
            independently_integrated_native_steps=0,
            geometry_frame_end_only=True,
            physical_contact_authority="recorded_native_force",
            hardware_authorized=False,
        ),
    )
    results = {}
    for case, expected_hash in (TRAINING_CASES if args.training else CASES).items():
        prefix = "rsi-approach-train" if args.training else "rsi-static-holdout"
        root = ROOT / f"{prefix}-{case}-20261027"
        report_path = root / "complete.json"
        if report_path.is_symlink() or digest(report_path) != "sha256:" + expected_hash:
            raise ValueError(f"exact holdout report required: {case}")
        report = json.loads(report_path.read_text())
        expected_status = (
            "CFTRACK_STATIC_APPROACH_TRAINING_DATA_NOT_PROMOTED"
            if args.training
            else "STATIC_CONTACT_PAIRED_HOLDOUT_NOT_PROMOTED"
        )
        case_key = "training_case" if args.training else "holdout_id"
        if report["status"] != expected_status or report[case_key] != case:
            raise ValueError("matching unpromoted holdout identity required")
        arms = {}
        traces = {}
        arm_names = (
            ["parent", *(f"hg{i}-{side}" for i in range(4) for side in ("minus", "plus"))]
            if args.training
            else ["parent", "candidate"]
        )
        for arm in arm_names:
            path = root / f"{arm}.npz"
            record = report["records"][arm]
            if (
                path.is_symlink()
                or digest(path) != record["artifact_hash"]
                or record["complete"] is not True
            ):
                raise ValueError("exact complete trace required")
            with np.load(path, allow_pickle=False) as f:
                arrays = {key: f[key].copy() for key in f.files}
            for key, shape in (
                ("qpos", (3000, model.nq)),
                ("qvel", (3000, model.nv)),
                ("torque", (3000, model.nu)),
                ("ball_forces", (3000, 1, 3)),
                ("targets", (193, 29)),
            ):
                if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                    raise ValueError("complete finite native arrays required")
            traces[arm] = arrays
            q = arrays["qpos"].reshape(300, 10, model.nq)
            force = arrays["ball_forces"].reshape(300, 10, 1, 3).max(axis=1)
            force6 = np.zeros((300, 1, 6), np.float64)
            force6[:, :, :3] = force
            outcome = label_first_touch(
                dict(
                    force_n=force6,
                    pelvis_z_per_substep_m=q[:, :, lo.rq + 2][:, None, :],
                    ball_position_after_step_m=q[:, -1, lo.bq : lo.bq + 3][:, None, :],
                )
            ).to_dict()
            if outcome != record["outcome"]:
                raise ValueError("independent force-based outcome differs from source")
            minimum = {side: (float("inf"), None) for side in feet}
            for frame in range(107, 300):
                scratch.qpos[:] = q[frame, -1]
                mujoco.mj_forward(model, scratch)
                for side, geoms in feet.items():
                    gap = min(
                        float(
                            mujoco.mj_geomDistance(
                                model, scratch, ball_geom, geom, 2.0, np.zeros(6)
                            )
                        )
                        for geom in geoms
                    )
                    if gap < minimum[side][0]:
                        minimum[side] = (gap, frame)
            arms[arm] = dict(
                outcome=outcome,
                minimum_frame_end_gap_m=minimum,
                peak_ball_forces_n=force.max(axis=(0, 1)).tolist(),
                artifact_hash=record["artifact_hash"],
            )
        prefix_exact = all(
            np.array_equal(traces["parent"][k][:1070], traces[arm][k][:1070])
            for arm in arm_names[1:]
            for k in ("qpos", "qvel", "torque", "ball_forces")
        )
        if not prefix_exact:
            raise ValueError("candidate and parent must start from same actual prefix")
        results[case] = dict(
            source_hash="sha256:" + expected_hash, shared_prefix_exact=True, arms=arms
        )
        write_once(args.output / f"{case}.json", results[case])
    write_once(
        args.output / "complete.json",
        dict(
            status=(
                "STATIC_TRAINING_HASH_FORCE_GEOMETRY_READBACK_NOT_EXECUTION"
                if args.training
                else "STATIC_HOLDOUT_HASH_FORCE_GEOMETRY_READBACK_NOT_EXECUTION"
            ),
            records=results,
            independent_native_steps=0,
            reference_geometry_only=True,
            task_promoted=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
