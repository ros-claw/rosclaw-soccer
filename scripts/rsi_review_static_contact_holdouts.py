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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
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
            cases=CASES,
            source_head=args.expected_head,
            raw_hashes_verified=True,
            independently_integrated_native_steps=0,
            geometry_frame_end_only=True,
            physical_contact_authority="recorded_native_force",
            hardware_authorized=False,
        ),
    )
    results = {}
    for case, expected_hash in CASES.items():
        root = ROOT / f"rsi-static-holdout-{case}-20261027"
        report_path = root / "complete.json"
        if report_path.is_symlink() or digest(report_path) != "sha256:" + expected_hash:
            raise ValueError(f"exact holdout report required: {case}")
        report = json.loads(report_path.read_text())
        if (
            report["status"] != "STATIC_CONTACT_PAIRED_HOLDOUT_NOT_PROMOTED"
            or report["holdout_id"] != case
        ):
            raise ValueError("matching unpromoted holdout identity required")
        arms = {}
        traces = {}
        for arm in ("parent", "candidate"):
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
            np.array_equal(traces["parent"][k][:1070], traces["candidate"][k][:1070])
            for k in ("qpos", "qvel", "torque", "ball_forces")
        )
        if not prefix_exact:
            raise ValueError("candidate and parent must start from same actual prefix")
        results[case] = dict(
            source_hash="sha256:" + expected_hash, shared_prefix_exact=True, arms=arms
        )
    write_once(
        args.output / "complete.json",
        dict(
            status="STATIC_HOLDOUT_HASH_FORCE_GEOMETRY_READBACK_NOT_EXECUTION",
            records=results,
            independent_native_steps=0,
            reference_geometry_only=True,
            task_promoted=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
