"""Independent readback of all four fixed, preregistered SIM holdouts.

No new integration or independent execution: validate hashes, physical force
labels, common prefixes and frame-end geometry from retained private futures.
"""

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from rosclaw_soccer.rsi.sim_contact_bandit import (
    ACTION_NAMES,
    contact_features,
    select_contact_action,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

ROOT = Path(os.environ["ROSCLAW_SOCCER_EVIDENCE"]).resolve(strict=True)
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
BANDIT_CASES = {
    "yminus-vminus": "65d11cf4f99646c536cce5f70a53e7626858f287c2a9d62b7e704dd278b476d7",
    "yminus-vplus": "4c1a7fd772ec6de4a3e4d30dc2e99b2cf420943843a73c0d1702c66624779e6c",
    "yplus-vminus": "98a3a3644148f3b72780f1ea55e7ac4da5c8031b66d508a75a6d02fd3c5c3a94",
    "yplus-vplus": "b62e6864bab721d5e7410ae32cae8d0caf14c785483d8af0d3abd08280536532",
}
GRID_CASES = {
    "yminus-far-vminus": "6828be3be9c84c2168a3a2c39031cf11e74131dbfe04430715f56c93e6d4544b",
    "yminus-far-vplus": "2fb939db43e670b13016831ea0104daa968746575f9530311cdc5258e38755bb",
    "yminus-near-vminus": "362fd5a6232eac1c357293ca395e40d137725f08d9c99a194ac018d4a9ef2535",
    "yminus-near-vplus": "d19e7c0e92542b99c3af7bfebad2cc1b1fac221091d60e1d1360a3f89b9f13bb",
    "yplus-far-vminus": "264b95b8242b33bf44e390885aabec2edd3c2034fae4652c6826602f7efe24c2",
    "yplus-far-vplus": "ada436f85facfce0290bbdd478e1a6b1bacccb87e3f0de69e7f7603cb440d9be",
    "yplus-near-vminus": "ec56a6629b4dd2da9144034269b1ea11ec7fa6a5848d2a790d55da5c01846f43",
    "yplus-near-vplus": "e13ec24c084e35bbc13123476a02295f77578c6dc64fd5080e7b786f9490f58b",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--bandit", action="store_true")
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    if sum((args.bandit, args.training, args.grid)) > 1:
        raise ValueError("training, grid and bandit audit are different splits")
    selected_cases = (
        GRID_CASES
        if args.grid
        else BANDIT_CASES
        if args.bandit
        else TRAINING_CASES
        if args.training
        else CASES
    )
    if args.bandit:
        bandit_path = ROOT / "rsi-contact-bandit-training-20261027/complete.json"
        if (
            digest(bandit_path)
            != "sha256:541d80cd4b8f87753a60b1f0cbff0e56097e1b74dbff199005b49a954d4e921f"
        ):
            raise ValueError("exact actor training report required")
        bandit = json.loads(bandit_path.read_text())
        models = {}
        for row in bandit["records"]:
            model_path = bandit_path.parent / row["model_path"]
            if (
                model_path.is_symlink()
                or model_path.parent != bandit_path.parent
                or digest(model_path) != row["model_hash"]
            ):
                raise ValueError("exact audited actor seed required")
            models[row["seed"]] = json.loads(model_path.read_text())
        if set(models) != {17, 43, 71}:
            raise ValueError("all three seeds required")
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
            cases=selected_cases,
            bandit_split=args.bandit,
            grid_training_split=args.grid,
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
    for case, expected_hash in selected_cases.items():
        prefix = (
            "rsi-contact-grid"
            if args.grid
            else "rsi-bandit-eval"
            if args.bandit
            else "rsi-approach-train"
            if args.training or args.grid
            else "rsi-static-holdout"
        )
        root = ROOT / f"{prefix}-{case}-20261027"
        report_path = root / "complete.json"
        if report_path.is_symlink() or digest(report_path) != "sha256:" + expected_hash:
            raise ValueError(f"exact holdout report required: {case}")
        report = json.loads(report_path.read_text())
        expected_status = (
            "OFFLINE_BANDIT_PAIRED_PRIVATE_HOLDOUT_NOT_PROMOTED"
            if args.bandit
            else "CFTRACK_STATIC_APPROACH_TRAINING_DATA_NOT_PROMOTED"
            if args.training or args.grid
            else "STATIC_CONTACT_PAIRED_HOLDOUT_NOT_PROMOTED"
        )
        case_key = "training_case" if args.training or args.grid else "holdout_id"
        if report["status"] != expected_status or report[case_key] != case:
            raise ValueError("matching unpromoted holdout identity required")
        arms = {}
        traces = {}
        arm_names = (
            ["parent", "seed-17", "seed-43", "seed-71"]
            if args.bandit
            else ["parent", *(f"hg{i}-{side}" for i in range(4) for side in ("minus", "plus"))]
            if args.training or args.grid
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
            if args.bandit and arm != "parent":
                context = contact_features(
                    float(arrays["qpos"][1069, lo.bq + 1]),
                    float(arrays["qpos"][1069, lo.rq + 1]),
                    float(arrays["qvel"][1069, lo.bv + 1]),
                )
                seed = int(arm.split("-")[1])
                chosen = select_contact_action(
                    context,
                    np.asarray(models[seed]["actor_weight"]),
                    np.asarray(models[seed]["actor_bias"]),
                )
                if ACTION_NAMES[chosen] != record["selected_action"] or not np.array_equal(
                    context, np.asarray(report["features"], dtype=np.float32)
                ):
                    raise ValueError("actor decision or actual state features mismatch")
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
                "STATIC_BANDIT_HASH_ACTOR_FORCE_READBACK_NOT_EXECUTION"
                if args.bandit
                else "STATIC_GRID_HASH_FORCE_GEOMETRY_READBACK_NOT_EXECUTION"
                if args.grid
                else "STATIC_TRAINING_HASH_FORCE_GEOMETRY_READBACK_NOT_EXECUTION"
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
