"""Eight fresh H300 bodies: fixed reference vs measured-ball entry teacher.

Past own-body motion estimates entry geometry. Ball positions remain exactly
the prior four failed courses; no source motion, body or ball is teleported.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prior",
        "prior-review",
        "motion-csv",
        "world",
        "model-root",
        "core-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-world-hash", "expected-prior-review-hash"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("pinned source and unused output required")
    if digest(args.prior_review / "complete.json") != args.expected_prior_review_hash:
        raise ValueError("verified prior identity differs")
    prior_declaration = json.loads((args.prior / "declaration.json").read_text())
    prior_complete = json.loads((args.prior / "complete.json").read_text())
    prior_admission = json.loads((args.prior / "admission.json").read_text())
    review = json.loads((args.prior_review / "complete.json").read_text())
    if (
        prior_complete["actual_native_steps"] != 27000
        or review["additional_native_steps"] != 27000
        or len(review["results"]) != 9
        or (args.prior_review / "failure.json").exists()
        or any(any(e != 0 for e in row["maximum_errors"].values()) for row in review["results"])
    ):
        raise ValueError("whole independent historical physics proof required")
    for p, h in prior_declaration["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("historical source/assets changed")
    for path in (args.world, args.motion_csv):
        if prior_declaration["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("current world/source differs from prior")
    paths = [
        p
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for root in (args.prior, args.prior_review) for p in root.iterdir() if p.is_file()]
    paths += [
        args.world,
        args.motion_csv,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}
    if digest(args.world) != args.expected_world_hash:
        raise ValueError("compiled world identity differs")
    for row in prior_complete["results"]:
        if digest(args.prior / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("historical raw trajectory hash differs")

    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.ball_entry_episode import run_ball_entry_episode
    from rosclaw_soccer.rsi.ball_entry_teacher import derive_entry_affordance
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    courses = [(0.35, -0.10, 0.0), (0.50, -0.10, 0.0), (0.35, -0.22, 0.0), (0.50, -0.22, 0.0)]
    declaration = dict(
        schema="soccer.rsi.measured_ball_entry_teacher4.v1",
        input_file_hashes=pins,
        source_head=head,
        courses=[list(c) for c in courses],
        consumed_development=True,
        independent_new_courses=0,
        reused_courses=4,
        arms=["fixed_reference", "ball_entry"],
        physical_jobs=8,
        planned_native_steps=24000,
        source_interval=[286, 767],
        source_frequency_hz=120,
        reference_entry_intervals=50,
        entry_maximum_velocity_mps=0.6,
        earliest_handoff=30,
        latest_handoff=90,
        entry_position_tolerance_m=0.04,
        body_horizon=300,
        teacher_is_learned=False,
        source_prior_is_from_past_own_physics_not_current_future=True,
        licensed_source_requires_separate_publicity_review=True,
        optimizer_steps=0,
        official_football_exam=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    try:
        expected = (
            "root_pos_x(m)",
            "root_pos_y(m)",
            "root_pos_z(m)",
            "root_rot_w",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
        )
        expected += tuple(f"dof_{n}(rad)" for n in G1_DDS_JOINT_NAMES)
        with args.motion_csv.open(newline="") as stream:
            if tuple(next(csv.reader(stream))) != expected:
                raise ValueError("exact canonical29 meter/radian/wxyz source required")
        source_poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        model = mujoco.MjModel.from_binary_path(str(args.world))
        with np.load(args.prior / "parked-quarantined.npz", allow_pickle=False) as archive:
            prior_q = archive["canonical_qpos"]
        affordance = derive_entry_affordance(
            model,
            prior_q,
            artifact_hash=digest(args.prior / "parked-quarantined.npz"),
            foot="right",
        )
        if (prior_admission["start_frame"], prior_admission["stop_frame_exclusive"]) != (286, 767):
            raise ValueError("historical motion phase differs")
        write_once(
            args.output / "entry-affordance.json",
            dict(
                **affordance.__dict__,
                contract_hash=affordance.contract_hash,
                source_historical_physics_independently_verified=True,
                contact_qualified=False,
                no_new_physical_steps=True,
                no_neural_training=True,
            ),
        )
        results = []
        for i, course in enumerate(courses):
            for arm in declaration["arms"]:
                arrays, report = run_ball_entry_episode(
                    args.model_root,
                    args.world,
                    world_hash=args.expected_world_hash,
                    source_poses=source_poses,
                    source_evidence_hash=hash_json(declaration),
                    source_interval=(286, 767),
                    affordance=affordance,
                    course=course,
                    arm=arm,
                )
                if arm == "fixed_reference":
                    with np.load(
                        args.prior / f"course{i}-quarantined.npz", allow_pickle=False
                    ) as gold:
                        for key in gold.files:
                            observed = (
                                arrays["encoder_reference_poses"][:, 0]
                                if key == "planned_reference"
                                else arrays[key]
                            )
                            np.testing.assert_array_equal(observed, gold[key])
                    report["all_historical_fixed_physics_and_neural_actions_bit_exact"] = True
                name = f"course{i}-{arm}"
                target = args.output / (name + ".npz")
                np.savez_compressed(target, **arrays)
                report.update(
                    job=name,
                    artifact_hash=digest(target),
                    strike=summarize_reference_ball_strike(model, arrays),
                )
                results.append(report)
                write_once(args.output / (name + "-complete.json"), report)
                print(
                    "MEASURED_BALL_ENTRY_CLOSED",
                    name,
                    "handoff",
                    report["handoff_frame"],
                    "entry_min",
                    report["minimum_entry_error_m"],
                    "strike",
                    report["strike"],
                    flush=True,
                )
        for p, h in {**prior_declaration["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("pinned sources/assets/evidence changed during experiment")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                actual_native_steps=24000,
                physical_jobs=8,
                reused_courses=4,
                optimizer_steps=0,
                official_football_exam_pass=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json", dict(error=repr(error), promotion_authorized=False)
        )
        raise


if __name__ == "__main__":
    main()
