"""Same four consumed courses, paired nominal vs bounded own-neural entry MPC.

All actual bodies execute H300. Prospective branches never supply actual
targets or states. Candidate artifacts include unsuccessful/unsafe branches.
"""

import argparse
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
        "forecast-qualification",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-prior-review-hash", "expected-qualification-hash"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed source and unused output required")
    if digest(args.prior_review / "complete.json") != args.expected_prior_review_hash:
        raise ValueError("independent historical physics proof differs")
    if digest(args.forecast_qualification / "complete.json") != args.expected_qualification_hash:
        raise ValueError("own-neural fork qualification differs")
    prior = json.loads((args.prior / "declaration.json").read_text())
    closed = json.loads((args.prior / "complete.json").read_text())
    review = json.loads((args.prior_review / "complete.json").read_text())
    if (
        prior["schema"] != "soccer.rsi.measured_ball_entry_teacher4.v1"
        or closed["actual_native_steps"] != 24000
        or review["additional_native_steps"] != 24000
        or len(review["results"]) != 8
        or any(any(x != 0 for x in row["maximum_errors"].values()) for row in review["results"])
        or (args.prior_review / "failure.json").exists()
    ):
        raise ValueError("whole historical eight-body evidence required")
    for p, h in prior["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("historical inputs changed")
    for path in (args.world, args.motion_csv):
        if prior["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("world/source differs from historical controls")
    for row in closed["results"]:
        if digest(args.prior / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("historical raw trajectory identity differs")
    paths = [
        p
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [
        p
        for root in (args.prior, args.prior_review, args.forecast_qualification)
        for p in root.iterdir()
        if p.is_file()
    ]
    paths += [
        args.world,
        args.motion_csv,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}

    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_episode import run_ball_entry_episode
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.predictive_ball_entry import PredictiveBallEntry
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.predictive_ball_entry4.v1",
        source_head=head,
        input_file_hashes=pins,
        courses=prior["courses"],
        reused_courses=4,
        independent_new_courses=0,
        consumed_development=True,
        arms=["nominal", "predictive"],
        actual_bodies=8,
        actual_horizon=300,
        planned_actual_native_steps=24000,
        maximum_forecast_native_steps=38400,
        source_interval=[286, 767],
        reference_entry_intervals=50,
        planning_frames=[20, 40, 60, 80],
        candidates_per_plan=6,
        forecast_control_frames=40,
        command_hold_control_frames=20,
        latest_handoff=90,
        entry_position_tolerance_m=0.04,
        native_nonfoot_force_rejection_n=1.0,
        native_minimum_pelvis_z_m=0.65,
        native_minimum_upright=0.9,
        score="ready:.01*offset/40+.02*speed2; else:1+10*min_error2+5*end_error2+.02*speed2",
        score_prefix="navigation only, stops before first predicted handoff",
        all_unsafe_fallback="zero_NOT_a_safety_certificate",
        actual_own_neural_targets_recomputed=True,
        no_forecast_target_or_state_adoption=True,
        old_future_is_comparator_only_not_actor_input=True,
        source_motion_is_frozen_contact_reference=True,
        optimizer_steps=0,
        actor_updated=False,
        official_football_exam=False,
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
        publicity_license_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
    try:
        stored = json.loads((args.prior / "entry-affordance.json").read_text())
        fields = {
            key: stored[key]
            for key in (
                "ball_offset_xy_m",
                "historical_frame",
                "historical_forward_speed_mps",
                "historical_artifact_hash",
            )
        }
        fields["ball_offset_xy_m"] = tuple(fields["ball_offset_xy_m"])
        affordance = SkillEntryAffordance(**fields)
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("historical affordance contract differs")
        write_once(args.output / "entry-affordance.json", stored)
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        model = mujoco.MjModel.from_binary_path(str(args.world))
        results, forecast_steps = [], 0

        def recorder_for(directory, manifest):
            def record(frame, index, forecast, row):
                name = f"frame{frame:03d}-candidate{index}"
                artifact = directory / (name + ".npz")
                np.savez_compressed(
                    artifact,
                    **forecast.arrays,
                    forecast_command=forecast.commands,
                    neural_action_isaac=forecast.neural_actions,
                    encoder_reference_poses=forecast.reference_poses,
                )
                receipt = dict(
                    **row,
                    frame=frame,
                    artifact_hash=digest(artifact),
                    new_attempted_native_steps=forecast.new_attempted_native_steps,
                    new_integrated_native_steps=forecast.new_integrated_native_steps,
                    new_planner_calls=forecast.new_planner_calls,
                )
                receipt_path = directory / (name + ".json")
                write_once(receipt_path, receipt)
                manifest.append(
                    dict(
                        frame=frame,
                        index=index,
                        artifact=str(artifact.relative_to(args.output)),
                        artifact_hash=digest(artifact),
                        receipt=str(receipt_path.relative_to(args.output)),
                        receipt_hash=digest(receipt_path),
                    )
                )

            return record

        for i, course in enumerate(declaration["courses"]):
            for arm in declaration["arms"]:
                name = f"course{i}-{arm}"
                directory = args.output / name
                directory.mkdir()
                manifest = []
                planner = (
                    PredictiveBallEntry(
                        world_hash=digest(args.world), recorder=recorder_for(directory, manifest)
                    )
                    if arm == "predictive"
                    else None
                )
                arrays, report = run_ball_entry_episode(
                    args.model_root,
                    args.world,
                    world_hash=digest(args.world),
                    source_poses=poses,
                    source_evidence_hash=hash_json(declaration),
                    source_interval=(286, 767),
                    affordance=affordance,
                    course=tuple(course),
                    arm="ball_entry",
                    entry_planner=planner,
                )
                if arm == "nominal":
                    with np.load(
                        args.prior / f"course{i}-ball_entry.npz", allow_pickle=False
                    ) as gold:
                        if set(gold.files) != set(arrays):
                            raise ValueError("historical nominal array coverage differs")
                        for key in gold.files:
                            np.testing.assert_array_equal(arrays[key], gold[key])
                    report["all_historical_nominal_arrays_bit_exact"] = True
                # An independently integrated branch must equal the actual actor
                # until the next command change or real handoff, never beyond it.
                prefix_checks = []
                for receipt in report["entry_planner_receipts"]:
                    start = receipt["frame"]
                    stop = min(start + 20, report["handoff_frame"] or 300)
                    chosen = next(
                        m
                        for m in manifest
                        if m["frame"] == start and m["index"] == receipt["selected_index"]
                    )
                    with np.load(args.output / chosen["artifact"], allow_pickle=False) as branch:
                        for key in branch.files:
                            actual_key = "entry_command" if key == "forecast_command" else key
                            np.testing.assert_array_equal(
                                arrays[actual_key][start:stop], branch[key][: stop - start]
                            )
                    prefix_checks.append(
                        dict(
                            start=start,
                            stop_exclusive=stop,
                            all_arrays_bit_exact=True,
                        )
                    )
                artifact = args.output / (name + ".npz")
                np.savez_compressed(artifact, **arrays)
                native = planner.forecaster.integrated_native_steps if planner else 0
                attempted = planner.forecaster.attempted_native_steps if planner else 0
                if native != attempted or native != sum(
                    r["new_forecast_native_steps"] for r in report["entry_planner_receipts"]
                ):
                    raise ValueError("forecast accounting differs")
                forecast_steps += native
                report.update(
                    job=name,
                    artifact_hash=digest(artifact),
                    candidate_artifacts=manifest,
                    selected_prefix_checks=prefix_checks,
                    actual_native_steps=3000,
                    forecast_native_steps=native,
                    forecast_attempted_native_steps=attempted,
                    strike=summarize_reference_ball_strike(model, arrays),
                )
                results.append(report)
                write_once(args.output / (name + "-complete.json"), report)
                print(
                    "PREDICTIVE_ENTRY_CLOSED",
                    name,
                    "handoff",
                    report["handoff_frame"],
                    "forecast_native",
                    native,
                    "strike",
                    report["strike"],
                    flush=True,
                )
        for p, h in {**prior["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("pinned sources/assets/evidence changed during execution")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                actual_native_steps=24000,
                forecast_native_steps=forecast_steps,
                total_new_native_steps=24000 + forecast_steps,
                reused_courses=4,
                independent_new_courses=0,
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
