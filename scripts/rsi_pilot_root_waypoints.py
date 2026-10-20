"""Default parity then fixed-root-waypoint native case21 trial, no new training."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("qualify", "produce", "reproduce"), required=True)
    for k in ("probe", "original", "calibration", "inputs", "core-source", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "expected-probe-hash", "probe-unit"):
        p.add_argument("--" + k, required=True)
    for k in ("qualification", "producer"):
        p.add_argument("--" + k, type=Path)
        p.add_argument("--" + k + "-unit")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen waypoint native source required")
    wait_for_terminal_collection(a.probe_unit)
    if digest(a.probe / "complete.json") != a.expected_probe_hash:
        raise ValueError("exact eight-call waypoint input probe required")
    probe = json.loads((a.probe / "complete.json").read_text())
    if (
        probe["status"] != "FROZEN_PLANNER_WAYPOINT_INPUT_PROBED"
        or probe["new_neural_calls"] != 8
        or not probe["flag0_sham_exact"]
        or not probe["waypoint_flag_consumed"]
        or probe["goal_vs_opposite_projected_root_contrast_m"] <= 1e-4
        or not probe["all_outputs_within_original_joint_limits"]
    ):
        raise ValueError("valid repeated planner signal required; not yet dynamic qualification")
    pins = dict(probe["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.probe.rglob("*") if f.is_file()})
    for f in (
        a.inputs,
        a.original / "complete.json",
        a.original / "declaration.json",
        a.calibration / "complete.json",
        a.calibration / "declaration.json",
    ):
        if pins.get(str(f)) != digest(f):
            raise ValueError("same whole historical case/entry/source/assets required")
    if a.mode != "qualify":
        if a.qualification is None or not a.qualification_unit:
            raise ValueError("default numerical qualification required")
        wait_for_terminal_collection(a.qualification_unit, wait=True)
        q = json.loads((a.qualification / "complete.json").read_text())
        if (
            q["status"] != "ROOT_WAYPOINT_DEFAULT_AND_CALIBRATED_PARITY"
            or q["source_head"] != a.expected_head
            or q["probe_hash"] != a.expected_probe_hash
            or q["native_steps"] != 6000
        ):
            raise ValueError("same-source whole default and calibrated numerical parity required")
        pins.update({str(f): digest(f) for f in a.qualification.rglob("*") if f.is_file()})
    producer = None
    if a.mode == "reproduce":
        if a.producer is None or not a.producer_unit:
            raise ValueError("actual producer required for independent NN/native repeat")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        producer = json.loads((a.producer / "complete.json").read_text())
        if (
            producer["status"] != "ROOT_WAYPOINT_CASE21_PRODUCED"
            or producer["native_steps"] != 9000
            or producer["source_head"] != a.expected_head
        ):
            raise ValueError("whole same-source actual producer required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    check_pins(pins)
    spec = json.loads(a.inputs.read_text())
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    a.motion_csv = next(Path(k) for k in pins if k.endswith(".csv"))
    prior_path = next(Path(k) for k in pins if Path(k).name == "entry-affordance.json")
    runtime = {
        str(f): digest(f)
        for directory in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in directory.rglob("*.py")
    }
    for f in [a.world, a.motion_csv] + [
        a.model_root / n
        for n in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(f)) != digest(f):
            raise ValueError("same native world/frozen Foundation/CSV required")
        runtime[str(f)] = digest(f)
    pins.update(runtime)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.native_entry_calibration import NativeEntryCalibration
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_motion_entry_episode import run_episode
    from scripts.rsi_pilot_motion_entry import compare_episode
    from scripts.rsi_probe_compact_source import load_arrays
    from scripts.rsi_review_motion_entry_pilot import review_episode

    courses = categorical_motion_holdout_courses()
    preflight_receiving_courses(courses)
    course = courses[21]
    if course.course[2] != 0.0:
        raise ValueError("fixed root goal pilot is only this unchanged stationary-ball context")
    world = mujoco.MjModel.from_binary_path(str(a.world))
    poses = np.loadtxt(a.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
    saved = json.loads(prior_path.read_text())
    prior = SkillEntryAffordance(
        tuple(saved["ball_offset_xy_m"]),
        saved["historical_frame"],
        saved["historical_forward_speed_mps"],
        saved["historical_artifact_hash"],
    )
    if prior.contract_hash != saved["contract_hash"]:
        raise ValueError("historical parent entry contract differs")
    cd = json.loads((a.calibration / "declaration.json").read_text())
    fields = (
        "motion_configuration_hash",
        "parent_affordance_hash",
        "audit_hash",
        "sample_hashes",
        "mean_displacement_xy_m",
        "maximum_training_residual_m",
    )
    vals = {k: cd["fit"][k] for k in fields}
    for k in ("sample_hashes", "mean_displacement_xy_m"):
        vals[k] = tuple(vals[k])
    fit = NativeEntryCalibration(**vals)
    if fit.contract_hash != cd["fit_hash"]:
        raise ValueError("fixed historical fit must not be refitted or tuned")
    reference = hash_json(
        dict(
            probe_hash=a.expected_probe_hash,
            entry_fit_hash=fit.contract_hash,
            intervention="fixed_root_waypoint_until_frame90",
            source_window=[466, 767],
        )
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            mode=a.mode,
            input_file_hashes=pins,
            probe_hash=a.expected_probe_hash,
            entry_fit_hash=fit.contract_hash,
            actual_teacher_reference_evidence_hash=reference,
            maximum_native_steps=6000 if a.mode == "qualify" else 9000,
            optimizer_steps=0,
            independent_world_contexts=1,
            goal_deadline_frame=90,
            football_team_qualification_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        rows = []
        if a.mode == "qualify":
            for tag, root, calib, arm in (
                ("original", a.original, None, "original"),
                ("calibrated", a.calibration / "produce", fit, "compact_unadapted"),
            ):
                evidence = json.loads(
                    (
                        (a.original if calib is None else a.calibration) / "declaration.json"
                    ).read_text()
                )["effective_motion_reference_evidence_hash"]
                local = SimpleNamespace(**{**vars(a), "output": a.output / tag})
                row = run_episode(
                    local,
                    course,
                    world,
                    poses,
                    prior,
                    arm=arm,
                    side="right",
                    law_index=0,
                    evidence=evidence,
                    runtime=runtime,
                    entry_calibration=calib,
                )
                compare_episode(local.output / "right-law0", root / "right-law0", row)
                rows.append(row)
            status = "ROOT_WAYPOINT_DEFAULT_AND_CALIBRATED_PARITY"
            summary = []
        else:
            for law in (0, 3, 4):
                row = run_episode(
                    a,
                    course,
                    world,
                    poses,
                    prior,
                    arm="compact_unadapted",
                    side="right",
                    law_index=law,
                    evidence=reference,
                    runtime=runtime,
                    entry_calibration=fit,
                    waypoint_entry=True,
                )
                if producer is not None:
                    compare_episode(
                        a.output / f"right-law{law}", a.producer / f"right-law{law}", row
                    )
                rows.append(row)
            summary = [
                review_episode(
                    world, load_arrays(a.output / f"right-law{r['law_index']}" / "native300.npz"), r
                )
                for r in rows
            ]
            status = (
                "ROOT_WAYPOINT_CASE21_REPRODUCED" if producer else "ROOT_WAYPOINT_CASE21_PRODUCED"
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=status,
                source_head=a.expected_head,
                probe_hash=a.expected_probe_hash,
                entry_fit_hash=fit.contract_hash,
                results=rows,
                summary=summary,
                native_steps=6000 if a.mode == "qualify" else 9000,
                new_optimizer_steps=0,
                actual_from0_body_and_neural_inference=True,
                not_offline_best_of_three_actor=True,
                independent_world_contexts=1,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_episodes=len(rows),
                actual_partial_step_counts_in_episode_receipts=True,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
