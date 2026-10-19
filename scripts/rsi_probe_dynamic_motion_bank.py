"""Fixed lexicographic16 MotionDecode source bank, no outcome-based selection.

All selected dynamic windows are admitted before ANY native physics. Rejected
windows stay in the manifest and are never replaced by quieter source frames.
"""

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion-directory", "license", "world", "model-root", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-world-hash", required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("pinned source and unused output required")
    motions = sorted(args.motion_directory.glob("BGI_Shooting_*.csv"))[:16]
    if [p.name for p in motions] != [f"BGI_Shooting_{i:05}.csv" for i in range(1, 17)]:
        raise ValueError("exact first16 shooting clips required, no replacement")
    paths = [
        p
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += motions + [
        args.license,
        args.world,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}
    if digest(args.world) != args.expected_world_hash:
        raise ValueError("world hash mismatch")

    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_motion_window import select_dynamic_leg_window
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.reference_tracking_episode import run_reference_tracking_episode
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_verify_timed_reference import tracking_metrics

    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.dynamic_motion_bank16.v1",
        input_file_hashes=pins,
        source_head=head,
        source_names=[p.name for p in motions],
        selector="first_argmax12leg_angularspeed_centered4seconds_capped_to_source",
        source_frequency_hz=120,
        entry_intervals=50,
        course=[10.0, 0.0, 0.0],
        source_windows_not_goal_labels=True,
        optimizer_steps=0,
        maximum_native_steps=48000,
        maximum_rollouts=16,
        licensed_source_requires_separate_publicity_review=True,
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
        entry = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, G1SonicRunupController.default_angles]
        admitted = []
        admission = []
        for path in motions:
            try:
                with path.open(newline="") as stream:
                    if tuple(next(csv.reader(stream))) != expected:
                        raise ValueError("exact meter/wxyz/radian canonical29 columns required")
                values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
                window = select_dynamic_leg_window(
                    values, entry, source_evidence_hash=hash_json(declaration)
                )
            except ValueError as error:
                admission.append(
                    dict(source=path.name, status="REJECTED_SOURCE_WINDOW", error=str(error))
                )
                continue
            admitted.append((path, window))
            admission.append(
                dict(
                    source=path.name,
                    status="ADMITTED_GEOMETRY_ONLY",
                    start_frame=window.source_start_frame,
                    stop_frame_exclusive=window.source_stop_frame_exclusive,
                    peak_interval=window.source_peak_leg_interval,
                    peak_leg_speed_rad_per_sec=window.source_peak_leg_speed_rad_per_sec,
                    preparation_hash=window.prepared.contract_hash,
                    reference=window.prepared.reference.poses,
                )
            )
        write_once(
            args.output / "admission.json",
            dict(
                rows=admission,
                admitted_count=len(admitted),
                rejected_count=16 - len(admitted),
                declared_native_steps=len(admitted) * 3000,
                before_any_physics=True,
            ),
        )
        print("ADMISSION_FIXED", len(admitted), "rejected", 16 - len(admitted), flush=True)
        results = []
        for path, window in admitted:
            arrays, report = run_reference_tracking_episode(
                args.model_root,
                args.world,
                world_hash=args.expected_world_hash,
                reference=window.prepared.reference,
            )
            target = args.output / (path.stem + ".npz")
            np.savez_compressed(target, **arrays)
            report.update(
                source=path.name,
                artifact_hash=digest(target),
                tracking_metrics=tracking_metrics(
                    arrays["canonical_qpos"], arrays["planned_reference"], np
                ),
            )
            results.append(report)
            write_once(args.output / (path.stem + "-complete.json"), report)
            print(
                "NATIVE_DYNAMIC_WINDOW_CLOSED",
                path.name,
                report["minimum_native_pelvis_z_m"],
                flush=True,
            )
        if any(digest(Path(p)) != h for p, h in pins.items()):
            raise ValueError("sources/assets changed during physics")
        write_once(
            args.output / "complete.json",
            dict(
                result="DYNAMIC_SOURCE_TRACKING_DIAGNOSTIC_NOT_FOOTBALL_OR_TRAINING",
                admitted_count=len(admitted),
                rejected_count=16 - len(admitted),
                actual_native_steps=3000 * len(results),
                results=results,
                optimizer_steps=0,
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
