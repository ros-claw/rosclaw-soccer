"""Closed full NN/native audit plus root-reference information-channel diagnosis."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("launch", "worker-source", "original", "calibration", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "expected-launch-hash"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or digest(a.launch) != a.expected_launch_hash
    ):
        raise ValueError("clean frozen read-only review and exact native launch required")
    launch = json.loads(a.launch.read_text())
    if (
        len(launch["jobs"]) != 3
        or len({j["unit"] for j in launch["jobs"]}) != 3
        or launch["maximum_native_steps"] != 24000
        or Path(launch["source"]) != a.worker_source
        or subprocess.check_output(
            ["git", "-C", str(a.worker_source), "rev-parse", "HEAD"], text=True
        ).strip()
        != launch["source_head"]
        or subprocess.check_output(["git", "-C", str(a.worker_source), "status", "--porcelain"])
    ):
        raise ValueError("exact original native worker source required")
    for folder in ("src", "scripts"):
        for f in (a.worker_source / folder).rglob("*.py"):
            if digest(f) != digest(Path.cwd() / f.relative_to(a.worker_source)):
                raise ValueError("review cannot alter executing native/planner/control math")
    pins = {str(a.launch): a.expected_launch_hash}
    pins.update(
        {
            str(f): digest(f)
            for directory in (Path.cwd() / "src", Path.cwd() / "scripts")
            for f in directory.rglob("*.py")
        }
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            worker_source_head=launch["source_head"],
            maximum_new_native_steps=0,
            maximum_new_neural_calls=0,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    imports = []
    try:
        expected = (
            "ROOT_WAYPOINT_DEFAULT_AND_CALIBRATED_PARITY",
            "ROOT_WAYPOINT_CASE21_PRODUCED",
            "ROOT_WAYPOINT_CASE21_REPRODUCED",
        )
        for j, status in zip(launch["jobs"], expected, strict=True):
            wait_for_terminal_collection(j["unit"], wait=True)
            root = Path(j["root"])
            d = json.loads((root / "declaration.json").read_text())
            c = json.loads((root / "complete.json").read_text())
            if (
                c["status"] != status
                or c["source_head"] != launch["source_head"]
                or d["source_head"] != launch["source_head"]
            ):
                raise ValueError("actual terminal0 same-source scientific result required")
            for k, h in d["input_file_hashes"].items():
                if k in pins and pins[k] != h:
                    raise ValueError("conflicting immutable source/assets")
                pins[k] = h
            pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
            imports.append((root, d, c))
        for f in (a.original / "complete.json", a.calibration / "complete.json"):
            if pins.get(str(f)) != digest(f):
                raise ValueError("exact original/failed calibrated comparator required")
        check_pins(pins)
        import mujoco
        import numpy as np

        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from rosclaw_soccer.rsi.root_waypoint_recorded_review import (
            encoder_root_channel_audit,
            review_waypoint_events,
        )
        from scripts.rsi_motor_student_rollout import exact_array
        from scripts.rsi_pilot_motion_entry import compare_episode
        from scripts.rsi_probe_compact_source import load_arrays
        from scripts.rsi_review_motion_entry_pilot import review_episode

        paths = [Path(k) for k in pins if k.endswith(".mjb")]
        if len(paths) != 1:
            raise ValueError("one unchanged compiled world required")
        world = mujoco.MjModel.from_binary_path(str(paths[0]))
        limits = world.jnt_range[[world.joint(n).id for n in G1_DDS_JOINT_NAMES]].copy()
        (qual, _, qc), (root, _, c), (repeat, _, rc) = imports
        if (
            qc["native_steps"] != 6000
            or len(qc["results"]) != 2
            or c["native_steps"] != 9000
            or rc["native_steps"] != 9000
        ):
            raise ValueError("whole declared 24000 actual native steps required")
        for tag, expected_root, row in zip(
            ("original", "calibrated"),
            (a.original, a.calibration / "produce"),
            qc["results"],
            strict=True,
        ):
            folder = qual / tag / "right-law0"
            compare_episode(folder, expected_root / "right-law0", row)
            review_episode(world, load_arrays(folder / "native300.npz"), row)
        if len(c["results"]) != 3 or len(rc["results"]) != 3:
            raise ValueError("all fixed candidate controls and actual repeats required")
        rows = []
        initial = None
        channel = None
        for law, row, rerow in zip((0, 3, 4), c["results"], rc["results"], strict=True):
            folder = f"right-law{law}"
            if (
                row != json.loads((root / folder / "complete.json").read_text())
                or rerow != json.loads((repeat / folder / "complete.json").read_text())
                or row["law_index"] != law
            ):
                raise ValueError("whole actual episode identity/order required")
            compare_episode(repeat / folder, root / folder, rerow)
            raw = load_arrays(root / folder / "native300.npz")
            prefix = load_arrays(root / folder / "own-prefix20.npz")
            for k, v in raw.items():
                exact_array(v[:20], prefix[k])
            if initial is None:
                initial = (raw["canonical_qpos"][0].copy(), raw["canonical_qvel"][0].copy())
            for k, value in zip(("canonical_qpos", "canonical_qvel"), initial, strict=True):
                exact_array(raw[k][0], value)
            primitive = review_episode(world, raw, row)
            event_file = root / folder / "root-waypoint-planner-events.json"
            if row["root_waypoint_planner_events_hash"] != digest(event_file):
                raise ValueError("actual NN waypoint input/output event commitment differs")
            events = json.loads(event_file.read_text())
            control = review_waypoint_events(raw, row, events, limits)
            nearest = 30 + int(np.argmin(raw["entry_error_m"][30:91]))
            rows.append(
                dict(
                    **primitive,
                    waypoint_control_review=control,
                    minimum_navigation_error_m=float(raw["entry_error_m"][nearest]),
                    closest_frame=nearest,
                    root_xy_at90_m=raw["canonical_qpos"][90, :2].tolist(),
                    reference_root_xy_at90_m=raw["encoder_reference_poses"][90, 0, :2].tolist(),
                )
            )
            if law == 0:
                channel = encoder_root_channel_audit(raw)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="ROOT_WAYPOINT_NATIVE24K_AND_INPUT_CHANNEL_AUDITED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                reviewed_native_steps=24000,
                rows=rows,
                encoder_root_channel_audit=channel,
                qualified_source_motion_episodes=sum(
                    r["qualified"] and r["first_touch_during_source_tracking"] for r in rows
                ),
                independent_world_contexts=1,
                new_native_steps=0,
                new_neural_calls=0,
                optimizer_steps=0,
                physics_improvement_proven=any(
                    r["qualified"] and r["first_touch_during_source_tracking"] for r in rows
                ),
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
                new_native_steps=0,
                new_neural_calls=0,
                automatic_physics_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
