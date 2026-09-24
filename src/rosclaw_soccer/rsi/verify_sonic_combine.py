"""Independent replay/metric check of a SONIC Athlete Combine evidence bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _receipt(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("manifest_hash", None)
    if claimed != hash_json(value):
        raise ValueError(f"receipt hash mismatch: {path}")
    return value


def _trace(path: Path, expected_hash: str) -> dict[str, NDArray[np.float64]]:
    if hash_bytes(path.read_bytes()) != expected_hash:
        raise ValueError(f"trajectory hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as stream:
        arrays = {key: stream[key].copy() for key in stream.files}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"trajectory is nonfinite: {path}")
    return arrays


def _metrics(arrays: dict[str, NDArray[np.float64]]) -> dict[str, float]:
    if (
        set(arrays) != {"qpos", "qvel", "target", "torque"}
        or arrays["qpos"].shape != (150, 36)
        or arrays["qvel"].shape != (150, 35)
        or arrays["target"].shape != (150, 29)
        or arrays["torque"].shape != (150, 29)
    ):
        raise ValueError("SONIC course trace has an unexpected tensor layout")
    q = arrays["qpos"]
    v = arrays["qvel"]
    w, x, y, z = q[:, 3:7].T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return {
        "minimum_pelvis_height_m": float(q[:, 2].min()),
        "maximum_absolute_roll_pitch_rad": float(max(np.abs(roll).max(), np.abs(pitch).max())),
        "displacement_xy_m": float(np.linalg.norm(q[-1, :2] - q[0, :2])),
        "absolute_yaw_change_rad": float(abs(yaw[-1] - yaw[0])),
        "final_speed_mps": float(np.linalg.norm(v[-10:, :2], axis=1).mean()),
    }


def verify_sonic_combine(evidence_dir: Path) -> dict[str, Any]:
    root = evidence_dir.expanduser().resolve()
    protocol = _receipt(root / "protocol.json")
    complete = _receipt(root / "complete.json")
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.sonic_combine_protocol.v1"
        or complete["schema"] != "rosclaw_soccer.rsi.sonic_combine_complete.v1"
        or protocol["partition"] != "DISCOVERY"
        or complete["partition"] != "DISCOVERY"
        or protocol["promotion_authorized"]
        or complete["promotion_authorized"]
        or protocol["source_hash"] != complete["source_hash"]
        or protocol["body_hash"] != complete["body_hash"]
        or protocol["repeats_per_course"] != 2
        or complete["physical_execution_count"] != 11
        or not complete["strict_replay"]
    ):
        raise ValueError("SONIC Combine protocol/completion boundary mismatch")
    courses = {item["name"]: item for item in protocol["courses"]}
    if len(courses) != 5 or len(complete["rows"]) != 5 or len(complete["replay_rows"]) != 5:
        raise ValueError("SONIC Combine has missing or duplicate courses")
    if set(courses) != {"stand", "walk", "run", "turn", "stop"}:
        raise ValueError("SONIC Combine course identity differs from declared battery")
    pass_count = 0
    run_row: dict[str, Any] | None = None
    for role, rows in (("primary", complete["rows"]), ("replay", complete["replay_rows"])):
        if {row["course"] for row in rows} != set(courses):
            raise ValueError("SONIC Combine execution rows are duplicated or omitted")
        for row in rows:
            name = row["course"]
            label = f"{name}-{role}"
            disk = _receipt(root / f"{label}.json")
            if (
                row != disk
                or row["execution_id"] != label
                or row["thresholds"] != courses[name]
                or row["partition"] != "DISCOVERY"
                or row["activation_ceiling"] != "SIM_ONLY"
                or row["teacher_active"]
                or row["physics_hash"] != protocol["body_hash"]
                or row["artifact_hash"] != hash_json(row["artifact"])
                or row["artifact"]["physics_hash"] != protocol["body_hash"]
                or row["artifact"]["joint_map_hash"] != protocol["joint_map_hash"]
            ):
                raise ValueError(f"{label}: execution or artifact commitment differs")
            arrays = _trace(root / f"{label}.npz", row["trajectory_hash"])
            measured = _metrics(arrays)
            if any(
                not math.isclose(row[key], value, rel_tol=0.0, abs_tol=1.0e-12)
                for key, value in measured.items()
            ):
                raise ValueError(f"{label}: physical metrics differ from trajectory")
            success = (
                measured["minimum_pelvis_height_m"] >= 0.55
                and measured["maximum_absolute_roll_pitch_rad"] <= 0.40
                and measured["displacement_xy_m"] >= courses[name]["minimum_displacement_m"]
                and measured["displacement_xy_m"] <= courses[name]["maximum_displacement_m"]
                and measured["absolute_yaw_change_rad"] >= courses[name]["minimum_yaw_rad"]
                and measured["final_speed_mps"] <= courses[name]["maximum_final_speed_mps"]
            )
            if row["task_success"] is not success:
                raise ValueError(f"{label}: success claim differs from physical trace")
            if role == "primary":
                pass_count += success
                if name == "run":
                    run_row = row
            else:
                primary_hash = next(
                    item["trajectory_hash"] for item in complete["rows"] if item["course"] == name
                )
                first = _trace(
                    root / f"{name}-primary.npz",
                    primary_hash,
                )
                if any(not np.array_equal(first[key], arrays[key]) for key in first):
                    raise ValueError(f"{name}: physical replay is not exact")
    if complete["pass_count"] != pass_count or run_row is None:
        raise ValueError("SONIC Combine pass count differs from verified courses")
    ablation = _receipt(root / "run-zero-torque.json")
    if ablation != complete["zero_torque_ablation"]:
        raise ValueError("zero-torque receipt differs from completion")
    zero = _trace(root / "run-zero-torque.npz", ablation["trajectory_hash"])
    q = zero.get("qpos")
    if q is None or q.shape != (1500, 36):
        raise ValueError("zero-torque trajectory shape differs")
    zero_height = float(q[:, 2].min())
    zero_displacement = float(np.linalg.norm(q[-1, :2] - q[0, :2]))
    if (
        ablation["initial_state_hash"] != run_row["initial_state_hash"]
        or not math.isclose(ablation["minimum_pelvis_height_m"], zero_height, abs_tol=1.0e-12)
        or not math.isclose(ablation["displacement_xy_m"], zero_displacement, abs_tol=1.0e-12)
    ):
        raise ValueError("zero-torque ablation lacks the same physical parent")
    causal_separation = (
        run_row["displacement_xy_m"] - zero_displacement >= 1.0 and zero_height < 0.55
    )
    if complete["causal_separation"] is not causal_separation:
        raise ValueError("causal-separation claim differs from physical evidence")
    return {
        "schema": "rosclaw_soccer.rsi.sonic_combine_verification.v1",
        "source_hash": protocol["source_hash"],
        "body_hash": protocol["body_hash"],
        "course_count": len(courses),
        "pass_count": pass_count,
        "physical_execution_count": complete["physical_execution_count"],
        "strict_replay": True,
        "causal_separation": causal_separation,
        "partition": "DISCOVERY",
        "promotion_authorized": False,
        "verification_hash": hash_json(
            {
                "protocol": hash_bytes((root / "protocol.json").read_bytes()),
                "complete": hash_bytes((root / "complete.json").read_bytes()),
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_sonic_combine(args.evidence_dir), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
