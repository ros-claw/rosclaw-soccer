"""Seal a local one-G1 live-SONIC effect-fidelity gate, never a policy promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _read(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    commitment = value.pop("report_hash")
    if (
        commitment != hash_json(value)
        or value["schema"] != schema
        or value["activation_ceiling"] != "SIM_ONLY"
        or value["promotion_authorized"] is not False
    ):
        raise ValueError("sealed unpromoted SIM_ONLY proxy evidence required")
    return value, commitment


def assess(
    *,
    captured: Path,
    zero: Path,
    hip: Path,
    full_hip: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.parents[1]):
        raise ValueError("new external SIM_ONLY proxy gate directory required")
    tape, tape_hash = _read(
        captured / "report.json", "rosclaw_soccer.rsi.receiving_live_motor_capture.v1"
    )
    base, base_hash = _read(
        zero / "report.json", "rosclaw_soccer.rsi.receiving_single_live_sonic_proxy.v1"
    )
    candidate, candidate_hash = _read(
        hip / "report.json", "rosclaw_soccer.rsi.receiving_single_live_sonic_proxy.v1"
    )
    original, original_hash = _read(
        full_hip / "report.json", "rosclaw_soccer.rsi.receiving_team_full_hip_probe.v1"
    )
    tape_path = captured / "live-motor-tape.npz"
    zero_path = zero / "single-live.npz"
    hip_path = hip / "single-live.npz"
    full_path = full_hip / "full-probe.npz"
    if (
        tape["trace_hash"] != hash_bytes(tape_path.read_bytes())
        or base["trajectory_hash"] != hash_bytes(zero_path.read_bytes())
        or candidate["trajectory_hash"] != hash_bytes(hip_path.read_bytes())
        or original["full_trace_hash"] != hash_bytes(full_path.read_bytes())
        or base["capture_report_hash"] != tape_hash
        or candidate["capture_report_hash"] != tape_hash
        or original["student_model_hash"] != candidate["student_model_hash"]
        or base["student_model_hash"] != candidate["student_model_hash"]
        or base["source_hashes"] != candidate["source_hashes"]
        or base["left_hip_roll_offset_rad"] != 0.0
        or candidate["left_hip_roll_offset_rad"] != -0.08
        or original["left_hip_roll_offset_rad"] != -0.08
        or tape["physical_prefix_exact"] is not True
    ):
        raise ValueError("matched sealed baseline and hip-intervention proxies required")
    with (
        np.load(tape_path, allow_pickle=False) as full_base,
        np.load(zero_path, allow_pickle=False) as proxy_base,
        np.load(hip_path, allow_pickle=False) as proxy_hip,
        np.load(full_path, allow_pickle=False) as full_candidate,
    ):
        base_speed = float(np.linalg.norm(full_base["ball_velocity"][86, :2]))
        proxy_base_speed = float(np.linalg.norm(proxy_base["ball_velocity"][86, :2]))
        full_speed = float(np.linalg.norm(full_candidate["ball_velocity"][86, :2]))
        proxy_speed = float(np.linalg.norm(proxy_hip["ball_velocity"][86, :2]))
        full_distance = float(
            np.linalg.norm(
                full_candidate["ball_pose"][86, :2]
                - full_candidate["red_finisher_pelvis_pose"][86, :2]
            )
        )
        proxy_distance = float(
            np.linalg.norm(proxy_hip["ball_pose"][86, :2] - proxy_hip["pelvis_pose"][86, :2])
        )
        ball_position_max_error = float(
            np.max(np.abs(proxy_hip["ball_pose"] - full_candidate["ball_pose"][:130]))
        )
        ball_velocity_max_error = float(
            np.max(np.abs(proxy_hip["ball_velocity"] - full_candidate["ball_velocity"][:130]))
        )
        pelvis_max_error = float(
            np.max(
                np.abs(proxy_hip["pelvis_pose"] - full_candidate["red_finisher_pelvis_pose"][:130])
            )
        )
        full_foot = np.flatnonzero(
            (full_candidate["ball_contact_agent_code"][:130] == 6)
            & (full_candidate["ball_contact_foot_code"][:130] > 0)
        ).tolist()
        full_nonfoot = np.flatnonzero(
            full_candidate["ball_nonfoot_contact_agent_code"][:130] == 6
        ).tolist()
    proxy_effect = proxy_speed - proxy_base_speed
    full_effect = full_speed - base_speed
    passed = bool(
        base["maximum_foundation_target_error_rad"] == 0.0
        and base["maximum_pre_qpos_error"] < 1e-9
        and base["maximum_pre_qvel_error"] < 1e-9
        and abs(proxy_base_speed - base_speed) < 1e-9
        and abs(proxy_speed - full_speed) < 1e-6
        and abs(proxy_distance - full_distance) < 0.001
        and ball_position_max_error < 1e-6
        and ball_velocity_max_error < 1e-6
        and pelvis_max_error < 0.03
        and proxy_effect * full_effect > 0
        and abs(proxy_effect - full_effect) < 1e-6
        and candidate["proxy_foot_frames"] == full_foot
        and candidate["proxy_nonfoot_frames"] == full_nonfoot == []
        and candidate["minimum_pelvis_height_m"] >= 0.65
        and candidate["maximum_tilt_rad"] < 0.30
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("proxy gate source changed during assessment")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_single_live_proxy_gate.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "capture_report_hash": tape_hash,
        "zero_report_hash": base_hash,
        "hip_report_hash": candidate_hash,
        "full_hip_report_hash": original_hash,
        "consumed_course_only": True,
        "intervention_hip_roll_rad": -0.08,
        "baseline_speed_error_mps": abs(proxy_base_speed - base_speed),
        "hip_speed_error_mps": abs(proxy_speed - full_speed),
        "hip_distance_error_m": abs(proxy_distance - full_distance),
        "hip_ball_position_max_error_m": ball_position_max_error,
        "hip_ball_velocity_max_error_mps": ball_velocity_max_error,
        "hip_pelvis_max_error": pelvis_max_error,
        "proxy_speed_effect_mps": proxy_effect,
        "full_speed_effect_mps": full_effect,
        "proxy_foot_frames": candidate["proxy_foot_frames"],
        "full_foot_frames": full_foot,
        "proxy_nonfoot_frames": candidate["proxy_nonfoot_frames"],
        "full_nonfoot_frames": full_nonfoot,
        "local_training_proxy_authorized": passed,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("captured", "zero", "hip", "full-hip", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    report = assess(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "local_training_proxy_authorized": report["local_training_proxy_authorized"],
                "hip_speed_error_mps": report["hip_speed_error_mps"],
                "hip_distance_error_m": report["hip_distance_error_m"],
                "proxy_speed_effect_mps": report["proxy_speed_effect_mps"],
                "full_speed_effect_mps": report["full_speed_effect_mps"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
