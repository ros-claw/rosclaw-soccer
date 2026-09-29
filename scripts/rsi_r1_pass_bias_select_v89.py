"""Select one consumed direct-physics pass-bias candidate without promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _row(directory: Path) -> dict[str, Any]:
    report_path = directory / "report.json"
    protocol_path = directory / "protocol.json"
    trace_path = directory / "trace.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_hash = report.pop("report_hash")
    if (
        hash_json(report) != report_hash
        or hash_bytes(protocol_path.read_bytes()) != report["protocol_hash"]
        or hash_bytes(trace_path.read_bytes()) != report["trace_hash"]
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
        or protocol["video_authorized"] is not False
    ):
        raise ValueError(f"integrity failure: {directory}")
    with np.load(trace_path, allow_pickle=False) as trace:
        chain = report["chain"] or {}
        receiver_sec = chain.get("receiver_contact_sec")
        shot_frames = (
            []
            if receiver_sec is None
            else [
                int(frame)
                for frame in np.flatnonzero(
                    (trace["ball_contact_agent_code"] == 4)
                    & np.isin(trace["ball_contact_foot_code"], (1, 2))
                )
                if float(trace["time"][frame]) >= float(receiver_sec) + 0.20
                and float(np.linalg.norm(trace["ball_velocity"][frame, :3])) >= 2.0
            ]
        )
        result = {
            "report_hash": report_hash,
            "protocol_hash": report["protocol_hash"],
            "trace_hash": report["trace_hash"],
            "safe": bool(report["result"]["safe"]),
            "clean_transfer": bool(chain.get("clean_transfer_observed")),
            "source_foot_contact_sec": chain.get("source_contact_sec"),
            "receiver_foot_contact_sec": receiver_sec,
            "later_finisher_foot_shot_count": len(shot_frames),
            "later_finisher_foot_shot_frames": shot_frames,
            "goal_crossed_inside": bool((report["crossing"] or {}).get("inside_geometry", False)),
            "minimum_finisher_joint_margin_rad": float(
                np.min(trace["red_finisher_joint_safety_margin_rad"])
            ),
            "nonfoot_ball_contact": bool(np.any(trace["ball_nonfoot_contact_agent_code"] != 0)),
        }
    return result


def select(evidence_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "rosclaw_soccer.rsi.r1_lateral_pass_learning_v89.protocol.v1":
        raise ValueError("frozen v89 protocol required")
    scenes = protocol["training_scenes"]
    biases = protocol["arms_lateral_bias_m"]
    if len(scenes) != 3 or biases != [-0.4, -0.2, 0.2, 0.4]:
        raise ValueError("frozen curriculum changed")
    baseline = {
        scene["id"]: _row(evidence_root / f"rsi-r1-pass-acquisition-v88-{scene['id']}-standoff0.35")
        for scene in scenes
    }
    baseline_shots = sum(row["later_finisher_foot_shot_count"] for row in baseline.values())
    arms = []
    for bias in biases:
        rows = {}
        for scene in scenes:
            scene_id = scene["id"]
            directory = evidence_root / f"rsi-r1-pass-bias-v89-{scene_id}-bias{bias}"
            run_protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
            position = run_protocol["scenario"]["ball_initial_position_m"]
            if (
                position[:2] != [scene["ball_x_m"], scene["ball_y_m"]]
                or run_protocol["scenario"]["seed"] != scene["seed"]
                or run_protocol["pass_lateral_bias_m"] != bias
                or run_protocol["precontact_pass_standoff_m"] != 0.35
                or run_protocol["stance_lateral_m"] != -0.19
                or run_protocol["handoff_profile"] != "tracking"
            ):
                raise ValueError("training run mismatches frozen scene or arm")
            rows[scene_id] = _row(directory)
        eligible = all(
            row["safe"] and row["clean_transfer"] and not row["nonfoot_ball_contact"]
            for row in rows.values()
        )
        shots = sum(row["later_finisher_foot_shot_count"] for row in rows.values())
        goals = sum(row["goal_crossed_inside"] for row in rows.values())
        margin = min(row["minimum_finisher_joint_margin_rad"] for row in rows.values())
        arms.append(
            {
                "bias_m": bias,
                "eligible": eligible,
                "shot_contacts": shots,
                "goal_count": goals,
                "minimum_finisher_joint_margin_rad": margin,
                "rows": rows,
            }
        )
    candidates = [arm for arm in arms if arm["eligible"] and arm["shot_contacts"] > baseline_shots]
    selected = (
        None
        if not candidates
        else max(
            candidates,
            key=lambda arm: (
                arm["shot_contacts"],
                arm["goal_count"],
                arm["minimum_finisher_joint_margin_rad"],
            ),
        )["bias_m"]
    )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_pass_bias_selection_v89.v1",
        "status": "SELECTED_CONSUMED_ONLY" if selected is not None else "REJECTED_TRAINING",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "baseline_shot_contacts": baseline_shots,
        "baseline": baseline,
        "arms": arms,
        "selected_lateral_bias_m": selected,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new immutable selection output required")
    result = select(args.evidence_root, args.protocol)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "baseline_shot_contacts",
                    "selected_lateral_bias_m",
                    "report_hash",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
