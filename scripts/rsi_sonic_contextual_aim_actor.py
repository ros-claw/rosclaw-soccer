"""Learn a bounded contextual contact actor from prior SIM_ONLY physical rollouts.

Only TRAIN reports may select actions. Reserved physical courses are declared
before action selection and opened once after the actor JSON is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from rsi_sonic_ball_contact_probe import run

from rosclaw_soccer.rsi.sonic_contact_selector import choose_lateral, load_selector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

TRAIN_X = (2.12, 2.15, 2.18)
TRAIN_Y = (0.06, 0.10, 0.14, 0.18)
RESERVED = ((2.19, 0.08), (2.19, 0.12), (2.19, 0.16))
PARENT = (0.48, 0.18, 0.0, 0.0)
SCHEMA = "rosclaw_soccer.rsi.sonic_contextual_aim_actor.v1"


def _read_verified_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    stored = report.pop("report_hash")
    if hash_json(report) != stored:
        raise ValueError(f"report integrity mismatch: {path}")
    trace = path.with_name("trajectory.npz")
    if hash_bytes(trace.read_bytes()) != report["trajectory_hash"]:
        raise ValueError(f"trajectory integrity mismatch: {path}")
    report["report_hash"] = stored
    return report


def _train_score(report: dict[str, Any]) -> float:
    if (
        report["minimum_pelvis_height_m"] < 0.62
        or report["peak_pelvis_tilt_rad"] > 0.32
        or report["actuator_saturation_fraction"] > 0.01
    ):
        return -10.0
    if not report["first_robot_ball_contact_is_foot"]:
        return -2.0
    crossing = report["goal_crossing_ball_center_xyz_m"]
    if crossing is None:
        forward = min(max(float(report["final_ball_displacement_xyz_m"][0]), 0.0), 2.9)
        lateral = abs(
            float(report["ball_initial_xy_m"][1])
            + float(report["final_ball_displacement_xyz_m"][1])
        )
        return -1.0 + 1.25 * forward - 1.5 * min(lateral, 3.0)
    return (
        10.0 + 0.3 * min(float(report["peak_ball_speed_mps"]), 6.0) - 5.0 * abs(float(crossing[1]))
    )


def _learn_anchors(train_root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], dict[float, dict[str, Any]]] = {}
    for path in train_root.glob("train-*/report.json"):
        report = _read_verified_report(path)
        x, y = (float(value) for value in report["ball_initial_xy_m"])
        if x not in TRAIN_X or y not in TRAIN_Y or report["partition"] != "DISCOVERY":
            raise ValueError("unexpected training course or partition")
        candidate = path.parent.name.split("-x", 1)[0].removeprefix("train-")
        key = (y, candidate)
        if x in grouped.setdefault(key, {}):
            raise ValueError("duplicate physical training course")
        grouped[key][x] = report
    if not grouped or any(set(courses) != set(TRAIN_X) for courses in grouped.values()):
        raise ValueError("incomplete train candidate/course matrix")
    anchors = []
    for y in TRAIN_Y:
        ranked = []
        for (course_y, candidate), courses in grouped.items():
            if course_y != y:
                continue
            reports = [courses[x] for x in TRAIN_X]
            scores = [_train_score(report) for report in reports]
            first = reports[0]
            action = (
                float(first["contact_envelope_relative_x_center_m"]),
                float(first["contact_envelope_relative_x_sigma_m"]),
                float(first["lateral_feedback_gain_s_inv"]),
                float(first["right_contact_residual_rad"][2]),
            )
            if any(
                action
                != (
                    float(report["contact_envelope_relative_x_center_m"]),
                    float(report["contact_envelope_relative_x_sigma_m"]),
                    float(report["lateral_feedback_gain_s_inv"]),
                    float(report["right_contact_residual_rad"][2]),
                )
                for report in reports[1:]
            ):
                raise ValueError("candidate action drifted across training x positions")
            ranked.append(
                (
                    float(np.mean(scores) + 0.5 * min(scores)),
                    candidate,
                    action,
                    tuple(report["report_hash"] for report in reports),
                )
            )
        if not ranked:
            raise ValueError("missing training anchor")
        fitness, candidate, action, evidence = max(ranked, key=lambda row: (row[0], row[1]))
        anchors.append(
            {
                "ball_y_m": y,
                "candidate": candidate,
                "action": list(action),
                "train_fitness": fitness,
                "training_report_hashes": evidence,
            }
        )
    return anchors


def _action(anchors: list[dict[str, Any]], ball_y_m: float) -> tuple[float, float, float, float]:
    if not math.isfinite(ball_y_m) or not 0.06 <= ball_y_m <= 0.18:
        raise ValueError("actor requires a covered ball-y observation")
    ys = [float(anchor["ball_y_m"]) for anchor in anchors]
    if ys != list(TRAIN_Y):
        raise ValueError("unexpected contextual anchor order")
    action = tuple(
        float(np.interp(ball_y_m, ys, [float(anchor["action"][index]) for anchor in anchors]))
        for index in range(4)
    )
    center, sigma, gain, ankle = action
    if not (
        0.35 <= center <= 0.65
        and 0.08 <= sigma <= 0.25
        and 0.0 <= gain <= 1.5
        and abs(ankle) <= 0.15
    ):
        raise ValueError("learned action leaves the SIM_ONLY safety envelope")
    return (action[0], action[1], action[2], action[3])


def _rollout(
    *,
    actor: str,
    action: tuple[float, float, float, float],
    x: float,
    y: float,
    repeat: int,
    output_root: Path,
    model_root: Path,
    stadium_assets: Path,
    selector: dict[str, Any],
) -> dict[str, Any]:
    center, sigma, gain, ankle = action
    report = run(
        model_root=model_root,
        stadium_assets=stadium_assets,
        output_dir=output_root
        / f"reserved-{actor}-x{round(x * 1000)}-y{round(y * 1000)}-r{repeat}",
        ball_x_m=x,
        ball_y_m=y,
        frames=300,
        run_speed_mps=1.4,
        run_lateral_mps=choose_lateral(selector, ball_x_m=x, ball_y_m=y),
        stop_frame=120,
        left_hip_residual_rad=-0.15,
        left_knee_residual_rad=0.15,
        right_hip_residual_rad=-0.25,
        right_knee_residual_rad=-0.25,
        right_ankle_residual_rad=ankle,
        contact_envelope_center_m=center,
        contact_envelope_sigma_m=sigma,
        lateral_feedback_gain_s_inv=gain,
        partition="FRESH",
        selector_hash=selector["model_hash"],
    )
    crossing = report["goal_crossing_ball_center_xyz_m"]
    return {
        "actor": actor,
        "course": [x, y],
        "repeat": repeat,
        "action": action,
        "safe": bool(
            report["minimum_pelvis_height_m"] >= 0.62
            and report["peak_pelvis_tilt_rad"] <= 0.32
            and report["actuator_saturation_fraction"] <= 0.01
        ),
        "foot_first": report["first_robot_ball_contact_is_foot"],
        "goal": report["whole_ball_goal_crossed"],
        "goal_error_m": abs(float(crossing[1])) if crossing is not None else None,
        "peak_ball_speed_mps": report["peak_ball_speed_mps"],
        "trajectory_hash": report["trajectory_hash"],
        "report_hash": report["report_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aim-evidence", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    checkout = Path.cwd().resolve()
    if (
        output_root.exists()
        or output_root == checkout
        or checkout in output_root.parents
        or not args.aim_evidence.is_dir()
        or not args.model_root.is_dir()
        or not args.stadium_assets.is_dir()
    ):
        raise ValueError("new external output and qualified local inputs required")
    selector = load_selector(args.selector)
    source = Path(__file__)
    probe = source.with_name("rsi_sonic_ball_contact_probe.py")
    source_hash = hash_bytes(source.read_bytes())
    probe_hash = hash_bytes(probe.read_bytes())
    selector_hash = hash_bytes(args.selector.read_bytes())
    model_hashes = {
        name: hash_bytes((args.model_root / "low_latency" / name).read_bytes())
        for name in ("model_encoder.onnx", "model_decoder.onnx", "config.yaml")
    }
    protocol = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "source_hash": source_hash,
        "probe_hash": probe_hash,
        "selector_hash": selector_hash,
        "model_hashes": model_hashes,
        "train_evidence_root": str(args.aim_evidence.resolve()),
        "train_evidence_protocol_hash": hash_bytes(
            (args.aim_evidence / "protocol.json").read_bytes()
        ),
        "train_courses": [(x, y) for x in TRAIN_X for y in TRAIN_Y],
        "reserved_courses": RESERVED,
        "acceptance": (
            "candidate_3/3_safe_foot_goals; strict_replay; mean_goal_error_gain>=0.10m; "
            "candidate_max_error<=0.40m; speed_noninferiority>=-0.40mps"
        ),
    }
    output_root.mkdir(parents=True)
    protocol["protocol_hash"] = hash_json(protocol)
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    anchors = _learn_anchors(args.aim_evidence)
    actor = {
        "schema": SCHEMA,
        "protocol_hash": protocol["protocol_hash"],
        "anchors": anchors,
        "interpolation": "linear_in_ball_y_inside_training_support_only",
    }
    actor["actor_hash"] = hash_json(actor)
    (output_root / "actor.json").write_text(
        json.dumps(actor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    futures = []
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for x, y in RESERVED:
            for name, action in (("parent", PARENT), ("candidate", _action(anchors, y))):
                for repeat in (0, 1):
                    futures.append(
                        pool.submit(
                            _rollout,
                            actor=name,
                            action=action,
                            x=x,
                            y=y,
                            repeat=repeat,
                            output_root=output_root,
                            model_root=args.model_root,
                            stadium_assets=args.stadium_assets,
                            selector=selector,
                        )
                    )
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: (row["actor"], row["course"], row["repeat"]))

    def actor_rows(name: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["actor"] == name]

    def strict(name: str) -> bool:
        chosen = actor_rows(name)
        return all(
            next(row for row in chosen if row["course"] == [x, y] and row["repeat"] == 0)[
                "trajectory_hash"
            ]
            == next(row for row in chosen if row["course"] == [x, y] and row["repeat"] == 1)[
                "trajectory_hash"
            ]
            for x, y in RESERVED
        )

    parent_rows = actor_rows("parent")
    candidate_rows = actor_rows("candidate")
    parent_errors = [
        float(row["goal_error_m"]) for row in parent_rows if row["goal_error_m"] is not None
    ]
    candidate_errors = [
        float(row["goal_error_m"]) for row in candidate_rows if row["goal_error_m"] is not None
    ]
    parent_mean = float(np.mean(parent_errors)) if len(parent_errors) == 6 else None
    candidate_mean = float(np.mean(candidate_errors)) if len(candidate_errors) == 6 else None
    candidate_max = max(candidate_errors) if len(candidate_errors) == 6 else None
    parent_speed = float(np.mean([row["peak_ball_speed_mps"] for row in parent_rows]))
    candidate_speed = float(np.mean([row["peak_ball_speed_mps"] for row in candidate_rows]))
    source_stable = bool(
        hash_bytes(source.read_bytes()) == source_hash
        and hash_bytes(probe.read_bytes()) == probe_hash
        and hash_bytes(args.selector.read_bytes()) == selector_hash
        and all(
            hash_bytes((args.model_root / "low_latency" / name).read_bytes()) == digest
            for name, digest in model_hashes.items()
        )
    )
    passed = bool(
        all(row["safe"] and row["foot_first"] and row["goal"] for row in candidate_rows)
        and parent_mean is not None
        and candidate_mean is not None
        and candidate_max is not None
        and parent_mean - candidate_mean >= 0.10
        and candidate_max <= 0.40
        and candidate_speed - parent_speed >= -0.40
        and strict("parent")
        and strict("candidate")
        and source_stable
    )
    result = {
        "schema": SCHEMA,
        "protocol_hash": protocol["protocol_hash"],
        "actor_hash": actor["actor_hash"],
        "reserved_physical_episodes": len(rows),
        "parent_foot_goals": sum(
            row["safe"] and row["foot_first"] and row["goal"] for row in parent_rows[::2]
        ),
        "candidate_foot_goals": sum(
            row["safe"] and row["foot_first"] and row["goal"] for row in candidate_rows[::2]
        ),
        "parent_mean_goal_error_m": parent_mean,
        "candidate_mean_goal_error_m": candidate_mean,
        "candidate_max_goal_error_m": candidate_max,
        "parent_speed_mps": parent_speed,
        "candidate_speed_mps": candidate_speed,
        "parent_strict_replay": strict("parent"),
        "candidate_strict_replay": strict("candidate"),
        "source_stable_during_run": source_stable,
        "passed": passed,
        "promotion_status": "FROZEN_LOCAL_CONTACT_CANDIDATE" if passed else "REJECTED_DEVELOPMENT",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "no_team_claim": True,
    }
    result["report_hash"] = hash_json(result)
    (output_root / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
