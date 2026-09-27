"""SIM_ONLY disjoint full-SONIC CPU exam of a frozen route and motor actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rsi_sonic_ball_contact_probe import run
from rsi_sonic_feedback_navigation_train import FRESH_COURSES, TRAIN_COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def examine(
    *,
    model_root: Path,
    stadium_assets: Path,
    actor_path: Path,
    selection_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    trainer = source.with_name("rsi_sonic_feedback_navigation_train.py")
    helper = source.with_name("rsi_sonic_ball_contact_probe.py")
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY holdout directory required")
    selection: dict[str, Any] = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_hash = selection.pop("result_hash", None)
    protocol: dict[str, Any] = json.loads(
        (selection_path.parent / "protocol.json").read_text(encoding="utf-8")
    )
    protocol_hash = protocol.pop("protocol_hash", None)
    actor_payload: dict[str, Any] = json.loads(actor_path.read_text(encoding="utf-8"))
    actor_hash = actor_payload.pop("result_hash", None)
    if (
        selection_hash != hash_json(selection)
        or protocol_hash != hash_json(protocol)
        or actor_hash != hash_json(actor_payload)
        or selection.get("protocol_hash") != protocol_hash
        or selection.get("actor_hash") != actor_hash
        or protocol.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or protocol.get("source_hash") != hash_bytes(trainer.read_bytes())
        or protocol.get("helper_hash") != hash_bytes(helper.read_bytes())
        or tuple(tuple(course) for course in protocol.get("train_courses", [])) != TRAIN_COURSES
        or tuple(tuple(course) for course in protocol.get("reserved_fresh_courses", []))
        != FRESH_COURSES
        or selection.get("fresh_holdout_evaluated") is not False
        or selection.get("promotion_authorized") is not False
    ):
        raise ValueError("training selection or source identity changed")
    lateral = float(selection["selected_lateral_m_s"])
    if lateral not in protocol["lateral_options_m_s"]:
        raise ValueError("selected route not in predeclared training options")
    output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, (ball_x, ball_y) in enumerate(FRESH_COURSES):
        reports: dict[str, dict[str, Any]] = {}
        for label, actor in (("parent", None), ("candidate", actor_path)):
            report = run(
                model_root=model_root,
                stadium_assets=stadium_assets,
                output_dir=output_dir / f"course-{index}-{label}",
                ball_x_m=ball_x,
                ball_y_m=ball_y,
                frames=300,
                run_speed_mps=1.4,
                run_lateral_mps=lateral,
                stop_frame=120,
                feedback_actor=actor,
                partition="FRESH",
                selector_hash=selection_hash,
            )
            success = bool(
                report["first_robot_ball_contact_is_foot"]
                and report["whole_ball_goal_crossed"]
                and report["minimum_pelvis_height_m"] >= 0.65
                and report["peak_pelvis_tilt_rad"] < 0.30
            )
            reports[label] = {
                "report_hash": report["report_hash"],
                "trajectory_hash": report["trajectory_hash"],
                "foot_first": report["first_robot_ball_contact_is_foot"],
                "goal": report["whole_ball_goal_crossed"],
                "minimum_pelvis_height_m": report["minimum_pelvis_height_m"],
                "peak_pelvis_tilt_rad": report["peak_pelvis_tilt_rad"],
                "goal_frame": report["goal_frame"],
                "success": success,
            }
        rows.append({"course": [ball_x, ball_y], **reports})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    parent_count = sum(bool(row["parent"]["success"]) for row in rows)
    candidate_count = sum(bool(row["candidate"]["success"]) for row in rows)
    local_passed = bool(
        candidate_count >= 3
        and candidate_count >= parent_count + 1
        and all(not row["parent"]["success"] or row["candidate"]["success"] for row in rows)
        and all(row["candidate"]["minimum_pelvis_height_m"] >= 0.65 for row in rows)
        and all(row["candidate"]["peak_pelvis_tilt_rad"] < 0.30 for row in rows)
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("CPU holdout source changed during physics")
    result: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.sonic_feedback_navigation_exam.v1",
        "activation_ceiling": "SIM_ONLY",
        "selection_hash": selection_hash,
        "actor_hash": actor_hash,
        "source_hash": source_hash,
        "selected_lateral_m_s": lateral,
        "parent_success_count": parent_count,
        "candidate_success_count": candidate_count,
        "local_gate_passed": local_passed,
        "promotion_authorized": False,
        "rows": rows,
    }
    result["report_hash"] = hash_json(result)
    (output_dir / "report.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--actor-path", required=True, type=Path)
    parser.add_argument("--selection-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    result = examine(**vars(parser.parse_args()))
    print(
        json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
