"""SIM_ONLY closed-loop SONIC navigation selection around a frozen motor actor.

Training uses only the declared development courses. The selector never opens
the disjoint full-episode holdout and never promotes the motor candidate.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from rsi_sonic_ball_contact_probe import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

TRAIN_COURSES = tuple(itertools.product((1.90, 2.10), (0.06, 0.14)))
FRESH_COURSES = tuple(itertools.product((1.95, 2.05), (0.08, 0.12)))
LATERAL_OPTIONS = (-0.04, 0.0, 0.04, 0.08, 0.12)


def train(
    *,
    model_root: Path,
    stadium_assets: Path,
    actor_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_sonic_ball_contact_probe.py")
    source_hash = hash_bytes(source.read_bytes())
    helper_hash = hash_bytes(helper.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY evidence directory required")
    payload: dict[str, Any] = json.loads(actor_path.read_text(encoding="utf-8"))
    actor_commitment = payload.pop("result_hash", None)
    if actor_commitment != hash_json(payload) or payload.get("promotion_authorized") is not False:
        raise ValueError("frozen unpromoted feedback actor required")
    output_dir.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.sonic_feedback_navigation_train.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "actor_hash": actor_commitment,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "train_courses": [list(course) for course in TRAIN_COURSES],
        "reserved_fresh_courses": [list(course) for course in FRESH_COURSES],
        "lateral_options_m_s": list(LATERAL_OPTIONS),
        "run_speed_m_s": 1.4,
        "stop_frame": 120,
        "frames": 240,
        "selection_rule": "max mean(score), tie: smaller abs lateral, then numeric lateral",
        "score_rule": "10*foot_first + 10*goal - 20*fall - 2*peak_tilt",
        "promotion_authorized": False,
    }
    protocol["protocol_hash"] = hash_json(protocol)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    options: list[dict[str, Any]] = []
    for option_index, lateral in enumerate(LATERAL_OPTIONS):
        courses: list[dict[str, Any]] = []
        for course_index, (ball_x, ball_y) in enumerate(TRAIN_COURSES):
            report = run(
                model_root=model_root,
                stadium_assets=stadium_assets,
                output_dir=output_dir / f"option-{option_index}-course-{course_index}",
                ball_x_m=ball_x,
                ball_y_m=ball_y,
                frames=240,
                run_speed_mps=1.4,
                run_lateral_mps=lateral,
                stop_frame=120,
                feedback_actor=actor_path,
            )
            fall = report["minimum_pelvis_height_m"] < 0.65
            score = (
                10.0 * float(report["first_robot_ball_contact_is_foot"])
                + 10.0 * float(report["whole_ball_goal_crossed"])
                - 20.0 * float(fall)
                - 2.0 * float(report["peak_pelvis_tilt_rad"])
            )
            courses.append(
                {
                    "course": [ball_x, ball_y],
                    "report_hash": report["report_hash"],
                    "foot_first": report["first_robot_ball_contact_is_foot"],
                    "goal": report["whole_ball_goal_crossed"],
                    "minimum_pelvis_height_m": report["minimum_pelvis_height_m"],
                    "peak_pelvis_tilt_rad": report["peak_pelvis_tilt_rad"],
                    "score": score,
                }
            )
            print(
                json.dumps(
                    {
                        "option": lateral,
                        "course": [ball_x, ball_y],
                        "score": score,
                        "foot_first": courses[-1]["foot_first"],
                        "goal": courses[-1]["goal"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        options.append(
            {
                "lateral_m_s": lateral,
                "mean_score": sum(float(row["score"]) for row in courses) / len(courses),
                "courses": courses,
            }
        )
        (output_dir / "progress.json").write_text(
            json.dumps(options, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    selected = max(
        options,
        key=lambda option: (
            option["mean_score"],
            -abs(option["lateral_m_s"]),
            -option["lateral_m_s"],
        ),
    )
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(helper.read_bytes()) != helper_hash
    ):
        raise RuntimeError("training source changed during closed-loop rollouts")
    result: dict[str, Any] = {
        "schema": protocol["schema"],
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": protocol["protocol_hash"],
        "actor_hash": actor_commitment,
        "options": options,
        "selected_lateral_m_s": selected["lateral_m_s"],
        "selected_training_score": selected["mean_score"],
        "fresh_holdout_evaluated": False,
        "promotion_authorized": False,
    }
    result["result_hash"] = hash_json(result)
    (output_dir / "selection.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--actor-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    result = train(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "selected_lateral_m_s": result["selected_lateral_m_s"],
                "selected_training_score": result["selected_training_score"],
                "result_hash": result["result_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
