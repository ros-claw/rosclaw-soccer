"""Create a sealed, single-course or single-joint Isaac candidate ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.first_touch_candidate import (
    JOINT_NAMES,
    candidate_manifest,
    load_first_touch_candidate,
)
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-folder", required=True, type=Path)
    parser.add_argument("--source-candidate", required=True, type=Path)
    parser.add_argument("--source-parent-folder", type=Path)
    parser.add_argument("--course-index", required=True, type=int)
    parser.add_argument("--joint", choices=JOINT_NAMES, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = audit_vector_first_touch(args.parent_folder)
    parent = json.loads((args.parent_folder / "report.json").read_text(encoding="utf-8"))
    if audit["source_report_hash"] != parent["report_hash"]:
        raise ValueError("Parent physical report failed authentication")
    courses = tuple(
        (
            row["course"]["ball_x_m"],
            row["course"]["ball_y_local_m"],
            row["course"]["ball_vx_m_s"],
        )
        for row in parent["environments"]
    )
    if not 0 <= args.course_index < len(courses):
        parser.error("course index outside authenticated Parent")
    source_parent_folder = args.source_parent_folder or args.parent_folder
    source_audit = audit_vector_first_touch(source_parent_folder)
    source_parent = json.loads((source_parent_folder / "report.json").read_text(encoding="utf-8"))
    if source_audit["source_report_hash"] != source_parent["report_hash"]:
        raise ValueError("source Parent physical report failed authentication")
    source_courses = tuple(
        (
            row["course"]["ball_x_m"],
            row["course"]["ball_y_local_m"],
            row["course"]["ball_vx_m_s"],
        )
        for row in source_parent["environments"]
    )
    source = load_first_touch_candidate(
        args.source_candidate,
        expected_courses=source_courses,
        parent_report_hash=source_parent["report_hash"],
    )
    if courses[args.course_index] not in source_courses:
        raise ValueError("target course absent from source physical curriculum")
    source_index = source_courses.index(courses[args.course_index])
    joint_indices = (
        set(range(len(JOINT_NAMES)))
        if not args.joint
        else {JOINT_NAMES.index(name) for name in args.joint}
    )
    source_row = source.actions_rad[source_index]
    actions = tuple(
        tuple(
            value if course_index == args.course_index and joint_index in joint_indices else 0.0
            for joint_index, value in enumerate(source_row)
        )
        for course_index in range(len(courses))
    )
    original = json.loads(args.source_candidate.read_text(encoding="utf-8"))
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        actions_rad=actions,
        seed=original["seed"],
    )
    manifest["ablation_source_candidate_hash"] = source.candidate_hash
    manifest["ablation_source_parent_report_hash"] = source_parent["report_hash"]
    manifest["ablation_course_index"] = args.course_index
    manifest["ablation_joints"] = [JOINT_NAMES[index] for index in sorted(joint_indices)]
    from rosclaw_soccer.sim.contracts import hash_json

    manifest.pop("candidate_hash")
    manifest["candidate_hash"] = hash_json(manifest)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {"candidate_hash": manifest["candidate_hash"], "course_index": args.course_index}
        )
    )


if __name__ == "__main__":
    main()
