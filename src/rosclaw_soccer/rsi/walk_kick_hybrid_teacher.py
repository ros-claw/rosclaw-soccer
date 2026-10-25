"""Frozen local teacher composition, selected by measured initial state only.

The routing rule is a post-pilot hypothesis: fast ONLY for high-x outward
balls. New public40 validation must precede any generalization claim.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course
from rosclaw_soccer.rsi.walk_kick_teacher_plan import teacher_courses
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_course_preflight import (
    FirstTouchCourse,
    preflight_receiving_courses,
)


def compose_teacher(slow: dict[str, Any], fast: dict[str, Any]) -> dict[str, Any]:
    expected_status = "PUBLIC16_CEM_ALL304_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED"
    bank = teacher_courses()
    bank_hash = preflight_receiving_courses(bank)
    for report, mode in ((slow, False), (fast, True)):
        if (
            report["status"] != expected_status
            or report.get("experimental_fast_approach", False) is not mode
            or report["course_bank_hash"] != bank_hash
            or [r["course_index"] for r in report["results"]] != list(range(16))
            or report["promotion_authorized"] is not False
        ):
            raise ValueError("two entire authenticated public16 teacher studies required")
    prototypes = []
    for i, c in enumerate(bank):
        fast_mode = validate_public_course(c.course) == 3
        report = fast if fast_mode else slow
        summary = report["results"][i]
        suffix = f"/course-{i:02d}/{summary['selected']}.json"
        paths = [Path(k) for k in report["input_file_hashes"] if k.endswith(suffix)]
        if (
            len(paths) != 1
            or paths[0].is_symlink()
            or (file_hash(paths[0]) != report["input_file_hashes"][str(paths[0])])
        ):
            raise ValueError("unique exact selected physical teacher record required")
        row = json.loads(paths[0].read_text())
        if row["course"] != list(c.course) or hash_json(row["outcome"]) != hash_json(
            summary["selected_outcome"]
        ):
            raise ValueError("teacher prototype differs from audited physical outcome")
        prototypes.append(
            dict(
                course=list(c.course),
                parameters=row["parameters"],
                experimental_fast=fast_mode,
                source_record_hash=report["input_file_hashes"][str(paths[0])],
            )
        )
    value = dict(
        schema="soccer.walk_kick.initial_context_hybrid_teacher.v1",
        routing="experimental fast iff initial x>=2.6 and vx>0; otherwise slow",
        nearest_context_scales=[0.1, 0.16, 0.3],
        prototypes=prototypes,
        per_state_oracle_replaced_with_fixed_initial_context_rule=True,
        teacher_qualification_proven=False,
        neural_training=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(value, table_hash=hash_json(value))


def select_teacher(table: dict[str, Any], course: tuple[float, float, float]) -> dict[str, Any]:
    bucket = validate_public_course(course)
    unhashed = {k: v for k, v in table.items() if k != "table_hash"}
    if hash_json(unhashed) != table["table_hash"] or len(table["prototypes"]) != 16:
        raise ValueError("unchanged frozen prototype teacher required")
    indices = [
        i
        for i, row in enumerate(table["prototypes"])
        if validate_public_course(tuple(row["course"])) == bucket
    ]
    if len(indices) != 4:
        raise ValueError("four physical teacher prototypes per launch class required")
    features = np.asarray(course)
    scales = np.array([0.1, 0.16, 0.3])
    chosen = min(
        indices,
        key=lambda i: float(
            np.mean(((np.asarray(table["prototypes"][i]["course"]) - features) / scales) ** 2)
        ),
    )
    row = table["prototypes"][chosen]
    return dict(
        parameters=dict(row["parameters"]),
        experimental_fast=bucket == 3,
        prototype_index=chosen,
        table_hash=table["table_hash"],
    )


def expanded_validation_courses() -> tuple[FirstTouchCourse, ...]:
    """Ten new initial states in each full public class, no outcome rejection."""
    rng = np.random.default_rng(2026102511)
    rows: list[FirstTouchCourse] = []
    for bucket in range(4):
        for _ in range(10):
            xmin = 2.2 if bucket < 2 else 2.6
            x = float(rng.uniform(xmin, xmin + 0.2))
            y = float(rng.uniform(-0.16, 0.16))
            vx = float(rng.uniform(0.25, 0.7)) * (1 if bucket % 2 else -1)
            rows.append(FirstTouchCourse("striker", 2026102600 + len(rows), (x, y, vx)))
    courses = tuple(rows)
    preflight_receiving_courses(courses)
    if len({r.course for r in courses}) != 40 or set(r.course for r in courses) & {
        r.course for r in teacher_courses()
    }:
        raise ValueError("entire distinct40 public validation bank required")
    return courses
