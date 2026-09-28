"""Candidate action tables cannot bypass Parent, course or joint bounds."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.first_touch_candidate import (
    JOINT_NAMES,
    candidate_manifest,
    guarded_residual_target,
    load_first_touch_candidate,
    project_residual_target,
)

COURSES = ((2.4, -0.1, -0.5), (2.4, -0.1, 0.5))
PARENT = "sha256:" + "a" * 64


def _candidate(path: Path) -> None:
    actions = tuple((0.0,) * len(JOINT_NAMES) for _ in COURSES)
    path.write_text(
        json.dumps(
            candidate_manifest(
                courses=COURSES, parent_report_hash=PARENT, actions_rad=actions, seed=7
            )
        ),
        encoding="utf-8",
    )


def test_candidate_is_parent_and_course_bound(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    _candidate(path)
    selected = load_first_touch_candidate(path, expected_courses=COURSES, parent_report_hash=PARENT)
    assert selected.actions_rad == ((0.0,) * 6, (0.0,) * 6)
    with pytest.raises(ValueError, match="Parent"):
        load_first_touch_candidate(
            path, expected_courses=COURSES, parent_report_hash="sha256:" + "b" * 64
        )
    with pytest.raises(ValueError, match="Parent"):
        load_first_touch_candidate(
            path,
            expected_courses=((2.4, 0.1, -0.5), COURSES[1]),
            parent_report_hash=PARENT,
        )


@pytest.mark.parametrize("bad", [0.081, float("nan"), True])
def test_invalid_residual_rejected(tmp_path: Path, bad: float) -> None:
    path = tmp_path / "candidate.json"
    _candidate(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["actions_rad"][0][0] = bad
    record.pop("candidate_hash")
    from rosclaw_soccer.sim.contracts import hash_json

    record["candidate_hash"] = (
        "invalid" if isinstance(bad, float) and math.isnan(bad) else hash_json(record)
    )
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="unbounded"):
        load_first_touch_candidate(path, expected_courses=COURSES, parent_report_hash=PARENT)


def test_residual_never_worsens_baseline_joint_limit_violation() -> None:
    assert guarded_residual_target(1.2, 0.0, -1.0, 1.0) == 1.2
    assert guarded_residual_target(1.2, -0.05, -1.0, 1.0) == pytest.approx(1.15)
    with pytest.raises(ValueError, match="outside"):
        guarded_residual_target(1.2, 0.05, -1.0, 1.0)
    with pytest.raises(ValueError, match="outside"):
        guarded_residual_target(0.98, 0.05, -1.0, 1.0)
    assert project_residual_target(1.2, 0.05, -1.0, 1.0) == (1.2, True)
    assert project_residual_target(1.2, -0.05, -1.0, 1.0)[1] is False
