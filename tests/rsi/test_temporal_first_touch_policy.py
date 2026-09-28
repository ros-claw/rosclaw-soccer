"""Temporal joint residuals are observation-bound, bounded, and SIM_ONLY."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.temporal_first_touch_policy import (
    FEATURE_NAMES,
    JOINT_NAMES,
    candidate_manifest,
    load_candidate,
    temporal_residual,
)


def test_temporal_residual_is_bounded_and_only_before_incoming_contact() -> None:
    weights = np.zeros((len(FEATURE_NAMES), len(JOINT_NAMES)))
    weights[0, 1] = 0.5
    inputs = {
        "ball_relative_xyz_m": (0.7, 0.0, -0.6),
        "ball_vx_m_s": -0.5,
        "joint_position_rad": np.zeros(29),
        "joint_velocity_rad_s": np.zeros(29),
    }
    action = temporal_residual(weights, **inputs)
    assert 0 < action[1] <= 0.08
    assert action[0] == action[2] == 0
    assert np.array_equal(temporal_residual(weights, **{**inputs, "ball_vx_m_s": 0.4}), np.zeros(3))
    assert np.array_equal(
        temporal_residual(weights, **{**inputs, "ball_relative_xyz_m": (1.3, 0.0, 0.0)}),
        np.zeros(3),
    )
    with pytest.raises(ValueError, match="unbounded"):
        temporal_residual(weights * 3, **inputs)


def test_temporal_manifest_rejects_tampering(tmp_path: Path) -> None:
    courses = sample_training_courses(20260930)
    weights = np.zeros((16, len(FEATURE_NAMES), len(JOINT_NAMES)))
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash="sha256:" + "a" * 64,
        weights_per_course=weights,
        seed=1,
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        load_candidate(
            path, expected_courses=courses, parent_report_hash="sha256:" + "a" * 64
        ).weights_per_course.shape
        == weights.shape
    )
    manifest["weights_per_course"][0][0][0] = 0.4
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="commitment"):
        load_candidate(path, expected_courses=courses, parent_report_hash="sha256:" + "a" * 64)
