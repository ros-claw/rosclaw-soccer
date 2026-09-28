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
    followthrough_residual,
    knee_extension_probe_weights,
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


def test_followthrough_is_bounded_and_fades_without_jump() -> None:
    contact = np.asarray((0.06, -0.04, 0.02))
    first = followthrough_residual(contact, elapsed_frames=1, followthrough_frames=20)
    last = followthrough_residual(contact, elapsed_frames=20, followthrough_frames=20)
    assert np.max(np.abs(first - contact)) < 0.003
    assert np.max(np.abs(last)) < 0.003
    assert np.array_equal(
        followthrough_residual(contact, elapsed_frames=21, followthrough_frames=20), np.zeros(3)
    )
    for elapsed in (1, 10, 20, 21):
        assert np.array_equal(
            followthrough_residual(np.zeros(3), elapsed_frames=elapsed, followthrough_frames=20),
            np.zeros(3),
        )
    with pytest.raises(ValueError, match="postcontact"):
        followthrough_residual(contact, elapsed_frames=1, followthrough_frames=31)


def test_knee_extension_probe_is_shared_and_only_changes_incoming_right_knee() -> None:
    weights = knee_extension_probe_weights()
    assert weights.shape == (16, len(FEATURE_NAMES), len(JOINT_NAMES))
    assert np.array_equal(weights[0], weights[-1])
    assert np.count_nonzero(weights) == 32
    q = np.zeros(29)
    q[9] = 1.1
    inputs = {
        "ball_relative_xyz_m": (0.7, 0.0, -0.6),
        "ball_vx_m_s": -0.5,
        "joint_position_rad": q,
        "joint_velocity_rad_s": np.zeros(29),
    }
    residual = temporal_residual(weights[0], **inputs)
    assert residual[0] == residual[2] == 0
    assert -0.08 <= residual[1] < 0
    assert np.array_equal(
        temporal_residual(weights[0], **{**inputs, "ball_vx_m_s": 0.5}), np.zeros(3)
    )
