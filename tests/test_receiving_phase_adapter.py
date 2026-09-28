"""Simulation-only causal receiving adapter contract."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.receiving_phase_adapter import ReceivingPhaseAdapter


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.zeros(43, dtype=np.float64)
    qpos[36] = 0.4
    qvel = np.zeros(41, dtype=np.float64)
    target = np.zeros(29, dtype=np.float64)
    ranges = np.tile(np.asarray((-1.0, 1.0)), (29, 1))
    return qpos, qvel, target, ranges


def test_phase_adapter_is_causal_bounded_and_immutable() -> None:
    parameters = np.asarray((-0.75, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0))
    actor = ReceivingPhaseAdapter(parameters)
    parameters[0] = 1.0
    assert actor.parameters[0] == -0.75
    assert not actor.parameters.flags.writeable
    qpos, qvel, target, ranges = _inputs()
    before = actor.motor_target(
        qpos=qpos,
        qvel=qvel,
        foundation_target=target,
        joint_ranges=ranges,
        has_foot_contact=False,
        elapsed_sec=-1.0,
    )
    after = actor.motor_target(
        qpos=qpos,
        qvel=qvel,
        foundation_target=target,
        joint_ranges=ranges,
        has_foot_contact=True,
        elapsed_sec=0.0,
    )
    assert before[1] < 0 and before[6] == 0
    assert after[1] == 0 and after[6] > 0
    assert np.max(np.abs(before)) <= 0.10
    assert np.max(np.abs(after)) <= 0.10
    assert np.array_equal(target, np.zeros(29))
    assert actor.artifact_hash.startswith("sha256:")


def test_phase_adapter_rejects_unbounded_or_nonfinite_actions() -> None:
    with pytest.raises(ValueError):
        ReceivingPhaseAdapter(np.full(8, 1.01))
    with pytest.raises(ValueError):
        ReceivingPhaseAdapter(np.full(8, np.nan))
    actor = ReceivingPhaseAdapter(np.zeros(8))
    qpos, qvel, target, ranges = _inputs()
    qpos[36] = np.inf
    with pytest.raises(ValueError):
        actor.motor_target(
            qpos=qpos,
            qvel=qvel,
            foundation_target=target,
            joint_ranges=ranges,
            has_foot_contact=False,
            elapsed_sec=-1.0,
        )
