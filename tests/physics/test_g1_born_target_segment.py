"""Actual toy native state handoff, synthetic policies, not G1 skill evidence."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_born_target_segment import G1BornTargetSegment
from rosclaw_soccer.rsi.full_task_branch import integration_hash
from tests.rsi.test_causal_impact_replay import (  # noqa: F401
    candidate,
    current,
    imitation_parent,
    model,
    smooth_parent,
)
from tests.rsi.test_full_task_branch import build


def test_owned_born_state_continuity_and_new_counters(monkeypatch, request):
    parent, wh = build(monkeypatch, request.getfixturevalue("imitation_parent"))
    with pytest.raises(ValueError):
        G1BornTargetSegment(parent, world_hash=wh)
    for f in range(31):
        parent.step(f)
    donor = integration_hash(parent.model, parent.data)
    segment = G1BornTargetSegment(parent, world_hash=wh)
    assert segment.data is not parent.data and segment.origin_frame == 31
    assert segment.attempted_steps == segment.integrated_steps == 0
    assert integration_hash(segment.model, segment.data) == donor
    # Independent direct native continuation of identical targets, preserving
    # original warmstart/cached state, without using the segment step method.
    other = mujoco.MjData(parent.model)
    mujoco.mj_copyData(other, parent.model, parent.data)
    target = np.zeros(29, dtype=np.float64)
    for f in range(31, 300):
        row = segment.step(f, target)
        for _ in range(10):
            raw = (target - other.qpos[segment.qi]) * segment.kp - other.qvel[
                segment.vi
            ] * segment.kd
            other.ctrl[segment.ai] = np.clip(raw, -segment.caps, segment.caps)
            mujoco.mj_step(parent.model, other)
        np.testing.assert_array_equal(segment.data.qpos, other.qpos)
        np.testing.assert_array_equal(segment.data.qvel, other.qvel)
        assert row["torque_nm"].shape == (1, 10, 29)
    assert integration_hash(parent.model, parent.data) == donor and parent.next_frame == 31
    review = segment.review()
    assert (
        review["integrated_steps"] == 2690
        and review["borrowed_prefix_steps_counted_as_new"] is False
    )
    assert review["policy_causality_verified"] is False
    with pytest.raises(ValueError):
        segment.step(300, target)


def test_partial_contact_and_fabricated_clock_are_rejected(monkeypatch, request):
    parent, wh = build(monkeypatch, request.getfixturevalue("imitation_parent"))
    for f in range(31):
        parent.step(f)
    parent.ik.contact = 30
    with pytest.raises(ValueError):
        G1BornTargetSegment(parent, world_hash=wh)
    parent.ik.contact = None
    parent.next_frame += 1
    with pytest.raises(ValueError):
        G1BornTargetSegment(parent, world_hash=wh)
