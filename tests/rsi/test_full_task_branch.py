"""Actual toy physics and synthetic ONNX/Foundation, not G1 skill evidence."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi import causal_impact_replay as causal
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.full_task_branch import CAUSAL_FIELDS, FullTaskBranch, integration_hash
from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    CompiledRecurrentSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.recurrent_success_motor import make_model
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.physics.test_g1_feedback_forecast import world
from tests.providers.test_sonic_branch_state import tracker
from tests.providers.test_sonic_navigation_branch import blank_navigation
from tests.rsi.test_causal_impact_replay import (  # noqa: F401
    candidate,
    current,
    imitation_parent,
    model,
    smooth_parent,
)


def build(monkeypatch, parent_model, *, born_release_capture=False):
    sim, _, _ = world()
    sim.opt.gravity[:] = 0
    sim.dof_armature[6:35] = 0.1
    sim.jnt_range[:] = [-2.0, 2.0]
    binary = np.empty(mujoco.mj_sizeModel(sim), dtype=np.uint8)
    mujoco.mj_saveModel(sim, None, binary)
    world_hash = hash_bytes(binary.tobytes())
    nav = blank_navigation()
    foundation, _, _ = tracker()
    foundation.model.qualification.reference_stride = 1
    foundation.model.qualification.heading_normalized = False
    foundation.model.qualification.qualification_hash = "synthetic"
    foundation.model.qualification.require_eligible = lambda: None
    monkeypatch.setattr(causal, "G1SonicNavigation", lambda *a, **kw: nav)
    policy = make_preview(make_sampling_view(make_model(parent_model), seed=901))
    parent = causal.CausalImpactReplay(
        sim,
        world_hash=world_hash,
        kp=nav.backend.kp.copy(),
        kd=nav.backend.kd.copy(),
        model_root=Path("synthetic"),
        foundation=foundation.model,
        expected_foundation_hash="synthetic",
        policy=policy,
        decoder=CompiledRecurrentSamplingMotor(policy),
        expected_policy_hash=policy["policy_hash"],
        course=(2.5, 0.0, 0.2),
        profile=BoundedFirstTouchControl(),
        heading=FootHeadingContactProbe(False),
        impact_enabled=False,
        born_release_capture=born_release_capture,
    )
    return parent, world_hash


def test_whole_future_body_nn_and_planner_recomputed_from_owned_state(monkeypatch, request):
    parent, wh = build(monkeypatch, request.getfixturevalue("imitation_parent"))
    for f in range(31):
        parent.step(f)
    donor = integration_hash(parent.model, parent.data)
    baseline = FullTaskBranch(parent, world_hash=wh, action_name="baseline")
    repeat = FullTaskBranch(parent, world_hash=wh, action_name="baseline")
    a = [baseline.step() for _ in range(269)]
    b = [repeat.step() for _ in range(269)]
    assert integration_hash(parent.model, parent.data) == donor and parent.next_frame == 31
    for branch in (baseline, repeat):
        with pytest.raises(ValueError, match="whole300"):
            branch._owned.review()
        r = branch.review()
        assert r["new_native_integrated_steps"] == 2690 and r["borrowed_prefix_frames"] == 31
        assert r["new_planner_onnx_calls"] == 13
        assert r["recomputed_foundation_encoder_decoder_pairs"] == 269
        assert not r["prefix_steps_counted_as_new"] and not r["teacher_qualified"]
        with pytest.raises(ValueError):
            branch.step()
    gold = [parent.step(f) for f in range(31, 300)]
    for predicted, repeated, actual in zip(a, b, gold, strict=True):
        assert predicted.frame == repeated.frame == actual.frame
        assert predicted.arrays.keys() == actual.arrays.keys() == repeated.arrays.keys()
        for k in actual.arrays:
            assert (
                predicted.arrays[k].tobytes()
                == actual.arrays[k].tobytes()
                == repeated.arrays[k].tobytes()
            ), k
    assert parent.review()["native_integrated_steps"] == 3000


@pytest.mark.parametrize("bad", ["extra", "unfinished", "recursive", "action"])
def test_invalid_or_recursive_source_never_allocates_future(monkeypatch, request, bad):
    parent, wh = build(monkeypatch, request.getfixturevalue("imitation_parent"))
    for f in range(31):
        parent.step(f)
    action = "baseline"
    if bad == "extra":
        parent.new_unreviewed_state = []
    elif bad == "unfinished":
        parent.failed = True
    elif bad == "recursive":
        parent.postcontact_exit_enabled = True
    else:
        action = "arbitrary-external-torque"
    before = integration_hash(parent.model, parent.data)
    with pytest.raises(ValueError, match="completed plain"):
        FullTaskBranch(parent, world_hash=wh, action_name=action)
    assert integration_hash(parent.model, parent.data) == before and parent.next_frame == 31


def test_nonzero_lease_not_forged_from_an_unconsumed_release_clock():
    # Deliberately incomplete synthetic state: rejection precedes any model,
    # data, decoder or forecaster allocation.
    parent = object.__new__(causal.CausalImpactReplay)
    for field in CAUSAL_FIELDS:
        setattr(parent, field, None)
    parent.failed = False
    parent.next_frame = 31
    parent.attempted_steps = parent.integrated_steps = 310
    parent.early_impact_enabled = parent.postcontact_exit_enabled = False
    parent.reuse_owned_postcontact_updates = parent.constant_forward_approach = False
    parent.profile = BoundedFirstTouchControl()
    with pytest.raises(ValueError, match="completed plain"):
        FullTaskBranch(parent, world_hash="unused", action_name="forward_minus")
