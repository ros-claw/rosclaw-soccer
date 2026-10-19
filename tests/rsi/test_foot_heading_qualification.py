"""Review validation only; synthetic flags are not dynamics certificates."""

import pytest

from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.foot_heading_evidence import TAG, require_heading_capture
from rosclaw_soccer.rsi.original_ik_planner_evidence import TAG as PULSE_TAG
from rosclaw_soccer.rsi.original_ik_planner_evidence import require_original_ik_capture
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_first_touch_qualification import fixture as base_fixture
from tests.rsi.test_foot_heading_evidence import reconstruct


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "implicit",
        "missing_pulse",
        "wrong_heading_hash",
        "missing_cpu",
        "short",
        "old_cpu",
        "geometry_claim",
        "new_steps",
        "authority",
        "original_source",
        "nav_link",
        "truthy",
        "unsealed",
        "new_implementation",
        "no_dynamics",
    ],
)
def test_explicit_combined_heading_cannot_bypass_original_ik_and_whole_cpu_review(
    enabled, mutation
):
    c, trace, replay = reconstruct(enabled)
    capture = require_heading_capture(c, c, trace)
    scalar = require_original_ik_capture(c, c, trace)
    cpu = replay.review()
    review, foundation, outcome = base_fixture()
    review.update(
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=True,
        original_ik_planner_pulse_review=scalar,
        foot_heading_capture_review=capture,
        foot_heading_cpu_reconstruction=cpu,
        foot_heading_capture_frames_compared=300,
    )
    kwargs = dict(
        original_ik_planner_pulse_contract_hash=c[PULSE_TAG]["contract_hash"],
        foot_heading_contract_hash=c[TAG]["contract_hash"],
    )
    if mutation == "implicit":
        kwargs.pop("foot_heading_contract_hash")
    elif mutation == "missing_pulse":
        kwargs.pop("original_ik_planner_pulse_contract_hash")
    elif mutation == "wrong_heading_hash":
        kwargs["foot_heading_contract_hash"] = "sha256:" + "e" * 64
    elif mutation == "missing_cpu":
        review.pop("foot_heading_cpu_reconstruction")
    elif mutation == "short":
        review["foot_heading_capture_frames_compared"] = 299
    elif mutation == "old_cpu":
        review["original_ik_cpu_reconstruction"] = {}
    elif mutation == "geometry_claim":
        capture["actual_cpu_geometry_reconstructed_here"] = True
    elif mutation == "new_steps":
        cpu["new_physical_steps"] = 3000
    elif mutation == "authority":
        cpu["training_authorized"] = True
    elif mutation == "original_source":
        cpu["original_ik_source_hash"] = "sha256:" + "f" * 64
    elif mutation == "nav_link":
        capture["navigation_review_hash"] = "sha256:" + "c" * 64
    elif mutation == "truthy":
        review["foot_heading_capture_frames_compared"] = 300.0
    elif mutation == "unsealed":
        cpu["report_hash"] = "sha256:" + "b" * 64
    elif mutation == "new_implementation":
        cpu["independent_heading_implementation"] = True
    elif mutation == "no_dynamics":
        review["actual_mujoco_dynamics_replayed"] = False
    if mutation != "unsealed":
        for r in (capture, scalar, cpu):
            r["report_hash"] = hash_json({k: v for k, v in r.items() if k != "report_hash"})
    if mutation is None:
        r = require_first_touch_qualification(review, foundation, outcome, **kwargs)
        assert r["foot_heading_contract_hash"] == c[TAG]["contract_hash"]
        assert r["original_ik_planner_pulse_contract_hash"] == c[PULSE_TAG]["contract_hash"]
        assert not r["promotion_authorized"]
    else:
        with pytest.raises(ValueError):
            require_first_touch_qualification(review, foundation, outcome, **kwargs)
