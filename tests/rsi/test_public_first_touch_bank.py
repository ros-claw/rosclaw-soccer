"""Reject incomplete or substituted declarations, retain unsafe and failed rows."""

from dataclasses import asdict

import pytest

from rosclaw_soccer.rsi.public_first_touch_bank import validate_public_training_rows
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan

H = "sha256:" + "a" * 64


def bank():
    plan = PublicFirstTouchTrainingPlan((20261060,), H, 18)
    rows = [
        dict(
            **asdict(d),
            actual_behavior_model_hash=H,
            frame_sample_count=270,
            physical_replay_performed_here=True,
            execution_origin="NEW",
            partition="DECLARED_PUBLIC_TRAINING",
            training_plan_hash=plan.contract()["plan_hash"],
            **{
                k: H
                for k in (
                    "view_hash",
                    "physical_report_hash",
                    "physical_trace_hash",
                    "independent_review_hash",
                    "foundation_review_hash",
                    "learning_frames_hash",
                )
            },
            private_fresh_accessed=False,
            promotion_authorized=False,
            hardware_authorized=False,
            outcome=dict(high_quality=False, safety_passed=False),
            terminal_mc_return=-100.0,
        )
        for d in plan.draws()
    ]
    return plan, rows


def test_all_failures_and_unsafe_rows_are_retained_without_authority():
    plan, rows = bank()
    rows[0]["course"] = list(rows[0]["course"])
    result = validate_public_training_rows(plan, rows)
    assert result["episodes"] == 16 and result["contexts"] == 4
    assert result["high_quality_episodes"] == result["safe_episodes"] == 0
    assert result["training_authorized"] is result["physics_validated_here"] is False


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop(),
        lambda r: r.reverse(),
        lambda r: r.append(r[0]),
        lambda r: r[0].update(group=True),
        lambda r: r[0].update(sampling_seed=0),
        lambda r: r[0].update(course=(2.45, 0.0, 0.5)),
        lambda r: r[0].update(training_plan_hash="sha256:" + "b" * 64),
        lambda r: r[0].update(actual_behavior_model_hash="sha256:" + "b" * 64),
        lambda r: r[0].update(partition="FRESH_CPU_EXAM"),
        lambda r: r[0].update(physical_replay_performed_here=1),
        lambda r: r[0].update(frame_sample_count=True),
        lambda r: r[0].update(physical_report_hash="bad"),
        lambda r: r[0].update(private_fresh_accessed=True),
        lambda r: r[0].update(promotion_authorized=True),
        lambda r: r[0].update(terminal_mc_return=float("nan")),
        lambda r: r[0].update(outcome=dict(high_quality=1, safety_passed=False)),
    ],
)
def test_substituted_or_partial_banks_rejected(change):
    plan, rows = bank()
    change(rows)
    with pytest.raises(ValueError):
        validate_public_training_rows(plan, rows)
