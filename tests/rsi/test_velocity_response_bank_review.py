"""Synthetic aggregate/receipt mutation checks, not physical success evidence."""

import pytest
from test_velocity_policy_native_evidence import fixture

from rosclaw_soccer.rsi.velocity_policy_native_evidence import review_velocity_recording
from rosclaw_soccer.rsi.velocity_response_bank_review import review_episode, summarize_bank


def receipt(raw, p, s, c):
    return dict(
        course_index=c.index,
        split=c.split,
        native_steps=3000,
        neural_calls=300,
        attempted_neural_calls=300,
        optimizer_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
        body_report=dict(frames=300, integrated_steps=3000, attempted_steps=3000),
        **review_velocity_recording(raw, p, s, c),
    )


def test_exact_episode_derived_from_raw_and_independent_declared_initial():
    raw, p, s, c = fixture()
    row = review_episode(raw, receipt(raw, p, s, c), p, s, c)
    assert row["course_index"] == 0
    assert not row["football_skill_proven"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("native_steps", 2990),
        ("neural_calls", 299),
        ("attempted_neural_calls", 301),
        ("hardware_authorized", True),
        ("pulse_mean_velocity_error_mps", 0.9),
        ("split", "validation"),
    ],
)
def test_wrong_receipt_rejected(key, value):
    raw, p, s, c = fixture()
    r = receipt(raw, p, s, c)
    r[key] = value
    with pytest.raises(ValueError):
        review_episode(raw, r, p, s, c)


def test_initial_state_not_inherited_from_record():
    raw, p, s, c = fixture()
    raw["canonical_qpos"][:, 0] = 0.1
    raw["target_body_canonical_qpos_after_substep"][:, :, 0] = 0.1
    with pytest.raises(ValueError, match="initial canonical"):
        review_episode(raw, receipt(raw, p, s, c), p, s, c)


def test_whole_unique_bank_and_no_football_claim():
    rows = [
        dict(
            course_index=i,
            split="train" if i < 9 else "validation",
            physical_response_guard=True,
            command_response_research_gate=i != 2,
            pulse_mean_velocity_error_mps=0.01,
            terminal_mean_speed_mps=0.02,
            after_pulse_root_drift_m=0.03,
        )
        for i in range(15)
    ]
    r = summarize_bank(rows)
    assert r["research_response_gate_passes"] == 14
    assert r["development_validation_gate_passes"] == 6
    assert r["independent_initial_worlds"] == 1
    assert r["goal_complete"] is False
    for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError):
            summarize_bank(bad)
