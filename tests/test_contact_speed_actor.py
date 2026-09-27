from __future__ import annotations

import copy

import pytest

from rosclaw_soccer.sim.contact_speed_actor import (
    choose_contact_speed,
    fit_contact_speed_actor,
)


def _report(x: float, speed: float, success: bool) -> dict[str, object]:
    return {
        "ball_x_m": x,
        "ball_y_m": -0.1,
        "forward_command_m_s": speed,
        "model_hash": "model",
        "asset_hash": "asset",
        "source_hash": "source",
        "track_ball_contacts": True,
        "clean_foot_only_contact_verified": success,
        "ball_horizontal_displacement_m": 1.0 if success else 0.1,
        "stand_passed": True,
        "individual": {"blue.playmaker": {"joint_projection_count": 0}},
        "report_hash": f"report-{x}-{speed}",
    }


def test_fits_consumed_paired_outcomes_and_seals_action() -> None:
    pairs = [
        (_report(2.38, 0.45, True), _report(2.38, 0.5, False)),
        (_report(2.48, 0.45, True), _report(2.48, 0.5, False)),
        (_report(2.58, 0.45, False), _report(2.58, 0.5, True)),
        (_report(2.68, 0.45, False), _report(2.68, 0.5, True)),
    ]
    actor = fit_contact_speed_actor(pairs)
    assert actor["ball_x_threshold_m"] == pytest.approx(2.53)
    assert actor["train_joint_success_count"] == 4
    assert choose_contact_speed(actor, observed_ball_x_m=2.42) == 0.45
    assert choose_contact_speed(actor, observed_ball_x_m=2.62) == 0.5
    tampered = copy.deepcopy(actor)
    tampered["ball_x_threshold_m"] = 2.9
    with pytest.raises(ValueError):
        choose_contact_speed(tampered, observed_ball_x_m=2.42)


def test_rejects_unpaired_training_data() -> None:
    pairs = [
        (_report(2.3 + index * 0.1, 0.45, True), _report(2.3 + index * 0.1, 0.5, False))
        for index in range(4)
    ]
    pairs[0][1]["ball_y_m"] = -0.2
    with pytest.raises(ValueError):
        fit_contact_speed_actor(pairs)
