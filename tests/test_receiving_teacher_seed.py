import numpy as np
import pytest

from rosclaw_soccer.training.receiving_teacher_seed import teacher_seeded_schedules


def inputs():
    return dict(
        agent_id="blue.playmaker",
        entry_frame=30,
        applied_leg_residual=np.zeros((300, 12)),
        teacher_leg_torque_nm=np.ones((300, 12)),
        valid_focal_teacher=np.ones(300, dtype=bool),
        leg_kp=np.full(12, 100.0),
    )


def test_four_named_immutable_bounded_proposals_and_no_input_mutation():
    args = inputs()
    before = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    proposals = dict(teacher_seeded_schedules(**args))
    assert tuple(proposals) == (
        "four_sampled",
        "four_inverse",
        "four_lead_100ms",
        "dense_lead_100ms",
    )
    for name, schedule in proposals.items():
        assert schedule.agent_id == "blue.playmaker" and schedule.start_frame == 30
        assert schedule.substrate == "A0_leg12"
        assert len(schedule.knots) == (27 if name.startswith("dense") else 4)
        assert schedule.knot_frames == (10 if name.startswith("dense") else 20)
        np.testing.assert_allclose(schedule.knots, 0.1)
    for key, value in before.items():
        np.testing.assert_array_equal(args[key], value)


def test_mask_removes_other_player_or_unexecuted_teacher_but_not_applied_residual():
    args = inputs()
    args["valid_focal_teacher"][:] = False
    args["applied_leg_residual"][:] = -0.02
    for _, schedule in teacher_seeded_schedules(**args):
        np.testing.assert_allclose(schedule.knots, -0.2)


def test_lead_is_future_information_and_inverse_is_not_rate_limit_inverse():
    args = inputs()
    args["teacher_leg_torque_nm"][:] = 0
    args["teacher_leg_torque_nm"][35] = 2
    proposals = dict(teacher_seeded_schedules(**args))
    np.testing.assert_array_equal(proposals["four_sampled"].knots[0], np.zeros(12))
    np.testing.assert_allclose(proposals["four_lead_100ms"].knots[0], 0.2)
    args["teacher_leg_torque_nm"][30] = 300
    for _, schedule in teacher_seeded_schedules(**args):
        assert np.max(abs(np.asarray(schedule.knots))) <= 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("entry_frame", True),
        ("entry_frame", -1),
        ("entry_frame", 40),
        ("leg_kp", np.zeros(12)),
        ("leg_kp", np.full(12, np.nan)),
        ("leg_kp", np.ones(29)),
        ("leg_kp", np.full(12, 1e-100)),
        ("teacher_leg_torque_nm", np.full((300, 12), np.inf)),
        ("teacher_leg_torque_nm", np.zeros((300, 29))),
        ("valid_focal_teacher", np.ones(300, dtype=int)),
        ("valid_focal_teacher", np.ones(299, dtype=bool)),
        ("applied_leg_residual", np.full((300, 12), 0.2)),
        ("applied_leg_residual", np.full((300, 12), "0.1")),
        ("agent_id", "bad\nagent"),
    ],
)
def test_invalid_evidence_or_window_is_rejected(key, value):
    args = inputs()
    args[key] = value
    with pytest.raises(ValueError):
        teacher_seeded_schedules(**args)
