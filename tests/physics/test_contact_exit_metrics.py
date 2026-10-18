import numpy as np
import pytest

from rosclaw_soccer.physics.contact_exit_metrics import contact_exit_metrics


def arrays():
    a = dict(
        canonical_qvel_before_substep=np.zeros((3000, 41)),
        canonical_qvel_after_substep=np.zeros((3000, 41)),
        all_contact_force_on_ball_world_n=np.zeros((3000, 3)),
        robot_contact_force_on_ball_world_n=np.zeros((3000, 3)),
    )
    a["all_contact_force_on_ball_world_n"][:, 2] = 0.43 * 9.81
    return a


def event(step, name="left_ankle_roll_link"):
    return dict(substep_index=step, force_norm_n=2.0, other_body_name=name)


def test_unique_steps_not_contact_point_count_and_distinct_duration_clocks():
    d = contact_exit_metrics(
        arrays(), [event(5), event(5), event(6), event(8)], mass_kg=0.43, dt_s=0.002
    )
    assert d["unique_active_foot_substeps"] == 3
    assert d["longest_contiguous_active_foot_substeps"] == 2
    assert d["summed_active_foot_duration_s"] == 0.006
    assert d["first_to_last_foot_contact_span_s"] == 0.006
    assert d["longest_contiguous_active_foot_duration_s"] == 0.004
    assert d["maximum_linear_newton_residual_n"] == 0


def test_absent_contact_never_fabricates_exit_or_reversal():
    d = contact_exit_metrics(arrays(), [], mass_kg=0.43, dt_s=0.002)
    assert d["first_foot_contact_substep"] is None
    assert d["initial_20ms"] is None
    assert d["slowing_until_reversal"] is None
    assert d["skill_success_inferred"] is False


def test_discontinuous_ball_clock_is_not_a_momentum_measurement():
    a = arrays()
    a["canonical_qvel_before_substep"][1, 35] = 1
    with pytest.raises(ValueError, match="contiguous"):
        contact_exit_metrics(a, [], mass_kg=0.43, dt_s=0.002)


def test_momentum_impulse_and_reversal_before_nonfoot_contact():
    a = arrays()
    v = np.zeros(3001)
    v[11:20] = 2
    v[20:] = -1
    a["canonical_qvel_before_substep"][:, 35] = v[:-1]
    a["canonical_qvel_after_substep"][:, 35] = v[1:]
    a["all_contact_force_on_ball_world_n"][:, 0] = 0.43 * np.diff(v) / 0.002
    d = contact_exit_metrics(a, [event(5), event(100, "left_knee_link")], mass_kg=0.43, dt_s=0.002)
    assert d["first_forward_to_backward_crossing_after_initial_20ms_substep"] == 19
    assert d["first_nonfoot_robot_contact_substep"] == 100
    assert d["slowing_until_reversal"]["ball_momentum_change_world_kg_m_s"] == [-1.29, 0, 0]
    assert d["maximum_linear_newton_residual_n"] == 0


@pytest.mark.parametrize(
    "key",
    [
        "canonical_qvel_before_substep",
        "canonical_qvel_after_substep",
        "all_contact_force_on_ball_world_n",
        "robot_contact_force_on_ball_world_n",
    ],
)
@pytest.mark.parametrize("kind", ["missing", "nan", "shape", "dtype"])
def test_invalid_dense_arrays_rejected(key, kind):
    a = arrays()
    if kind == "missing":
        del a[key]
    elif kind == "nan":
        a[key].flat[0] = np.nan
    elif kind == "dtype":
        a[key] = a[key].astype(np.float32)
    else:
        a[key] = a[key][:-1]
    with pytest.raises(ValueError):
        contact_exit_metrics(a, [], mass_kg=0.43, dt_s=0.002)


@pytest.mark.parametrize("step,force", [(True, 2), (-1, 2), (3000, 2), (5, 1), (5, float("nan"))])
def test_contact_events_must_have_valid_clock_and_threshold(step, force):
    with pytest.raises(ValueError):
        contact_exit_metrics(
            arrays(),
            [dict(event(0), substep_index=step, force_norm_n=force)],
            mass_kg=0.43,
            dt_s=0.002,
        )
