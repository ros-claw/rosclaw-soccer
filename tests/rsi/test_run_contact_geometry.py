import numpy as np
import pytest

from rosclaw_soccer.rsi.run_contact_geometry import summarize_contacts


def event(body="right_ankle_roll_link", role=1, force=20.0, point=None):
    return dict(
        robot_body=body,
        role=role,
        force_norm_N=force,
        solver_time_s=2.0,
        ball_solver_center_world=[0.0, 0.0, 0.11],
        contact_position_world=point or [-0.11, 0.0, 0.09],
    )


def test_forward_radial_geometry_and_body_identity():
    r = summarize_contacts(
        [event(), event(body="right_knee_link", role=2, force=40.0)], np.array([6.0, 0.0, 0.11])
    )
    assert r["nonfoot_body_names"] == ["right_knee_link"]
    foot = r["bodies"]["right_ankle_roll_link"]
    assert foot["radial_heading_degrees"] == 0.0
    assert foot["contact_height_relative_ball_center_m"] == pytest.approx(-0.02)
    assert r["radial_direction_is_not_measured_force_direction"]


def test_side_contact_keeps_signed_radial_heading():
    row = summarize_contacts([event(point=[0.0, 0.11, 0.11])], np.array([6.0, 0.0, 0.11]))
    assert row["bodies"]["right_ankle_roll_link"]["radial_heading_degrees"] == -90.0


def test_multiple_contacts_integral_not_max_role_impulse():
    r = summarize_contacts([event(), event()], np.array([6.0, 0.0, 0.11]))
    assert r["bodies"]["right_ankle_roll_link"]["per_contact_norm_integral_Ns"] == pytest.approx(
        0.08
    )
    assert r["aggregate_norm_integral_is_not_source_max_per_role_impulse"]


def test_body_role_cannot_change():
    with pytest.raises(ValueError):
        summarize_contacts([event(), event(role=2)], np.array([6.0, 0.0, 0.11]))


@pytest.mark.parametrize("force", [1.0, float("nan"), float("inf")])
def test_invalid_native_events_rejected(force):
    with pytest.raises(ValueError):
        summarize_contacts([event(force=force)], np.array([6.0, 0.0, 0.11]))
