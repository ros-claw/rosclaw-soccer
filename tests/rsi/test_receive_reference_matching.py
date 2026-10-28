import numpy as np
import pytest

from rosclaw_soccer.rsi.receive_reference_matching import match_error, yaw_local
from scripts.rsi_match_train_receiving_reference import eligible_geometry_events


def test_yaw_invariant_reference_comparison():
    quat = np.array([2**-0.5, 0.0, 0.0, 2**-0.5])
    np.testing.assert_allclose(yaw_local(quat, np.array([0.0, 1.0, 2.0])), [1, 0, 2], atol=1e-12)
    item = dict(
        ball_relative=np.array([1.0, 0, 0]),
        velocity_relative=np.zeros(3),
        joints=np.zeros(29),
        root_height=0.75,
    )
    result = match_error(item, item)
    assert result["geometry_state_compatible"] is True
    assert result["dynamic_feasibility_verified"] is False
    assert result["score"] == 0


def test_wrong_shape_and_nonfinite_rejected():
    item = dict(
        ball_relative=np.zeros(3),
        velocity_relative=np.zeros(3),
        joints=np.zeros(29),
        root_height=0.75,
    )
    bad = dict(item, joints=np.full(29, np.nan))
    with pytest.raises(ValueError):
        match_error(item, bad)


def test_contact_geometry_filter_excludes_far_or_penetrating_labels():
    def row(capture, gap, floor):
        return dict(
            capture=capture,
            event_frame=2,
            foot="left",
            samples=[{}, dict(foot_surface_gap_m=gap, ball_floor_clearance_m=floor), {}],
        )

    data = dict(samples=[row("valid", 0.005, -0.005), row("far", 0.02, 0), row("floor", 0, -0.02)])
    assert eligible_geometry_events(data) == {("valid", 2, "left")}
