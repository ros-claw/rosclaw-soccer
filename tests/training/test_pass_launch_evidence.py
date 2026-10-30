import numpy as np
import pytest

from rosclaw_soccer.training.pass_launch_evidence import inspect_pass_launch


def arguments():
    return {
        "control_time_sec": np.array([1.0, 1.02, 1.04]),
        "source_codes": np.array([1, 1, 0]),
        "receiver_codes": np.array([2, 2, 0]),
        "sender_code": 1,
        "receiver_code": 2,
        "contact_time_sec": 1.03,
        "ball_xy_m": (0.0, 0.0),
        "ball_velocity_xy_mps": (1.0, 0.0),
        "receiver_position_xy_m": (1.0, 0.0),
    }


def test_committed_forward_launch_observed_not_promoted():
    r = inspect_pass_launch(**arguments())
    assert r.directed_launch_observed and not r.promotion_authorized


def test_prior_commitment_not_reused_after_withdrawal():
    args = arguments()
    args["contact_time_sec"] = 1.05
    r = inspect_pass_launch(**args)
    assert not r.directed_launch_observed
    assert "no_matching_precontact_commitment" in r.reasons


def test_future_commitment_never_used():
    args = arguments()
    args["contact_time_sec"] = 1.0
    assert not inspect_pass_launch(**args).commitment_visible_before_touch


@pytest.mark.parametrize("velocity", [(0.0, 0.0), (-1.0, 0.0), (0.39, 0.0)])
def test_weak_or_wrong_direction_not_a_launch(velocity):
    args = arguments()
    args["ball_velocity_xy_mps"] = velocity
    assert not inspect_pass_launch(**args).directed_launch_observed


def test_missing_direction_and_nonfinite_input_fail_closed():
    args = arguments()
    args["receiver_position_xy_m"] = (0.0, 0.0)
    assert not inspect_pass_launch(**args).directed_launch_observed
    args["ball_velocity_xy_mps"] = (float("nan"), 0.0)
    with pytest.raises(ValueError):
        inspect_pass_launch(**args)


def test_stale_snapshot_rejected():
    args = arguments()
    args["contact_time_sec"] = 1.2
    assert not inspect_pass_launch(**args).commitment_visible_before_touch
