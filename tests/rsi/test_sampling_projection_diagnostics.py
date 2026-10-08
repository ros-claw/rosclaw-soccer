import numpy as np
import pytest

from rosclaw_soccer.rsi.sampling_projection_diagnostics import projection_diagnostics


def fixture(*, constrained=False):
    latent = np.zeros((300, 12))
    latent[30:, 0] = 1
    nominal = np.zeros((300, 12))
    bounds = np.tile([-2.0, 2.0], (12, 1))
    if constrained:
        bounds[0, 1] = 0.03
    delta = np.zeros_like(latent)
    previous = np.zeros(12)
    for frame in range(30, 300):
        desired = 0.16 * np.tanh(latent[frame])
        proposed = previous + np.clip(desired - previous, -0.012, 0.012)
        lower = np.maximum(np.minimum(0, bounds[:, 0] - nominal[frame]), -0.16)
        upper = np.minimum(np.maximum(0, bounds[:, 1] - nominal[frame]), 0.16)
        delta[frame] = np.clip(proposed, lower, upper)
        previous = delta[frame]
    return [latent, nominal, delta, bounds]


def run(values, **kw):
    return projection_diagnostics(
        *values, cap_rad=0.16, slew_rad=0.012, first_contact_frame=68, **kw
    )


def test_slew_and_joint_bounds_separate():
    ordinary = run(fixture())
    limited = run(fixture(constrained=True))
    assert ordinary["full_controlled"]["slew_changed_coordinate_rows"] > 0
    assert ordinary["full_controlled"]["joint_limit_changed_coordinate_rows"] == 0
    assert limited["full_controlled"]["joint_limit_changed_coordinate_rows"] > 0
    assert ordinary["before_first_contact"]["coordinate_rows"] == 38 * 12
    assert ordinary["first_contact_through_plus60"]["coordinate_rows"] == 61 * 12
    assert (
        sum(
            ordinary[k]["coordinate_rows"]
            for k in (
                "before_first_contact",
                "first_contact_through_plus60",
                "after_contact_horizon",
            )
        )
        == 3240
    )
    assert ordinary["projection_replay_max_abs_error_rad"] == 0
    assert ordinary["counterfactual_dynamics_claimed"] is False
    assert ordinary["runtime_execution_authorized"] is False


@pytest.mark.parametrize("index,value", [(0, np.nan), (1, np.inf), (2, np.nan), (3, np.inf)])
def test_nonfinite_rejected(index, value):
    values = fixture()
    values[index].flat[0] = value
    with pytest.raises(ValueError):
        run(values)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_partial_or_misaligned_array_rejected(index):
    values = fixture()
    values[index] = values[index][:-1]
    with pytest.raises(ValueError):
        run(values)


def test_actual_projection_mismatch_rejected():
    values = fixture()
    values[2][60, 3] = 0.001
    with pytest.raises(ValueError, match="EXACTLY"):
        run(values)


def test_predecision_motion_rejected():
    values = fixture()
    values[2][29, 0] = 0.001
    with pytest.raises(ValueError, match="pre-control"):
        run(values)


def test_unordered_joint_bounds_rejected():
    values = fixture()
    values[3][2] = [2, -2]
    with pytest.raises(ValueError, match="ordered"):
        run(values)


@pytest.mark.parametrize("first", [None, 0, 299])
def test_empty_intervals_and_no_contact(first):
    r = projection_diagnostics(*fixture(), cap_rad=0.16, slew_rad=0.012, first_contact_frame=first)
    assert (
        sum(
            r[k]["coordinate_rows"]
            for k in (
                "before_first_contact",
                "first_contact_through_plus60",
                "after_contact_horizon",
            )
        )
        == 3240
    )
    if first is None:
        assert r["first_contact_through_plus60"]["desired_to_actual_rms_rad"] is None
    elif first == 0:
        assert r["first_contact_through_plus60"]["coordinate_rows"] == 31 * 12


@pytest.mark.parametrize(
    "cap,slew,first",
    [
        (True, 0.012, 68),
        (0.16, 0, 68),
        (0.16, 0.2, 68),
        (0.16, 0.012, True),
        (0.16, 0.012, 300),
        (0.16, np.nan, 68),
    ],
)
def test_invalid_envelope_or_event(cap, slew, first):
    with pytest.raises(ValueError):
        projection_diagnostics(*fixture(), cap_rad=cap, slew_rad=slew, first_contact_frame=first)
