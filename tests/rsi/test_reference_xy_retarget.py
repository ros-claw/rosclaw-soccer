import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_xy_retarget import ball_strike_xy_plan


def fixture():
    t = np.arange(200) * 0.02
    x = t + 0.2 * np.tanh(4 * (t - 2))
    return dict(
        capsule_centers=np.column_stack((x, np.zeros(200), np.full(200, 0.03))),
        capsule_axes=np.tile([1.0, 0.0, 0.0], (200, 1)),
        capsule_half_length=0.093,
        capsule_radius=0.01,
        ball_radius=0.11,
        initial_ball=np.array([4.0, 0.5, 0.11]),
        goal=np.array([8.0, 0.5, 1.0]),
    )


def test_rigid_translation_starts_without_a_jump_and_preserves_height():
    values = fixture()
    original = values["capsule_centers"].copy()
    r = ball_strike_xy_plan(**values)
    np.testing.assert_array_equal(values["capsule_centers"], original)
    np.testing.assert_array_equal(r["offset"][0], 0)
    np.testing.assert_array_equal(r["offset_velocity"][0], 0)
    assert np.linalg.norm(r["applied_offset_xy_m"]) == pytest.approx(0.3)
    np.testing.assert_allclose(r["offset"][50:, :2], np.tile(r["applied_offset_xy_m"], (150, 1)))
    assert not r["offset"][:, 2].any()
    assert not r["offset_velocity"][50:].any()


def test_world_translation_does_not_change_relative_retarget():
    values = fixture()
    a = ball_strike_xy_plan(**values)
    delta = np.array([30.0, -5.0, 0.0])
    for key in ("capsule_centers", "initial_ball", "goal"):
        values[key] += delta
    b = ball_strike_xy_plan(**values)
    assert a["strike_reference_frame"] == b["strike_reference_frame"] == 100
    np.testing.assert_allclose(a["offset"], b["offset"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(a["offset_velocity"], b["offset_velocity"], atol=1e-12, rtol=0)


@pytest.mark.parametrize("bad", ["axis", "height", "nan"])
def test_invalid_or_unreachable_reference_rejected(bad):
    values = fixture()
    if bad == "axis":
        values["capsule_axes"] *= 2
    elif bad == "height":
        values["capsule_centers"][:, 2] = 1
    else:
        values["initial_ball"][0] = np.nan
    with pytest.raises(ValueError):
        ball_strike_xy_plan(**values)
