import numpy as np
import pytest

from rosclaw_soccer.rsi.native_contact_window import native_contact_window


def inputs(first=100, length=1000):
    q = np.zeros((length, 43))
    q[:, 2] = 0.8
    q[:, 3] = 1
    q[:, 36] = np.arange(length) * 0.002
    force = np.zeros((length, 2))
    if first is not None:
        force[first, 0] = 11
    return dict(
        force=force,
        canonical_qpos=q,
        canonical_qvel=np.zeros((length, 41)),
        goal=np.array([10.0, 0.0, 2.0]),
        intended_foot=0,
    )


@pytest.mark.parametrize("first,observed", [(399, True), (400, False), (999, False)])
def test_exact_post_contact_boundary(first, observed):
    r = native_contact_window(**inputs(first))
    assert r["post_contact_quality_observed"] is observed
    assert r["correct_foot_first"]
    if observed:
        assert r["goal_progress_1p2s_m"] == pytest.approx(1.2)
        assert r["goal_lateral_1p2s_m"] == 0
    else:
        assert r["goal_progress_1p2s_m"] is None
        assert r["goal_lateral_1p2s_m"] is None
        assert r["other_foot_contact_in_window"] is None


def test_no_contact_has_no_fabricated_quality():
    r = native_contact_window(**inputs(None))
    assert r["post_contact_status"] == "NO_FOOT_CONTACT"
    assert r["first_foot_contact_substep"] is None
    assert r["goal_progress_1p2s_m"] is None
    assert r["maximum_pelvis_retreat_after_contact_m"] is None


def test_wrong_and_ambiguous_first_contact_never_credited():
    values = inputs()
    values["force"][100, 1] = 12
    r = native_contact_window(**values)
    assert not r["correct_foot_first"]
    assert r["other_foot_contact_in_window"]
    values["force"][100, 0] = 0
    assert not native_contact_window(**values)["correct_foot_first"]


def test_pelvis_retreat_uses_actual_target_direction_and_state():
    values = inputs()
    values["canonical_qpos"][100:, 0] = np.linspace(0, -0.2, 900)
    r = native_contact_window(**values)
    assert r["maximum_pelvis_retreat_after_contact_m"] == pytest.approx(0.2)
    assert not r["causal_kick_proven"] and not r["football_skill_proven"]


@pytest.mark.parametrize("key", ["force", "canonical_qpos", "canonical_qvel", "goal"])
def test_nonfinite_measurements_rejected(key):
    values = inputs()
    values[key].flat[0] = np.nan
    with pytest.raises(ValueError):
        native_contact_window(**values)
