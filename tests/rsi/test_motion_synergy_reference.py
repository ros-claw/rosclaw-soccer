import numpy as np
import pytest
from test_motion_reference_timing import policy

from rosclaw_soccer.rsi.motion_synergy_reference import (
    MotionSynergyPlan,
    condition_synergy_reference,
    fit_motion_synergies,
)


def plan(coeff=0.03):
    basis = tuple(tuple(float(i == j) for j in range(29)) for i in range(6))
    coefficients = ((coeff,) * 6, (-coeff,) * 6)
    return MotionSynergyPlan(20, basis, coefficients)


def test_fit_data_model_is_sign_canonical_and_recovers_low_rank_subspace():
    rng = np.random.default_rng(10)
    latent, decoder = rng.normal(size=(500, 6)), rng.normal(size=(6, 29))
    samples = (latent @ decoder).astype(np.float64)
    basis, report = fit_motion_synergies(samples)
    assert basis.shape == (6, 29)
    assert report["explained_variance_fraction"] == pytest.approx(1.0)
    assert all(row[np.argmax(abs(row))] == 1 for row in basis)
    assert np.linalg.matrix_rank(np.concatenate((basis, decoder))) == 6
    with pytest.raises(ValueError, match="rank"):
        fit_motion_synergies(np.zeros((100, 29)))


def test_smooth_lobes_preserve_entry_mid_exit_and_reference_anchor_bits():
    p = policy()
    before = {k: v.copy() for k, v in vars(p).items() if isinstance(v, np.ndarray)}
    original_time = p.time_step
    r = condition_synergy_reference(p, plan())
    assert r["applied"] and r["maximum_reference_offset_rad"] <= 0.06
    for k in ("motion_joint_pos", "motion_joint_vel"):
        for rows in (slice(None, 21), [44, 68], slice(68, None)):
            np.testing.assert_array_equal(getattr(p, k)[rows], before[k][rows])
    for k in ("motion_body_pos", "motion_body_quat"):
        np.testing.assert_array_equal(getattr(p, k), before[k])
    pos, vel = plan().offsets()
    assert pos[12, 0] == pytest.approx(0.03)
    assert pos[36, 0] == pytest.approx(-0.03)
    assert vel[6, 0] == pytest.approx(0.03 * np.pi / 24 / 0.02)
    assert p.time_step == original_time


def test_zero_branch_does_not_replace_original_arrays():
    p = policy()
    q, v = p.motion_joint_pos, p.motion_joint_vel
    assert condition_synergy_reference(p, plan(0.0))["zero_branch_exact"]
    assert p.motion_joint_pos is q and p.motion_joint_vel is v


def test_radial_projection_and_nonfinite_plan_rejection():
    repeated = tuple((1.0,) * 29 for _ in range(6))
    p = MotionSynergyPlan(20, repeated, ((0.06,) * 6, (0.06,) * 6))
    pos, vel = p.offsets()
    assert abs(pos).max() == pytest.approx(0.06)
    assert abs(vel).max() < 0.4
    with pytest.raises(ValueError, match="bounded"):
        plan(float("nan")).offsets()
    bad = policy()
    bad.time_step = 21
    with pytest.raises(ValueError, match="future"):
        condition_synergy_reference(bad, plan())
