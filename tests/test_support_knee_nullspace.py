"""Support-knee retreat must be causal, bounded, and support-foot preserving."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.support_knee_nullspace import support_knee_nullspace_delta


def _geometry() -> tuple[np.ndarray, np.ndarray]:
    foot = np.zeros((3, 6), dtype=np.float64)
    foot[:, :3] = np.eye(3)
    knee = np.zeros(6, dtype=np.float64)
    knee[3] = 1.0
    return foot, knee


BASELINE = np.zeros(6, dtype=np.float64)
LIMITS = np.tile(np.array((-1.0, 1.0), dtype=np.float64), (6, 1))


def test_measured_nullspace_moves_knee_but_not_support_foot() -> None:
    foot, knee = _geometry()
    proposal = support_knee_nullspace_delta(
        foot,
        knee,
        BASELINE,
        LIMITS,
        retract_m=0.04,
        support_foot_grounded=True,
        swing_foot_airborne=True,
    )
    assert not proposal.abstained
    assert proposal.predicted_knee_retract_m == pytest.approx(0.04, abs=1e-3)
    assert proposal.predicted_foot_drift_m == pytest.approx(0.0, abs=1e-8)
    assert np.max(np.abs(proposal.joint_delta_rad)) <= 0.18


@pytest.mark.parametrize("grounded,airborne", [(False, True), (True, False), (False, False)])
def test_unverified_support_abstains(grounded: bool, airborne: bool) -> None:
    foot, knee = _geometry()
    proposal = support_knee_nullspace_delta(
        foot,
        knee,
        BASELINE,
        LIMITS,
        retract_m=0.08,
        support_foot_grounded=grounded,
        swing_foot_airborne=airborne,
    )
    assert proposal.abstained and proposal.joint_delta_rad == (0.0,) * 6


def test_no_independent_knee_direction_abstains() -> None:
    foot, _ = _geometry()
    proposal = support_knee_nullspace_delta(
        foot,
        foot[0].copy(),
        BASELINE,
        LIMITS,
        retract_m=0.04,
        support_foot_grounded=True,
        swing_foot_airborne=True,
    )
    assert proposal.abstained


def test_joint_delta_cap_is_enforced() -> None:
    foot, knee = _geometry()
    knee[3] = 0.1
    proposal = support_knee_nullspace_delta(
        foot,
        knee,
        BASELINE,
        LIMITS,
        retract_m=0.08,
        support_foot_grounded=True,
        swing_foot_airborne=True,
    )
    assert not proposal.abstained
    assert max(abs(value) for value in proposal.joint_delta_rad) == pytest.approx(0.18)
    assert proposal.predicted_knee_retract_m < 0.08


@pytest.mark.parametrize("retract", [0.03, 0.081, float("nan"), True])
def test_invalid_retract_fail_closed(retract: float) -> None:
    foot, knee = _geometry()
    with pytest.raises(ValueError):
        support_knee_nullspace_delta(
            foot,
            knee,
            BASELINE,
            LIMITS,
            retract_m=retract,
            support_foot_grounded=True,
            swing_foot_airborne=True,
        )


def test_joint_limit_violation_abstains_as_a_whole_action() -> None:
    foot, knee = _geometry()
    limits = LIMITS.copy()
    limits[3, 0] = -0.01
    proposal = support_knee_nullspace_delta(
        foot,
        knee,
        BASELINE,
        limits,
        retract_m=0.04,
        support_foot_grounded=True,
        swing_foot_airborne=True,
    )
    assert proposal.abstained and proposal.joint_delta_rad == (0.0,) * 6
