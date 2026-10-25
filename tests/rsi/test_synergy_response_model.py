import numpy as np
import pytest

from rosclaw_soccer.rsi.synergy_response_model import propose_synergy_response


def data(constant=False):
    rng = np.random.default_rng(12)
    basis = np.eye(6, 29, dtype=np.float64)
    samples = []
    for _ in range(80):
        c = rng.uniform(-0.04, 0.04, size=(2, 6))
        x = c.ravel() / 0.06
        forward, lateral = (0.7, 0.5) if constant else (0.7 + 0.7 * x[0], 0.4 - 0.3 * x[1])
        samples.append(
            (
                c,
                dict(
                    quality_observed=True,
                    predicted_body_healthy=True,
                    predicted_nonfoot_clean=True,
                    forward_60_m=forward,
                    lateral_60_m=lateral,
                    reward=forward - lateral,
                ),
            )
        )
    return basis, samples


def test_data_driven_constrained_response_has_no_physical_success_authority():
    basis, samples = data()
    proposal, report = propose_synergy_response(basis, samples)
    assert proposal is not None
    assert not report["physics_verified"] and not report["success_claim"]
    assert report["neural_updates"] == 0
    forward, lateral = report["predicted_ball_m"]
    assert forward >= 1.05 - 1e-7 and abs(lateral) <= 0.27 * forward + 1e-7
    x = np.stack([c for c, _ in samples])
    assert (proposal >= x.min(axis=0) - 1e-7).all()
    assert (proposal <= x.max(axis=0) + 1e-7).all()
    assert abs(proposal @ basis).max() <= 0.060001


def test_unreachable_local_response_abstains_instead_of_extrapolating():
    proposal, report = propose_synergy_response(*data(constant=True))
    assert proposal is None and "NO_LINEAR_FEASIBILITY" in report["status"]


def test_censored_and_failed_body_data_never_become_response_labels():
    basis, samples = data()
    for _, w in samples[:70]:
        w["quality_observed"] = False
        w["forward_60_m"] = None
    proposal, report = propose_synergy_response(basis, samples)
    assert proposal is None and report["complete_healthy_samples"] == 10
    samples[-1][0][0, 0] = np.nan
    with pytest.raises(ValueError, match="bounded"):
        propose_synergy_response(basis, samples)
