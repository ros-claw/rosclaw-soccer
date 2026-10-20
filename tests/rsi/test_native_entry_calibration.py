from dataclasses import replace

import numpy as np
import pytest
from test_motion_entry_profile import fixture

from rosclaw_soccer.rsi.native_entry_calibration import bind_native_entry, fit_displacements


def sample_fit(b, e, samples=None):
    return fit_displacements(
        np.array([[0.20, -0.15], [0.21, -0.16]]) if samples is None else samples,
        configuration_hash=b.configuration.contract_hash,
        parent_affordance_hash=e.contract_hash,
        audit_hash="sha256:" + "a" * 64,
        sample_hashes=("sha256:" + "b" * 64, "sha256:" + "c" * 64),
    )


def test_fit_mean_and_residual_bind_only_this_configuration():
    b, e, p, _, _, kw = fixture(transport=False)
    fit = sample_fit(b, e)
    np.testing.assert_allclose(fit.mean_displacement_xy_m, [0.205, -0.155])
    assert fit.maximum_training_residual_m == pytest.approx(np.sqrt(2) * 0.005)
    bound, entry = bind_native_entry(b, e, fit)
    np.testing.assert_allclose(entry.ball_offset_xy_m, [0.405, -0.125])
    assert entry.historical_artifact_hash == e.historical_artifact_hash
    assert entry.calibration_hash == fit.contract_hash
    assert not fit.report()["dynamically_qualified"]
    bound.validate(
        p, entry, world_hash=kw["world_hash"], source_window=(466, 767), reference_side="right"
    )
    with pytest.raises(ValueError):
        bind_native_entry(bound, entry, fit)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((3, 2)),
        np.zeros((2, 2), dtype=np.float32),
        np.array([[np.nan, 0.0], [0.2, 0.1]]),
        np.array([[0.2, -0.1], [0.4, -0.1]]),
        np.array([[0.5, -0.1], [0.5, -0.1]]),
    ],
)
def test_bad_data_rejected_before_body_allocation(samples):
    b, e, *_ = fixture(transport=False)
    with pytest.raises(ValueError):
        sample_fit(b, e, samples)


@pytest.mark.parametrize("change", ["world", "parent", "audit", "samples", "side", "transported"])
def test_lineage_and_domain_fail_closed(change):
    b, e, *_ = fixture(transport=False)
    fit = sample_fit(b, e)
    with pytest.raises(ValueError):
        if change == "world":
            fit = replace(fit, motion_configuration_hash="sha256:" + "f" * 64)
        elif change == "parent":
            fit = replace(fit, parent_affordance_hash="sha256:" + "f" * 64)
        elif change == "audit":
            fit = replace(fit, audit_hash="invalid")
        elif change == "samples":
            fit = replace(fit, sample_hashes=("sha256:" + "b" * 64,) * 2)
        elif change == "side":
            b, e, *_ = fixture(side="left", transport=False)
        else:
            b, e, *_ = fixture(transport=True)
        bind_native_entry(b, e, fit)
