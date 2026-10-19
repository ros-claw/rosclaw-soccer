"""Profiler bounds only; no factory or physics execution."""

import pytest

from scripts.rsi_profile_fixed_mean_constructor import require_scope


def scope():
    return dict(
        mode="SIM_ONLY",
        profile_seconds=180,
        hardware_authorized=False,
        promotion_authorized=False,
        training_authorized=False,
        stop_live_worker_authorized=False,
    )


def test_exact_isolated_profile():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_widened_missing_or_truthy_scope(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    cfg[key] = 180.0 if key == "profile_seconds" else "REAL" if key == "mode" else 0
    with pytest.raises(ValueError):
        require_scope(cfg)
