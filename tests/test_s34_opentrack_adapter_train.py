from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.evidence.opentrack_adapter_train import (
    StableAdapterTrainingOverrides,
)


def _overrides() -> StableAdapterTrainingOverrides:
    return StableAdapterTrainingOverrides(
        supervised_loss_weight=2e-4,
        entropy_cost=1e-3,
        policy_learning_rate=1e-5,
        world_model_learning_rate=1e-4,
        rehearsal_fraction=0.6,
        acquisition_fraction=0.4,
        maximum_world_steps=20_000_000,
    )


def test_stable_adapter_overrides_are_sim_only_and_balanced() -> None:
    overrides = _overrides()

    assert overrides.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="sum to one"):
        replace(overrides, rehearsal_fraction=0.7)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(overrides, activation_ceiling="HARDWARE")


def test_stable_adapter_overrides_reject_unbounded_or_zero_loss() -> None:
    overrides = _overrides()

    with pytest.raises(ValueError, match="positive"):
        replace(overrides, maximum_world_steps=0)
    with pytest.raises(ValueError, match="finite and positive"):
        replace(overrides, supervised_loss_weight=0.0)
