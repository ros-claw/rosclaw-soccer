from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.training.role_isolated_contact_failure_update import (
    RoleIsolatedContactFailureUpdateConfig,
)


def test_failure_update_is_a_small_sim_only_trust_region() -> None:
    config = RoleIsolatedContactFailureUpdateConfig()

    assert config.local_plasticity_gain == 0.05
    assert config.proprioceptive_feedback_gain_n_per_mps == 6.0
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.hardware_authorized is False
    with pytest.raises(ValueError, match="trust region"):
        replace(config, local_plasticity_gain=0.25)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)


def test_failure_update_requires_a_named_task_failure() -> None:
    with pytest.raises(ValueError, match="unique task-level"):
        RoleIsolatedContactFailureUpdateConfig(required_failed_gates=())
    with pytest.raises(ValueError, match="unique task-level"):
        RoleIsolatedContactFailureUpdateConfig(
            required_failed_gates=("outward_physical_save", "outward_physical_save")
        )
