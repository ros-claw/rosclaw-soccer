from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.evidence.opentrack_adapter_audit import (
    compare_opentrack_policy_parameters,
)


def _layer(value: float, shape: tuple[int, ...] = (2, 2)) -> dict[str, np.ndarray]:
    return {
        "kernel": np.full(shape, value, dtype=np.float32),
        "bias": np.full((shape[-1],), value, dtype=np.float32),
    }


def test_parameter_audit_proves_frozen_base_and_counts_adapter() -> None:
    parent = {"hidden_0": _layer(1.0), "hidden_1": _layer(2.0)}
    candidate = {**parent, "adapter_0": _layer(0.1), "adapter_1": _layer(0.2)}

    result = compare_opentrack_policy_parameters(parent_policy=parent, candidate_policy=candidate)

    assert result["frozen_base_hash_before"] == result["frozen_base_hash_after"]
    assert result["maximum_frozen_parameter_drift"] == 0.0
    assert result["examined_frozen_parameter_count"] == 12
    assert result["examined_trainable_parameter_count"] == 12


def test_parameter_audit_rejects_hidden_topology_change() -> None:
    with pytest.raises(ValueError, match="topology"):
        compare_opentrack_policy_parameters(
            parent_policy={"hidden_0": _layer(1.0)},
            candidate_policy={"hidden_1": _layer(1.0), "adapter_0": _layer(0.0)},
        )
