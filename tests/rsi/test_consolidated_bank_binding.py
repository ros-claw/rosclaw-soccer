"""Qualification adapter fixtures, not learned-policy or physical evidence."""

import pytest

from rosclaw_soccer.rsi import consolidated_smooth_motor
from scripts.rsi_collect_protected_phase_bank_validation import validate_bank_models


def candidate(parent: dict) -> dict:
    return {
        "schema": "soccer.rsi.consolidated_smooth_motor.v1",
        "base_model": {"frozen_parent": parent, "model_hash": "rejected-ar"},
    }


def test_exact_neural_parent_required(monkeypatch) -> None:
    monkeypatch.setattr(consolidated_smooth_motor, "validate_model", lambda _: None)
    parent = {"schema": "soccer.rsi.output_memory_step_motor.v1", "model_hash": "qualified-nn"}
    child = candidate(parent)
    validate_bank_models(child, dict(parent))
    with pytest.raises(ValueError, match="exact qualified"):
        validate_bank_models(child, child["base_model"])
    with pytest.raises(ValueError, match="exact qualified"):
        validate_bank_models(child, {**parent, "model_hash": "stale-nn"})


def test_consolidation_validation_is_not_bypassed(monkeypatch) -> None:
    def reject(_):
        raise ValueError("unsealed memory")

    monkeypatch.setattr(consolidated_smooth_motor, "validate_model", reject)
    with pytest.raises(ValueError, match="unsealed"):
        validate_bank_models(candidate({}), {})
