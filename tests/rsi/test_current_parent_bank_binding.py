import copy

import pytest

from rosclaw_soccer.rsi import (
    kernel_guarded_step_network,
    output_memory_step_motor,
    smooth_memory_motor,
)
from scripts.rsi_collect_protected_phase_bank_validation import validate_bank_models


@pytest.fixture
def lineage(monkeypatch):
    for module in (kernel_guarded_step_network, output_memory_step_motor, smooth_memory_motor):
        monkeypatch.setattr(module, "validate_model", lambda model: None)
    parent = dict(schema="soccer.rsi.kernel_guarded_step_actor_critic.v1", model_hash="kernel")
    zero = dict(
        schema="soccer.rsi.output_memory_step_motor.v1",
        model_hash="zero",
        generation=0,
        frozen_parent=parent,
        output_memory=dict(memory_hash="retained"),
        learning_receipt=None,
    )
    learned = copy.deepcopy(zero)
    learned.update(
        model_hash="learned", generation=1, learning_receipt=dict(learner_parent_hash="zero")
    )
    return parent, zero, learned


def test_later_memory_learning_checks_current_parent_not_just_encoder(lineage):
    parent, zero, learned = lineage
    validate_bank_models(learned, parent)
    validate_bank_models(learned, zero)
    stale = dict(parent, model_hash="old-kernel")
    with pytest.raises(ValueError, match="current learned parent"):
        validate_bank_models(learned, stale)
    other = dict(zero, model_hash="other-zero")
    with pytest.raises(ValueError, match="current learned parent"):
        validate_bank_models(learned, other)


def test_success_memory_and_generation_cannot_change_during_current_parent_comparison(lineage):
    _, zero, learned = lineage
    for key, value in (("generation", 2), ("output_memory", dict(memory_hash="different"))):
        changed = dict(learned, **{key: value})
        with pytest.raises(ValueError):
            validate_bank_models(changed, zero)


def test_smooth_actor_cannot_fall_back_to_an_older_nonmemory_parent(lineage):
    parent, zero, _ = lineage
    child = dict(
        schema="soccer.rsi.smooth_memory_motor.v1",
        model_hash="smooth-zero",
        generation=0,
        frozen_parent=zero,
        learning_receipt=None,
    )
    validate_bank_models(child, zero)
    with pytest.raises(ValueError, match="current learned parent"):
        validate_bank_models(child, parent)
    updated = dict(
        child,
        model_hash="smooth-learned",
        generation=1,
        learning_receipt=dict(learner_parent_hash="smooth-zero"),
    )
    validate_bank_models(updated, child)
    with pytest.raises(ValueError):
        validate_bank_models(updated, dict(child, model_hash="another-child"))
