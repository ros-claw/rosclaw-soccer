import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.selective_phase_memory import (
    CompiledSelectivePhaseMemory,
    make_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_memory_guarded_phase_transfer import transfer  # noqa: F401
from tests.rsi.test_protected_phase_step_network import protected  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_explicit_new_width_leaves_inherited_artifact_and_anchors_unchanged(transfer):  # noqa: F811
    original = copy.deepcopy(transfer)
    value = make_model(transfer)
    decoder = CompiledSelectivePhaseMemory(make_preview(value))
    assert transfer == original
    assert value["transfer_model"] == original
    anchors = np.asarray(transfer["anchor_guard"]["anchors"])
    assert np.array_equal(decoder._guard.gates(anchors), np.zeros(len(anchors)))
    assert decoder._guard.bandwidth == 1e-4
    assert value["new_optimizer_steps"] == 0


def test_resealed_width_or_authority_change_is_not_the_committed_experiment(transfer):  # noqa: F811
    for field, item in (("bandwidth", 1e-3), ("hardware_authorized", True)):
        value = make_model(transfer)
        value[field] = item
        value["model_hash"] = hash_json({k: v for k, v in value.items() if k != "model_hash"})
        with pytest.raises(ValueError):
            validate_model(value)
