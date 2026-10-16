"""Byte identity and snapshot ownership; synthetic, not robot evidence."""

import pytest

import rosclaw_soccer.rsi.owned_recurrent_sampling_preview as preview_module
from rosclaw_soccer.rsi.recurrent_success_motor import make_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_complete_numeric_types_and_mutation_after_snapshot_are_not_grandfathered(
    imitation_parent,  # noqa: F811
    monkeypatch,
):
    compiler = preview_module.OwnedRecurrentSamplingPreview(make_model(imitation_parent))
    expected = compiler.preview_from_seed(seed=77)
    typed_change = compiler.sampling_view(seed=77)
    mean = typed_change["mean_model"]
    assert type(mean["parameters"]["head_bias"][0]) is float
    assert mean["parameters"]["head_bias"][0] == 0.0
    mean["parameters"]["head_bias"][0] = 0
    mean["model_hash"] = hash_json({k: v for k, v in mean.items() if k != "model_hash"})
    typed_change["model_hash"] = hash_json(
        {k: v for k, v in typed_change.items() if k != "model_hash"}
    )
    with pytest.raises(ValueError, match="actor/critic/parent"):
        compiler.preview(typed_change)

    incoming = compiler.sampling_view(seed=77)
    original_snapshot = preview_module.CanonicalJSONSnapshot

    def snapshot_then_mutate(value):
        snapshot = original_snapshot(value)
        if value is incoming:
            value["seed"] = 999
            value["mean_model"]["parameters"]["head_bias"][0] = 777.0
        return snapshot

    with monkeypatch.context() as patch:
        patch.setattr(preview_module, "CanonicalJSONSnapshot", snapshot_then_mutate)
        actual = compiler.preview(incoming)
    assert incoming["seed"] == 999
    assert actual == expected
    assert actual["step_motor_proof"]["model"]["seed"] == 77
    assert actual["policy_hash"] == hash_json(
        {k: v for k, v in actual.items() if k != "policy_hash"}
    )
