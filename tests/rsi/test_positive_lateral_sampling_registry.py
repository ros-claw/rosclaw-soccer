import copy

import pytest
from rosclaw.growth.shared_proof_payload import detach_payload, restore_payload

from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.rsi.positive_lateral_sampling_registry import registry_views
from rosclaw_soccer.sim.contracts import hash_json


def source():
    # Synthetic numeric transport fixture: not a trained or physically qualified model.
    mean = dict(schema="synthetic_transport_test", weight=[0.25, -0.5])
    mean["model_hash"] = hash_json(mean)
    view = dict(
        schema="soccer.rsi.recurrent_motor_sampling.v1",
        mean_model=mean,
        seed=12,
        std_raw=0.1,
        rho=0.9,
        activation_ceiling="SIM_ONLY",
        training_only=True,
        promotion_authorized=False,
        runtime_execution_authorized=False,
        hardware_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    view["model_hash"] = hash_json(view)
    envelope, _ = detach_payload(view, ("mean_model",))
    return mean, view, envelope


def test_eight_new_views_restore_unchanged_whole_mean_and_only_new_sampling_seed():
    mean, original, envelope = source()
    scope = build_scope(mean["model_hash"])
    snapshots = copy.deepcopy((mean, original, envelope, scope))
    views = registry_views(scope, envelope, mean)
    assert list(views) == [12, 13, 28, 29, 76, 77, 92, 93]
    for context in scope["contexts"]:
        value = restore_payload(views[context["source_group"]], mean)
        expected = dict(original, seed=context["sampling_seed"])
        expected["model_hash"] = hash_json({k: v for k, v in expected.items() if k != "model_hash"})
        assert value == expected
    assert (mean, original, envelope, scope) == snapshots


@pytest.mark.parametrize("tamper", ["mean", "scope", "envelope", "law", "authority"])
def test_do_not_relabel_a_changed_policy_or_partial_scope_as_source_transport(tamper):
    mean, view, envelope = source()
    scope = build_scope(mean["model_hash"])
    if tamper == "mean":
        mean["weight"][0] += 1
    elif tamper == "scope":
        scope["contexts"].pop()
    elif tamper == "envelope":
        envelope["stripped_document"]["seed"] = 13
    else:
        view["std_raw" if tamper == "law" else "hardware_authorized"] = (
            0.2 if tamper == "law" else True
        )
        view["model_hash"] = hash_json({k: v for k, v in view.items() if k != "model_hash"})
        envelope, _ = detach_payload(view, ("mean_model",))
    with pytest.raises(ValueError):
        registry_views(scope, envelope, mean)
