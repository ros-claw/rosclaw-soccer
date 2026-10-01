import copy

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_sampling_view_memory import bounded_views


def factory(base, *, seed, std):
    view = dict(
        schema="soccer.rsi.smooth_memory_sampling.v1",
        mean_model=copy.deepcopy(base),
        seed=seed,
        std_raw=std,
    )
    view["model_hash"] = hash_json(view)
    return view


def test_every_complete_sampling_payload_and_identity_remains_exact():
    base = dict(weights=list(range(2000)), frozen_parent=dict(memory=[0.1, 0.2]))
    expected = [factory(base, seed=i, std=0.1) for i in range(120)]
    actual = bounded_views(factory, base, range(120))
    assert actual == expected
    assert all(v["mean_model"] is base for v in actual)
    assert len({id(v["mean_model"]) for v in expected}) == 120
    assert all(
        v["model_hash"] == hash_json({k: x for k, x in v.items() if k != "model_hash"})
        for v in actual
    )


def test_constructor_mean_changes_are_rejected():
    def changed(base, **kwargs):
        value = factory(base, **kwargs)
        value["mean_model"]["weight"] = 999
        return value

    with pytest.raises(ValueError, match="complete declared mean"):
        bounded_views(changed, dict(weight=1), [1])


def test_output_memory_schema_is_also_supported_without_payload_changes():
    def output_factory(base, **kwargs):
        value = factory(base, **kwargs)
        value["schema"] = "soccer.rsi.output_memory_step_sampling.v1"
        value.pop("model_hash")
        value["model_hash"] = hash_json(value)
        return value

    base = dict(weight=1)
    assert bounded_views(output_factory, base, [1]) == [output_factory(base, seed=1, std=0.1)]


def test_wrong_serialized_hash_is_rejected():
    def changed(base, **kwargs):
        value = factory(base, **kwargs)
        value["model_hash"] = "sha256:" + "0" * 64
        return value

    with pytest.raises(ValueError, match="policy identity"):
        bounded_views(changed, dict(weight=1), [1])


def test_constructor_mutation_of_shared_mean_is_rejected():
    def changed(base, **kwargs):
        base["weight"] += 1
        return factory(base, **kwargs)

    with pytest.raises(ValueError, match="immutable mean"):
        bounded_views(changed, dict(weight=1), [1])
