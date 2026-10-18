"""Synthetic input contracts only; no NN/physics/teacher qualification."""

import copy
import json

import pytest

from rosclaw_soccer.rsi import positive_lateral_episode_inputs as module
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.sim.contracts import hash_json


def fixture(tmp_path, monkeypatch):
    behavior = "sha256:" + "a" * 64
    payload = "sha256:" + "b" * 64
    scope = build_scope(behavior)
    views, hashes = {}, {str(tmp_path / "parent"): "old-bytes"}
    rows = [
        dict(draw=d, witnesses={}, metadata_only_no_physical_trajectory=True)
        for d in scope["source_plan"]["jobs"]
    ]
    for c in scope["contexts"]:
        path = tmp_path / "models" / f"view-{c['source_group']}.json.gz"
        hashes[str(path)] = f"view-bytes-{c['source_group']}"
        rows[c["source_group"]]["witnesses"]["sampling_view"] = dict(
            path=str(path), sha256=hashes[str(path)]
        )
        views[str(path)] = dict(
            schema="soccer.rsi.recurrent_motor_sampling.v1",
            seed=c["sampling_seed"],
            mean_model={"$rosclaw_shared_payload": payload},
            std_raw=0.1,
            rho=0.9,
            training_only=True,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            runtime_execution_authorized=False,
            hardware_authorized=False,
            fresh_holdout_open_authorized=False,
        )
    mean_path = str(tmp_path / ".shared-models" / "mean.json.gz")
    hashes[mean_path] = "mean-bytes"
    reg = dict(
        schema="soccer.rsi.positive_lateral_sampling_registry.v1",
        scope=scope,
        scope_hash=scope["scope_hash"],
        records=rows,
        input_file_hashes={str(tmp_path / "parent"): "old-bytes"},
        shared_mean=dict(
            path=mean_path, sha256="mean-bytes", model_hash=behavior, payload_hash=payload
        ),
        declared_physical_source_groups=[c["source_group"] for c in scope["contexts"]],
        source_view_count=8,
        metadata_registry_rows=96,
        source_registry_fillers_are_not_physical_trials=True,
        all_old_receding_failures_retained_not_relabelled=True,
        new_nn_model_allocations=0,
        new_neural_inferences=0,
        new_physical_steps=0,
        new_optimizer_updates=0,
        execution_authorized=False,
        dataset_admission_authorized=False,
        teacher_qualified=False,
        fresh_exam=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    reg["report_hash"] = hash_json(reg)
    monkeypatch.setattr(module, "logical_document", lambda p: copy.deepcopy(views[str(p)]))
    # Scientific inputs are persisted JSON: course tuples become lists. The
    # complete canonical JSON identity must survive this real serialization.
    return json.loads(json.dumps(reg)), behavior, hashes, views


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "partial",
        "order",
        "filler",
        "domain",
        "view_bytes",
        "parent_bytes",
        "seed",
        "mean",
        "law",
        "hardware",
        "truthy",
    ],
)
def test_all_registry_inputs_required_before_physical_allocation(tmp_path, monkeypatch, tamper):
    reg, behavior, hashes, views = fixture(tmp_path, monkeypatch)
    if tamper == "partial":
        reg["records"].pop()
    elif tamper == "order":
        reg["records"].reverse()
    elif tamper == "filler":
        reg["records"][0]["witnesses"] = {"sampling_view": {}}
    elif tamper == "domain":
        reg["scope"]["contexts"][0]["course"][0] = 1.0
    elif tamper == "view_bytes":
        hashes[next(iter(views))] = "changed"
    elif tamper == "parent_bytes":
        hashes[str(tmp_path / "parent")] = "changed"
    elif tamper == "seed":
        views[next(iter(views))]["seed"] += 1
    elif tamper == "mean":
        views[next(iter(views))]["mean_model"] = {"$rosclaw_shared_payload": "other"}
    elif tamper == "law":
        views[next(iter(views))]["rho"] = 0.5
    elif tamper == "hardware":
        reg["hardware_authorized"] = True
    elif tamper == "truthy":
        reg["new_optimizer_updates"] = False
    # Even resealing cannot bypass deterministic domain and content guards.
    reg["report_hash"] = hash_json({k: v for k, v in reg.items() if k != "report_hash"})
    pins = {}
    args = dict(
        expected_registry_hash=reg["report_hash"],
        behavior_model_hash=behavior,
        digest=lambda p: hashes[str(p)],
        pins=pins,
    )
    if tamper:
        with pytest.raises(ValueError):
            module.episode_inputs(reg, **args)
    else:
        plan, manifest, contexts = module.episode_inputs(reg, **args)
        assert len(manifest["records"]) == 96 and len(contexts) == 8
        assert hash_json(plan.contract()) == hash_json(reg["scope"]["source_plan"])
        assert len(pins) == 10


def test_prospective_split_keeps_perturbations_and_controller_families_together():
    scope = build_scope("sha256:" + "a" * 64)
    split = module.prospective_split(scope)
    assert len(split["rows"]) == 8
    assert sum(r["partition"] == "train" for r in split["rows"]) == 6
    assert sum(r["partition"] == "validation" for r in split["rows"]) == 2
    for seed in scope["source_seed_search"]["selected_seeds"]:
        assert len({r["partition"] for r in split["rows"] if r["source_seed"] == seed}) == 1
    assert not split["fresh_exam"] and not split["neural_training_authorized"]
    assert not split["global_ancestry_separation_proven"]
    assert (
        split["controller_pairs_share_partition"] and split["training_only_normalization_required"]
    )


def test_registry_seal_is_bound_to_predeclared_identity(tmp_path, monkeypatch):
    reg, behavior, hashes, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="whole original"):
        module.episode_inputs(
            reg,
            expected_registry_hash="other",
            behavior_model_hash=behavior,
            digest=lambda p: hashes[str(p)],
            pins={},
        )
