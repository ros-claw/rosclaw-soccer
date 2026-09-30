import copy

import pytest

from rosclaw_soccer.rsi import motor_bootstrap_network as base_module
from rosclaw_soccer.rsi.core_online_growth import online_growth_payload
from rosclaw_soccer.rsi.online_motor_actor_critic import make_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_motor_bootstrap_network import bank


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    base = base_module.fit_bootstrap(bank(), epochs=2)
    return make_model(base, [[i / 12] * 13 for i in range(9)], "sha256:" + "a" * 64)


def review(model, clean_loss=0):
    score = dict(
        predecessor_high_quality_loss=0,
        raw_gain_high_quality_loss=0,
        old_high_quality_loss=0,
        safe_pelvis_guardrail=True,
        old_clean_foot_loss=clean_loss,
        new_out_of_play=0,
        high_quality_count=11,
    )
    result = dict(
        training_complete=True,
        promotion_authorized=False,
        fresh_holdout_open_authorized=False,
        best_model_hash=model["model_hash"],
        body_hash="sha256:" + "b" * 64,
        generations=[
            dict(model_hash=model["model_hash"], score=score, previous_nine_high_quality_loss=0)
        ],
    )
    result["report_hash"] = hash_json(result)
    return result


def test_consumed_pass_still_cannot_fill_fresh_and_cpu_gates(model):
    result = online_growth_payload(review(model), model)
    manifest = result["consolidation_manifest"]
    assert manifest["decision"] == "need_more_evidence"
    assert sum(g["status"] == "missing" for g in manifest["gate_results"]) == 2
    assert result["evidence_use_policy"]["promotion_truth_allowed"] is False
    assert result["registry_write_count"] == 0


def test_dirty_contact_fails_core_safety_and_rejects_candidate(model):
    result = online_growth_payload(review(model, clean_loss=1), model)
    assert result["consolidation_manifest"]["decision"] == "reject"


def test_missing_or_tampered_review_cannot_export(model):
    value = review(model)
    value["training_complete"] = False
    value["report_hash"] = hash_json({k: v for k, v in value.items() if k != "report_hash"})
    with pytest.raises(ValueError):
        online_growth_payload(value, model)
    altered = copy.deepcopy(review(model))
    altered["generations"][0]["score"]["high_quality_count"] += 1
    with pytest.raises(ValueError):
        online_growth_payload(altered, model)
