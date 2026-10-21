import json

import pytest
from test_paid_fresh_protocol import source_cases

from rosclaw_soccer.rsi.paid_fresh_evidence import digest, load_fresh_declaration
from rosclaw_soccer.rsi.paid_fresh_protocol import PROTOCOL, fresh_lifecycle_cases


def fixture():
    cases = fresh_lifecycle_cases(source_cases())
    declaration = dict(
        schema=PROTOCOL,
        source_head="h",
        cases=cases,
        arms=["r2", "parent", "progressive"],
        repeats=2,
        skills=3,
        standing_controls=100,
        body_resets=0,
        candidate_seed_index=0,
        candidate_frozen_before_generation=True,
        candidate_seed_chosen_on_fresh=False,
        case_generation_seed=2026102207,
        Fresh_episodes_fit=0,
        evaluation=dict(
            candidate_stable_required=32,
            no_case_regression_vs_R2=True,
            no_case_regression_vs_parent=True,
            third_gain_vs_R2_minimum=4,
            third_gain_vs_parent_minimum=1,
            all_independent_audits_required=True,
        ),
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report = dict(
        status="PAID_FRESH_ALL32_CASES_DECLARED_AND_PREFLIGHT_PASS_NO_ROLLOUT",
        source_head="h",
        protocol=PROTOCOL,
        own_native_steps=0,
        Fresh_episodes_executed=0,
        Fresh_episodes_fit=0,
        native_entries=[
            dict(
                case=c["case"],
                position=c["ball_world_initial"],
                velocity=c["velocity_world_initial"],
            )
            for c in cases
        ],
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return declaration, report


def publish(root, d, r):
    path = root / "declaration.json"
    path.write_text(json.dumps(d))
    r["declaration_hash"] = digest(path)
    path = root / "complete.json"
    path.write_text(json.dumps(r))
    return digest(path)


def test_exact_new_context_declaration_can_be_admitted_without_rollouts(tmp_path):
    d, r = fixture()
    h = publish(tmp_path, d, r)
    out, decl = load_fresh_declaration(tmp_path, h)
    assert decl == d and out["Fresh_episodes_executed"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "missing_case",
        "missing_admission",
        "class",
        "duplicate_class",
        "cluster",
        "velocity",
        "candidate_selection",
        "seed_bool",
        "fit",
        "criteria",
        "resampled",
        "hardware",
    ],
)
def test_partial_or_rewritten_fresh_contract_cannot_unlock_physics(tmp_path, fault):
    d, r = fixture()
    if fault == "missing_case":
        d["cases"].pop()
    elif fault == "missing_admission":
        r["native_entries"].pop()
    elif fault == "class":
        d["cases"][0]["fresh_class"] = "easy"
    elif fault == "duplicate_class":
        d["cases"][1]["fresh_class"] = d["cases"][0]["fresh_class"]
    elif fault == "cluster":
        d["cases"][0]["source_cluster"] = 9
    elif fault == "velocity":
        r["native_entries"][0]["velocity"] = [0, 0]
    elif fault == "candidate_selection":
        d["candidate_seed_chosen_on_fresh"] = True
    elif fault == "seed_bool":
        d["candidate_seed_index"] = False
    elif fault == "fit":
        r["Fresh_episodes_fit"] = 1
    elif fault == "criteria":
        d["evaluation"]["third_gain_vs_parent_minimum"] = 0
    elif fault == "resampled":
        r["protocol"] = "another-draw"
    else:
        r["hardware_authorized"] = True
    h = publish(tmp_path, d, r)
    with pytest.raises(ValueError):
        load_fresh_declaration(tmp_path, h)


def test_receipt_hash_or_failed_marker_prevents_fresh_admission(tmp_path):
    d, r = fixture()
    h = publish(tmp_path, d, r)
    with pytest.raises(ValueError):
        load_fresh_declaration(tmp_path, h + "bad")
    (tmp_path / "failed.json").write_text("{}")
    with pytest.raises(ValueError):
        load_fresh_declaration(tmp_path, h)
