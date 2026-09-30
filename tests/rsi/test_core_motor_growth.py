import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_motor_primitive import make_policy
from rosclaw_soccer.rsi.core_motor_growth import growth_payload
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES


def fixture():
    rows = []
    for i, (seed, lane) in enumerate(COURSES):
        arms = {}
        for arm in ("gain_08", "gain_12", "learned"):
            high = (
                i in (1, 6)
                if arm == "gain_08"
                else i >= 7
                if arm == "gain_12"
                else i in (1, 6) or i >= 7
            )
            arms[arm] = {
                "high_quality": high,
                "clean_foot_only": high,
                "maximum_lateral_excursion_m": 3,
                "minimum_pelvis_z_m": 0.7,
            }
        rows.append({"seed": seed, "lane": lane, "arms": arms})
    commitment = hash_json({"experiment": 303})
    return dict(
        validation={"rows": rows, "report_hash": hash_json({"validation": 305})},
        policy=make_policy(np.full((3, 12), 0.01), commitment),
        protocol={"courses": [list(c) for c in COURSES]},
        body_hash=hash_json({"g1": True}),
        parent_hash=make_policy(np.zeros((3, 12)), commitment)["policy_hash"],
    )


def test_even_perfect_consumed_score_cannot_consolidate_without_fresh_or_cpu_evidence():
    payload = growth_payload(**fixture())
    assert payload["consolidation_manifest"]["decision"] == "need_more_evidence"
    assert payload["hardware_authorized"] is False
    assert payload["registry_write_count"] == 0
    assert payload["evidence_use_policy"]["promotion_truth_allowed"] is False
    gates = {g["name"]: g for g in payload["consolidation_manifest"]["gate_results"]}
    assert gates["darwin"]["status"] == "missing"
    assert gates["darwin"]["report_hash"] is None
    assert gates["applicability"]["status"] == "missing"


def test_observed_learning_gain_does_not_override_forgetting():
    args = copy.deepcopy(fixture())
    args["validation"]["rows"][1]["arms"]["learned"]["high_quality"] = False
    args["validation"]["rows"][0]["arms"]["learned"]["high_quality"] = True
    payload = growth_payload(**args)
    assert payload["consolidation_manifest"]["decision"] == "reject"
    assert payload["consolidation_manifest"]["forgotten_capability_ids"]


def test_incomplete_ledger_is_not_exportable():
    args = fixture()
    args["validation"]["rows"].pop()
    with pytest.raises(ValueError):
        growth_payload(**args)
