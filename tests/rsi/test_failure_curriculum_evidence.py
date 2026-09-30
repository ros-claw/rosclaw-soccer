import copy
import json

import numpy as np
import pytest

from rosclaw_soccer.rsi import failure_curriculum_evidence as evidence
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.failure_curriculum_evidence import retention_score
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES
from scripts.rsi_train_failure_curriculum_motor_v306 import rank


def fixtures():
    rows = [
        {
            "high_quality": True,
            "clean_foot_only": True,
            "minimum_pelvis_z_m": 0.7,
            "maximum_lateral_excursion_m": 3.0,
            "reward": 2.0,
        }
        for _ in range(12)
    ]
    reference = [
        {"arms": {arm: copy.deepcopy(r) for arm in ("gain_08", "gain_12", "learned")}} for r in rows
    ]
    return rows, reference


def test_independent_named_counters_reconstruct_training_rank():
    rows, reference = fixtures()
    result = retention_score(rows, reference)
    assert result["rank"] == list(rank(rows, reference))
    assert result["consumed_training_gate_passed"] is True
    rows[0].update(high_quality=False, clean_foot_only=False, maximum_lateral_excursion_m=4.1)
    rows[1]["minimum_pelvis_z_m"] = 0.64
    result = retention_score(rows, reference)
    assert result["rank"] == list(rank(rows, reference))
    assert result["consumed_training_gate_passed"] is False
    for key in (
        "predecessor_high_quality_loss",
        "raw_gain_high_quality_loss",
        "old_clean_foot_loss",
        "new_out_of_play",
        "old_high_quality_loss",
    ):
        assert result[key] == 1
        assert result["lost_courses"][key] == [[20261177, 0]]


@pytest.mark.parametrize(
    "key,value",
    [
        ("reward", float("nan")),
        ("minimum_pelvis_z_m", float("inf")),
        ("high_quality", 1),
        ("reward", True),
    ],
)
def test_nonfinite_and_truthy_metrics_cannot_pass(key, value):
    rows, reference = fixtures()
    rows[0][key] = value
    with pytest.raises(ValueError):
        retention_score(rows, reference)


def test_incomplete_suite_and_old_nonboolean_outcome_rejected():
    rows, reference = fixtures()
    with pytest.raises(ValueError):
        retention_score(rows[:-1], reference)
    reference[0]["arms"]["learned"]["high_quality"] = "yes"
    with pytest.raises(ValueError):
        retention_score(rows, reference)


def test_large_reward_does_not_override_forgotten_success():
    rows, reference = fixtures()
    good = retention_score(rows, reference)
    rows[0]["high_quality"] = False
    rows[0]["reward"] = 100000
    bad = retention_score(rows, reference)
    assert good["rank"] > bad["rank"]
    assert not bad["consumed_training_gate_passed"]


def _write(path, content, seal=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = copy.deepcopy(content)
    content.pop("report_hash", None)
    if seal:
        content["report_hash"] = hash_json(content)
    path.write_text(json.dumps(content))
    return content


@pytest.fixture
def review_fixture(tmp_path, monkeypatch):
    root, old = tmp_path / "training", tmp_path / "old"
    rows, reference = fixtures()
    anchors = [
        {"seed": s, "lane": lane, **r} for (s, lane), r in zip(COURSES, reference, strict=True)
    ]
    # Historical collector order is intentionally not curriculum order.
    prior = _write(
        old / "validation_summary.json",
        {
            "rows": anchors[::-1],
            "promotion_authorized": False,
            "fresh_holdout_open_authorized": False,
        },
    )
    commitment = {
        "prior_validation_hash": prior["report_hash"],
        "runner_hash": hash_json({"runner": 1}),
        "asset_hash": hash_json({"asset": 1}),
    }
    _write(root / "commitment.json", commitment, seal=False)
    policy = make_policy(np.zeros((3, 12)), 0.25, hash_json(commitment))
    _write(root / "policies/g0-c0.json", policy, seal=False)
    # Policy seals use policy_hash, not report_hash.
    for seed, lane in COURSES:
        parent = _write(
            root / f"seed{seed}-lane{lane}-reproduction-parent/report.json",
            {
                "sonic_qualification_hash": hash_json({"sonic": 1}),
                "environments": [{"course": [seed, lane]}],
            },
        )
        raw = {
            "training_course_seed": seed,
            "single_course_lane": lane,
            "parent_report_hash": parent["report_hash"],
            "sonic_qualification_hash": parent["sonic_qualification_hash"],
            "environments": parent["environments"],
            "body_trace_hash": hash_json({"body": seed}),
            "trace_hash": hash_json({"ball": seed}),
        }
        _write(root / f"seed{seed}-lane{lane}-g0-c0-actor/report.json", raw)
        _write(root / f"seed{seed}-lane{lane}-reproduction-actor/report.json", raw)
        _write(old / f"seed{seed}-lane{lane}-learned-actor/report.json", raw)

    def measured(folder, policy_hash, commitment):
        raw = evidence._sealed(folder / "report.json")
        return {
            "report": raw,
            "outcome": rows[
                COURSES.index((raw["training_course_seed"], raw["single_course_lane"]))
            ],
        }

    monkeypatch.setattr(evidence, "_outcome", measured)
    return root, old


def test_perfect_progress_is_not_a_completed_or_promotable_experiment(review_fixture):
    root, old = review_fixture
    report = evidence.review_curriculum(root, old)
    assert report["complete_candidate_count"] == 1
    assert report["best"]["score"]["consumed_training_gate_passed"] is True
    assert report["training_complete"] is False
    assert report["consumed_training_gate_passed"] is False
    assert report["promotion_authorized"] is False
    assert report["fresh_holdout_open_authorized"] is False


def test_false_complete_summary_rejected(review_fixture):
    root, old = review_fixture
    _write(root / "training_summary.json", {"consumed_training_gate_passed": True})
    with pytest.raises(ValueError, match="final training summary"):
        evidence.review_curriculum(root, old)


def test_trainer_reward_ledger_cannot_disagree_with_measured_physics(review_fixture):
    root, old = review_fixture
    _write(root / "g0-c0-result.json", {"rows": [], "reports": [], "score": [1000]}, seal=False)
    with pytest.raises(ValueError, match="trainer result"):
        evidence.review_curriculum(root, old)


def test_initial_clone_must_match_historical_body_and_ball(review_fixture):
    root, old = review_fixture
    path = old / "seed20261177-lane0-learned-actor/report.json"
    raw = json.loads(path.read_text())
    raw["trace_hash"] = hash_json({"different": True})
    _write(path, raw)
    with pytest.raises(ValueError, match="reproduction mismatch"):
        evidence.review_curriculum(root, old)
