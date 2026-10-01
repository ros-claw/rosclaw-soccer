import copy
import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi import approach_lateral_tracking_evidence as evidence
from scripts import rsi_collect_protected_phase_bank_validation as bank


def fixture(monkeypatch):
    parent = dict(
        report_hash="parent",
        source_hash="source",
        body_trace_hash="body",
        trace_hash="trace",
        asset_hash="asset",
        sonic_qualification_hash="sonic",
    )
    course = dict(seed=1, lane=2, parent_report="/old/report.json")
    commitment = dict(runner_hash="source", warm_model_hash="warm", model_hash="candidate")
    outcomes = {
        arm: dict(high_quality=True, minimum_pelvis_z_m=0.7) for arm in ("warm", "candidate")
    }
    raw = {
        arm: dict(
            report_hash=arm,
            parent_report_hash="parent",
            source_hash="source",
            contact_motor_policy_hash=arm,
            contact_motor_policy=dict(step_motor_proof=dict(model=dict(model_hash=arm))),
            environments=[dict(course={"physical": "same-course"})],
        )
        for arm in outcomes
    }
    parent["environments"] = [dict(course={"physical": "same-course"})]
    recorded = dict(index=0, seed=1, lane=2, parent_report_hash="parent")
    for arm in outcomes:
        recorded[arm] = dict(report_hash=arm, **outcomes[arm])
    visits = []

    def sealed(path):
        if str(path) == "/old/report.json" or "reproduction-parent" in str(path):
            return parent
        return raw["candidate" if "candidate-actor" in str(path) else "warm"]

    def outcome(folder, policy_hash, commitment):
        visits.append(str(folder))
        return dict(outcome=outcomes[policy_hash])

    monkeypatch.setattr(bank, "_sealed", sealed)
    monkeypatch.setattr(bank, "_outcome", outcome)
    monkeypatch.setattr(
        evidence, "audit_lateral_approach", lambda folder: visits.append(str(folder))
    )
    job = dict(root="/new", index=0, course=course, recorded=recorded, commitment=commitment)
    return job, raw, outcomes, visits


def test_row_worker_audits_parent_and_both_motor_arms(monkeypatch):
    job, raw, outcomes, visits = fixture(monkeypatch)
    original = copy.deepcopy(job)
    assert bank.review_course(job) == job["recorded"]
    assert job == original
    assert visits == [
        str(Path("/new/seed1-lane2-reproduction-parent")),
        str(Path("/new/seed1-lane2-warm-actor")),
        str(Path("/new/seed1-lane2-candidate-actor")),
    ]


@pytest.mark.parametrize("change", ["identity", "parent", "source", "model", "course", "outcome"])
def test_row_worker_cannot_accept_recorded_labels_without_bindings(monkeypatch, change):
    job, raw, outcomes, _ = fixture(monkeypatch)
    candidate = raw["candidate"]
    if change == "identity":
        job["recorded"]["index"] = 1
    elif change == "parent":
        candidate["parent_report_hash"] = "different"
    elif change == "source":
        candidate["source_hash"] = "different"
    elif change == "model":
        candidate["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"] = "different"
    elif change == "course":
        candidate["environments"][0]["course"] = {}
    else:
        outcomes["candidate"]["high_quality"] = False
    with pytest.raises(ValueError):
        bank.review_course(job)


def test_declared_52_contexts_cannot_be_duplicate_courses(monkeypatch, tmp_path):
    commitment = dict(learning_bank_hash="bank", partition="TRAIN_CONSUMED")
    summary = dict(
        schema="soccer.rsi.protected_phase_bank_physics.v1",
        commitment=commitment,
        physical_executions=156,
        independent_contexts=52,
        rows=[{} for _ in range(52)],
        promotion_authorized=False,
        hardware_authorized=False,
    )
    training = dict(report_hash="bank", courses=[dict(seed=1, lane=0) for _ in range(52)])
    path = tmp_path / "bank.json"
    (tmp_path / "commitment.json").write_text(json.dumps(commitment))
    monkeypatch.setattr(bank, "_sealed", lambda p: training if p == path else summary)
    with pytest.raises(ValueError, match="complete consumed comparison"):
        bank.review(tmp_path, path, workers=4)
