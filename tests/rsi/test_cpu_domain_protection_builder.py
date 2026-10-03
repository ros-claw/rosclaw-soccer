"""Input and authority preflight tests, not new physical evidence."""

from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_build_cpu_domain_protection import audit_success, checked_success_rows


def seal(value):
    value.pop("report_hash", None)
    value["report_hash"] = hash_json(value)
    return value


def declaration():
    rows = []
    for i in range(52):
        report = f"sha256:{i:064x}"
        outcome = seal(
            dict(
                physical_substeps=3000,
                high_quality=i < 12,
                reviewed_report_hash=report,
                actual_mujoco_dynamics_replayed=True,
                actual_pd_torque_reconstructed=True,
                neural_target_reconstructed=True,
                promotion_authorized=False,
                hardware_authorized=False,
            )
        )
        rows.append(
            dict(
                index=i,
                seed=42 + i,
                lane=0,
                report_hash=report,
                review_hash=outcome["report_hash"],
                outcome=outcome,
            )
        )
    commitment = seal(
        dict(
            schema="soccer.rsi.cpu_retained_parent_full_coverage_commitment.v1",
            model_hash="parent",
            partition="TRAIN_CONSUMED_CPU_DOMAIN",
            courses=[[r["seed"], r["lane"]] for r in rows],
            physics_change=False,
            fresh_exam_authorized=False,
            learning_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
    )
    summary = seal(
        dict(
            schema="soccer.rsi.cpu_retained_parent_full_coverage.v1",
            commitment_hash=commitment["report_hash"],
            rows=rows,
            independent_contexts=52,
            all_physical_substeps_replayed=156000,
            high_quality=12,
            fresh_exam_authorized=False,
            learning_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
    )
    return summary, commitment


def test_preflight_returns_all_successes_in_sealed_order():
    summary, commitment = declaration()
    assert [r["index"] for r in checked_success_rows(summary, commitment, "parent")] == list(
        range(12)
    )


@pytest.mark.parametrize(
    "fault", ["partial", "authority", "substeps", "duplicate", "high-quality-boolean"]
)
def test_invalid_complete_bank_rejected_before_worker_allocation(fault):
    summary, commitment = declaration()
    if fault == "partial":
        summary["rows"].pop()
    elif fault == "authority":
        summary["hardware_authorized"] = True
    elif fault == "substeps":
        summary["rows"][51]["outcome"]["physical_substeps"] = 2999
        seal(summary["rows"][51]["outcome"])
    elif fault == "duplicate":
        summary["rows"][51]["seed"] = 42
    else:
        summary["rows"][51]["outcome"]["high_quality"] = 0
        seal(summary["rows"][51]["outcome"])
    seal(summary)
    with pytest.raises(ValueError):
        checked_success_rows(summary, commitment, "parent")


def test_unbound_parent_is_rejected_before_cpu_replay(monkeypatch, tmp_path):
    raw = dict(
        step_model_hash="wrong", compiled_model_hash="world", report_hash="report", seed=1, lane=0
    )
    monkeypatch.setattr("scripts.rsi_build_cpu_domain_protection._sealed", lambda _: raw)
    monkeypatch.setattr(
        "scripts.rsi_build_cpu_domain_protection.audit_cpu_transfer",
        lambda *_: pytest.fail("must not replay an unbound parent"),
    )
    job = dict(
        row=dict(folder=str(tmp_path), report_hash="report", seed=1, lane=0),
        runner=str(Path("runner.py")),
        parent_hash="expected",
        world_hash="world",
    )
    with pytest.raises(ValueError, match="original parent"):
        audit_success(job)


def test_changed_success_review_aborts_before_state_extraction(monkeypatch, tmp_path):
    raw = dict(
        step_model_hash="parent", compiled_model_hash="world", report_hash="report", seed=1, lane=0
    )
    monkeypatch.setattr("scripts.rsi_build_cpu_domain_protection._sealed", lambda _: raw)
    monkeypatch.setattr(
        "scripts.rsi_build_cpu_domain_protection.audit_cpu_transfer",
        lambda *_: dict(high_quality=False),
    )
    monkeypatch.setattr(
        "scripts.rsi_build_cpu_domain_protection.cpu_features",
        lambda *_: pytest.fail("must not extract a failed audit"),
    )
    job = dict(
        row=dict(
            folder=str(tmp_path),
            report_hash="report",
            seed=1,
            lane=0,
            outcome=dict(high_quality=True),
        ),
        runner="runner.py",
        parent_hash="parent",
        world_hash="world",
    )
    with pytest.raises(ValueError, match="independent replay"):
        audit_success(job)
