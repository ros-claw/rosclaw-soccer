import copy

import numpy as np
import pytest

from scripts import rsi_consolidate_retained_parent_memory as consolidation


@pytest.fixture
def success(monkeypatch):
    outcome = dict(high_quality=True, clean_foot_only=True)
    parent = dict(output_memory=dict(observations=[[0.0] * 135]), model_hash="parent")
    raw = dict(
        report_hash="actual",
        contact_motor_policy_hash="policy",
        contact_motor_policy=dict(step_motor_proof=dict(model=parent)),
    )
    row = dict(seed=42, lane=2, candidate=dict(report_hash="actual", **outcome))
    job = dict(row=row, commitment=dict(model_hash="parent"), root="fixture")

    class Decoder:
        def features(self, x):
            return x

        def raw_mean(self, x, phase):
            return np.ones(12) * phase

    decoder = Decoder()
    monkeypatch.setattr(consolidation, "_sealed", lambda _: raw)

    def audited(folder, policy, commitment, *, decoder_sink):
        decoder_sink.append(decoder)
        return dict(report=raw, outcome=outcome)

    monkeypatch.setattr(consolidation, "_outcome", audited)
    monkeypatch.setattr(
        consolidation,
        "gpu_observations",
        lambda *a: (np.ones((270, 134)), np.ones(270, dtype=int), raw),
    )
    return job, raw, row


def test_each_success_has_all_frames_and_explicit_old_coverage(success):
    job, _, _ = success
    record, states, means = consolidation.audit_success(job)
    assert record["frames"] == 270
    assert record["legacy_unprotected_frames"] == 270
    assert states.shape == (270, 135)
    assert means.shape == (270, 12)


def test_failed_child_label_cannot_enter_parent_memory(success):
    job, _, row = success
    row["candidate"]["high_quality"] = False
    with pytest.raises(ValueError):
        consolidation.audit_success(job)


def test_unrelated_parent_policy_cannot_be_consolidated(success):
    job, raw, _ = success
    raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"] = "other"
    with pytest.raises(ValueError):
        consolidation.audit_success(job)


def test_partial_frame_extraction_is_rejected(success, monkeypatch):
    job, raw, _ = success
    monkeypatch.setattr(
        consolidation,
        "gpu_observations",
        lambda *a: (np.ones((269, 134)), np.ones(269, dtype=int), raw),
    )
    with pytest.raises(ValueError, match="every active"):
        consolidation.audit_success(job)


def test_audit_failure_is_not_a_skipped_memory_record(success, monkeypatch):
    job, _, _ = success
    before = copy.deepcopy(job)

    def failure(*a, **k):
        raise ValueError("physical replay failed")

    monkeypatch.setattr(consolidation, "_outcome", failure)
    with pytest.raises(ValueError, match="physical replay"):
        consolidation.audit_success(job)
    assert job == before
