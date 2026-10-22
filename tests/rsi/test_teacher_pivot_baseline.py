import hashlib
import json

import pytest

from rosclaw_soccer.rsi.teacher_pivot_baseline import audited_teacher_baseline


def fixture(tmp_path, *, wins=1, source_training="trained"):
    source = tmp_path / "source"
    audit = tmp_path / "audit"
    source.mkdir()
    audit.mkdir()
    p = source / "complete.json"
    p.write_text(
        json.dumps(
            dict(
                status="DUAL_MOTOR_ACTUAL_STATE_TEACHER_RECOVERY_CLOSED_PENDING_OWN_AUDIT",
                source_head="pinned",
                training_hash=source_training,
                hardware_authorized=False,
                promotion_authorized=False,
                actual_native_steps=70020,
                actual_actor_decisions=7002,
                selected_teacher_successes=wins,
            )
        )
    )
    h = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    (audit / "complete.json").write_text(
        json.dumps(
            dict(
                status="DUAL_MOTOR_ASSISTED18_CONTINUAL_OWN_NATIVE_AUDIT_PASS",
                source_head="pinned",
                worlds=18,
                results=[{}] * 18,
                teacher_mode="measured_contact_observed_shin",
                selected_teacher_successes=wins,
                hardware_authorized=False,
                promotion_authorized=False,
                input_file_hashes={str(p): h},
                source_hash=h,
                counts=dict(native=70020, neural=7002),
            )
        )
    )
    return source, audit


def test_same_candidate_incomplete_teacher_is_required(tmp_path):
    _, audit = fixture(tmp_path)
    h, pins = audited_teacher_baseline(audit, head="pinned", training_hash="trained")
    assert pins[str(audit / "complete.json")] == h
    with pytest.raises(ValueError):
        audited_teacher_baseline(audit, head="other", training_hash="trained")
    with pytest.raises(ValueError):
        audited_teacher_baseline(audit, head="pinned", training_hash="other")


def test_already_successful_teacher_cannot_authorize_this_pivot(tmp_path):
    _, audit = fixture(tmp_path, wins=3)
    with pytest.raises(ValueError):
        audited_teacher_baseline(audit, head="pinned", training_hash="trained")


def test_source_mutation_is_detected(tmp_path):
    source, audit = fixture(tmp_path)
    (source / "complete.json").write_text("{}")
    with pytest.raises(ValueError):
        audited_teacher_baseline(audit, head="pinned", training_hash="trained")
