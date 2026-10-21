import json

import pytest

from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate
from rosclaw_soccer.sim.contracts import hash_bytes


def sealed_pair(tmp_path):
    exam, audit = tmp_path / "exam", tmp_path / "audit"
    exam.mkdir()
    audit.mkdir()
    e = dict(
        status="PAID_NEURAL_STANDING_LIFECYCLE_CLOSED_PENDING_AUDIT",
        variant=1,
        course_bank="train",
        skills=3,
        worlds=[{}] * 16,
        summary=dict(stable=16),
        actual_native_steps=1000,
        actual_neural_rows=100,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        old_exam_result=False,
        team_game=False,
    )
    ep = exam / "complete.json"
    ep.write_text(json.dumps(e))
    a = dict(
        status="PAID_NEURAL_STANDING_LIFECYCLE_NATIVE_AND_NEURAL_AUDIT_PASS",
        input_file_hashes={str(ep): hash_bytes(ep.read_bytes())},
        summary=e["summary"],
        counts=dict(native=1000, neural=100, own_neural=100, worlds=32),
        own_native_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        old_exam_result=False,
        team_game=False,
    )
    (audit / "complete.json").write_text(json.dumps(a))
    return exam, audit, e, a


def test_continuation_requires_scientific_receipt_not_only_worker_exit(tmp_path):
    exam, audit, _, _ = sealed_pair(tmp_path)
    assert standing_gate(exam, audit, variant=1, bank="train")["summary"] == {"stable": 16}


@pytest.mark.parametrize("fault", ["hash", "inference", "worlds", "summary", "hardware"])
def test_continuation_rejects_forged_or_incomplete_gate(tmp_path, fault):
    exam, audit, _, a = sealed_pair(tmp_path)
    if fault == "hash":
        a["input_file_hashes"] = {}
    elif fault == "inference":
        a["counts"]["own_neural"] = 99
    elif fault == "worlds":
        a["counts"]["worlds"] = 31
    elif fault == "summary":
        a["summary"] = {"stable": 15}
    else:
        a["hardware_authorized"] = True
    (audit / "complete.json").write_text(json.dumps(a))
    with pytest.raises(ValueError):
        standing_gate(exam, audit, variant=1, bank="train")


def test_failed_artifact_cannot_be_retried_or_hidden_by_complete(tmp_path):
    exam, audit, _, _ = sealed_pair(tmp_path)
    (audit / "failed.json").write_text("{}")
    with pytest.raises(ValueError, match="never automatic retry"):
        standing_gate(exam, audit, variant=1, bank="train")


def test_dev_result_cannot_be_admitted_as_train_or_another_variant(tmp_path):
    exam, audit, _, _ = sealed_pair(tmp_path)
    for variant, bank in ((2, "train"), (1, "dev")):
        with pytest.raises(ValueError):
            standing_gate(exam, audit, variant=variant, bank=bank)
