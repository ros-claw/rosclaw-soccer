import hashlib
import json

import pytest

from rosclaw_soccer.rsi.paid_recovery_proof import recovery_proof, unique_proof_path


def fixture(tmp_path, *, mutate=None):
    root, audit = tmp_path / "chain", tmp_path / "audit"
    root.mkdir()
    audit.mkdir()

    def save(path, value):
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    cases = [dict(source=f"clip-{i}") for i in range(16)]
    dp = root / "declaration.json"
    dh = save(dp, dict(cases=cases, body_resets_between_skills=0, fixed_world_goal=True))
    report = dict(
        status="PAID_CONTINUING_CHAIN_CPU_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT",
        skills=3,
        actual_neural_rows=17,
        all_independent_chain_repeats_exact=True,
    )
    if mutate:
        report.update(mutate)
    rp = root / "complete.json"
    rh = save(rp, report)
    ap = audit / "complete.json"
    ah = save(
        ap,
        dict(
            status="PAID_CONTINUING_CHAIN_NATIVE_AND_NEURAL_AUDIT_PASS",
            own_native_steps=0,
            own_neural_rows=17,
            input_file_hashes={str(rp): rh, str(dp): dh},
        ),
    )
    return root, rh, audit, ah, cases


def test_exact_physical_and_neural_recovery_reference(tmp_path):
    root, rh, audit, ah, cases = fixture(tmp_path)
    declaration, pins = recovery_proof(root, rh, audit, ah, cases=cases)
    assert declaration["cases"] == cases
    assert pins[str(audit / "complete.json")] == ah


@pytest.mark.parametrize(
    "mutation",
    [
        dict(status="PENDING"),
        dict(actual_neural_rows=0),
        dict(skills=4),
        dict(all_independent_chain_repeats_exact=False),
        dict(recovery_teacher={"nested": True}),
    ],
)
def test_unqualified_or_nested_teacher_never_becomes_recovery_reference(tmp_path, mutation):
    root, rh, audit, ah, cases = fixture(tmp_path, mutate=mutation)
    with pytest.raises(ValueError, match="pure-neural"):
        recovery_proof(root, rh, audit, ah, cases=cases)


def test_fail_marker_blocks_even_with_complete_receipt(tmp_path):
    root, rh, audit, ah, cases = fixture(tmp_path)
    (root / "failed.json").write_text("{}")
    with pytest.raises(ValueError, match="without failures"):
        recovery_proof(root, rh, audit, ah, cases=cases)


def test_changed_courses_and_hashes_rejected(tmp_path):
    root, rh, audit, ah, cases = fixture(tmp_path)
    with pytest.raises(ValueError, match="hash changed"):
        recovery_proof(root, rh + "bad", audit, ah, cases=cases)
    with pytest.raises(ValueError, match="same declared"):
        recovery_proof(root, rh, audit, ah, cases=cases[::-1])


def test_actor_path_must_be_unambiguous(tmp_path):
    assert unique_proof_path({str(tmp_path / "a"): "h"}, "h") == tmp_path / "a"
    with pytest.raises(ValueError, match="unambiguous"):
        unique_proof_path({str(tmp_path / "a"): "h", str(tmp_path / "b"): "h"}, "h")
