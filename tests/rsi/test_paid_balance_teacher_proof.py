import hashlib
import json

import pytest

from rosclaw_soccer.rsi.paid_balance_teacher_proof import balance_teacher_proof


def fixture(tmp_path, mutate=None):
    """Schema-only evidence fixture, never a physical success claim."""
    root, audit = tmp_path / "teacher", tmp_path / "audit"
    root.mkdir()
    audit.mkdir()

    def save(path, data):
        raw = json.dumps(data).encode()
        path.write_bytes(raw)
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    d = dict(
        recovery_start="reference-end",
        pilot_cases=[0, 15],
        source_cases=[{}] * 16,
        max_delta_rad=0.2,
        max_delta_per_control_rad=0.02,
        new_phase_controls=100,
    )
    dh = save(root / "declaration.json", d)
    r = dict(
        status="PAID_ACTIVE_BALANCE_TEACHER_PILOT_CLOSED_PENDING_AUDIT",
        recovery_start="reference-end",
        selected_rank=[2, 2, 100, 0.0],
        selected_latent=[[0.0] * 4 for _ in range(9)],
        actual_native_steps=40,
        actual_neural_rows=8,
        neural_optimizer_updates=0,
        final=[
            [
                dict(
                    case=i,
                    recovery=dict(healthy=True, recovery_stability_observed=True),
                    third=dict(body_failed=False, neural_rows=2),
                    continuation=dict(status="READY"),
                )
                for _ in range(2)
            ]
            for i in (0, 15)
        ],
    )
    if mutate:
        mutate(r)
    rh = save(root / "complete.json", r)
    ah = save(
        audit / "complete.json",
        dict(
            status="PAID_ACTIVE_BALANCE_ALL_PRIMITIVE_AND_CONTINUATION_AUDIT_PASS",
            counts=dict(producer_native=40, producer_neural=8),
            own_native_steps=0,
            own_neural_rows=8,
            input_file_hashes={str(root / "complete.json"): rh, str(root / "declaration.json"): dh},
        ),
    )
    return root, rh, audit, ah


def test_frozen_common_gain_admission(tmp_path):
    r, d, pins, gain = balance_teacher_proof(*fixture(tmp_path))
    assert gain.shape == (9, 4) and not gain.any()
    assert d["pilot_cases"] == [0, 15]
    assert len(pins) == 3


@pytest.mark.parametrize("fault", ["late", "rank", "gain", "body", "repeat", "neural"])
def test_missing_real_recovery_or_continuation_cannot_enter_transfer(tmp_path, fault):
    def mutate(r):
        if fault == "late":
            r["recovery_start"] = "complete-skill"
        if fault == "rank":
            r["selected_rank"][0] = 1
        if fault == "gain":
            r["selected_latent"][0][0] = 2
        if fault == "body":
            r["final"][0][0]["third"]["body_failed"] = True
        if fault == "repeat":
            r["final"][0].pop()
        if fault == "neural":
            r["final"][1][1]["third"]["neural_rows"] = 3

    with pytest.raises(ValueError):
        balance_teacher_proof(*fixture(tmp_path, mutate))
