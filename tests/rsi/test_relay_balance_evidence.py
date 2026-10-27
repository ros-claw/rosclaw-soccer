import hashlib
import json

import pytest

from rosclaw_soccer.rsi.relay_balance_evidence import admit_relay_balance_teachers


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_three_distinct_audit_bound_teachers(tmp_path):
    pairs = []
    for i in range(3):
        source, audit = tmp_path / f"source{i}.json", tmp_path / f"audit{i}.json"
        source.write_text(
            json.dumps(
                dict(
                    status="PAID_ACTIVE_BALANCE_TEACHER_PILOT_CLOSED_PENDING_AUDIT",
                    selected_latent=[[0.0] * 4] * 9,
                    all_selected_recovery_and_third_repeats_exact=True,
                    goal_complete=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                )
            )
        )
        audit.write_text(
            json.dumps(
                dict(
                    status="PAID_ACTIVE_BALANCE_ALL_PRIMITIVE_AND_CONTINUATION_AUDIT_PASS",
                    input_file_hashes={str(source): digest(source)},
                    goal_complete=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                )
            )
        )
        pairs.append([str(source), str(audit), digest(source), digest(audit)])
    latents, pins = admit_relay_balance_teachers(pairs)
    assert len(latents) == 3 and len(pins) == 6
    with pytest.raises(ValueError):
        admit_relay_balance_teachers([pairs[0]] * 3)
    audit = tmp_path / "audit1.json"
    audit.write_text("{}")
    with pytest.raises(ValueError):
        admit_relay_balance_teachers(pairs)
