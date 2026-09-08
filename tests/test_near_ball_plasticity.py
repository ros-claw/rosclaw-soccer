import pytest

from rosclaw_soccer.training.near_ball_plasticity import (
    begin_update,
    finish_update,
    verify_update_record,
)


def test_core_lease_rejects_frozen_role_change_and_wrong_start() -> None:
    before = {"red.mid": "sha256:" + "a" * 64, "blue.mid": "sha256:" + "b" * 64}
    lease = begin_update(
        before=before,
        focal="red.mid",
        generation=1,
        dataset_hash="sha256:" + "c" * 64,
        context_hash="sha256:" + "d" * 64,
        maximum_steps=4,
    )
    after = {**before, "red.mid": "sha256:" + "e" * 64}
    result = finish_update(lease=lease, before=before, after=after, steps=4)
    assert result["audit"]["passed"] and result["audit"]["changed_agent_ids"] == ["red.mid"]
    with pytest.raises(ValueError, match="FROZEN_AGENT_CHANGED"):
        finish_update(
            lease=lease, before=before, after={**after, "blue.mid": "sha256:" + "f" * 64}, steps=4
        )
    with pytest.raises(ValueError, match="starting weights"):
        finish_update(lease=lease, before=after, after=after, steps=4)
    with pytest.raises(ValueError, match="BUDGET"):
        finish_update(lease=lease, before=before, after=after, steps=5)


def test_record_verification_binds_weights_context_and_integer_budget() -> None:
    before = {"red.mid": "sha256:" + "a" * 64, "blue.mid": "sha256:" + "b" * 64}
    context = dict(
        before=before,
        focal="red.mid",
        generation=1,
        dataset_hash="sha256:" + "c" * 64,
        context_hash="sha256:" + "d" * 64,
        maximum_steps=4,
    )
    lease = begin_update(**context)
    after = {**before, "red.mid": "sha256:" + "e" * 64}
    record = finish_update(lease=lease, before=before, after=after, steps=4)
    verify_update_record(record, after=after, **context)
    with pytest.raises(ValueError, match="proof differs"):
        verify_update_record(record, after=before, **context)
    with pytest.raises(ValueError, match="proof differs"):
        verify_update_record(
            record, after=after, **{**context, "dataset_hash": "sha256:" + "f" * 64}
        )
    for steps in (True, 1.0, -1):
        with pytest.raises(ValueError, match="nonnegative integer"):
            finish_update(lease=lease, before=before, after=after, steps=steps)
