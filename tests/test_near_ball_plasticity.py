import pytest

from rosclaw_soccer.training.near_ball_plasticity import begin_update, finish_update


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
