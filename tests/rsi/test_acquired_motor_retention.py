import pytest

from rosclaw_soccer.rsi.acquired_motor_retention import acquired_motor_retention, native_successes


def rows(good):
    return [dict(context=i, repeat=j, quality=i in good) for i in range(22) for j in range(2)]


def test_original_protection_flag_cannot_hide_loss_of_later_acquired_skill():
    result = acquired_motor_retention(
        context_hashes=[f"sha256:{i:064x}" for i in range(22)],
        original_protected=[1, 4, 5],
        historical=[
            ("sha256:" + "a" * 64, rows({1, 5, 8})),
            ("sha256:" + "b" * 64, rows({1, 4, 8, 20, 21})),
        ],
        current=rows({1, 4, 5, 20, 21}),
    )
    assert result["original_protection_retained"] is True
    assert result["acquired_frontier_retained"] is False
    assert result["lost_contexts"] == [8]
    assert result["worst_critical_drop"] == 1
    assert all(t["critical"] for t in result["task_retention"])
    assert result["promotion_authorized"] is False


def test_union_preserves_new_skills_without_requiring_new_model_to_repeat_old_failures():
    r = acquired_motor_retention(
        context_hashes=[f"sha256:{i:064x}" for i in range(22)],
        original_protected=[1],
        historical=[("sha256:" + "c" * 64, rows({1, 4, 20, 21}))],
        current=rows({1, 4, 5, 20, 21}),
    )
    assert r["required_contexts"] == [1, 4, 20, 21]
    assert r["acquired_frontier_retained"] is True and r["worst_critical_drop"] == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "not_bool", "inconsistent"])
def test_replays_cannot_inflate_or_selectively_erase_failed_outcomes(fault):
    r = rows({1})
    if fault == "missing":
        r.pop()
    elif fault == "duplicate":
        r[-1] = dict(r[0])
    elif fault == "not_bool":
        r[0]["quality"] = 0
    else:
        r[0]["quality"] = True
    with pytest.raises(ValueError):
        native_successes(r)
