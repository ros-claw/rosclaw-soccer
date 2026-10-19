from collections import Counter

from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from rosclaw_soccer.rsi.postcontact_robustness_plan import declare_robustness40
from rosclaw_soccer.rsi.public_response_probe_plan import declare_response_probes


def test_all_public_worlds_retained_with_outcome_independent_opposed_probes():
    model_hash = "sha256:" + "a" * 64
    plan = declare_response_probes(mean_model_hash=model_hash)
    parent = declare_robustness40(mean_model_hash=model_hash)
    assert plan == declare_response_probes(mean_model_hash=model_hash)
    assert plan["parent_public_plan_hash"] == parent["report_hash"]
    assert plan["native_step_budget"] == 80 * 3000
    assert plan["forecast_step_budget"] == 0
    assert plan["outcomes_used_to_choose_probe"] is False
    assert len(plan["records"]) == 40
    for row, original in zip(plan["records"], parent["records"], strict=True):
        for key in ("course", "sampling_seed", "source_cluster", "stratum"):
            assert row[key] == original[key]
        assert row["course_index"] == original["index"]
        assert row["heading_enabled"] is False
        assert len(set(row["actions"])) == 2
    counts = Counter(action for row in plan["records"] for action in row["actions"])
    assert set(counts) == set(ACTION_NAMES) - {"baseline"}
    assert min(counts.values()) == 6 and max(counts.values()) == 7
