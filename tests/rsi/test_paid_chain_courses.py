from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.paid_chain_courses import chain_parent_profile


def fixture():
    train = [dict(source=f"train-{i}-right.npz") for i in range(16)]
    dev = [dict(source=f"dev-{i}-left.npz") for i in range(16)]

    def row(case, stage):
        return dict(
            source=case["source"], raw_path=f"{stage}/{case['source']}", raw_hash="sha256:test"
        )

    train_worlds = [dict(first=row(c, "first"), second=row(c, "second")) for c in train]
    dev_rows = [row(c, "dev-first") for c in dev]
    d = dict(old_cases=train, DEV_cases=dev)
    exam = dict(baseline_continuous=deepcopy(train_worlds), baseline_isolated=dict(dev16=dev_rows))
    candidate = dict(
        continuous=dict(results=train_worlds), isolated=dict(dev16=dict(results=deepcopy(dev_rows)))
    )
    return d, exam, candidate


def test_train_requires_both_actual_parent_actions():
    d, exam, candidate = fixture()
    cases, prefixes = chain_parent_profile(d, exam, candidate, "train")
    assert cases == d["old_cases"]
    for arm in prefixes.values():
        assert len(arm) == 16 and all(set(row) == {0, 1} for row in arm)


def test_dev_has_only_one_qualified_action_not_a_fabricated_second():
    d, exam, candidate = fixture()
    cases, prefixes = chain_parent_profile(d, exam, candidate, "dev")
    assert cases == d["DEV_cases"]
    for arm in prefixes.values():
        assert len(arm) == 16 and all(set(row) == {0} for row in arm)
    assert prefixes["learned"][3][0] == candidate["isolated"]["dev16"]["results"][3]


@pytest.mark.parametrize("bank", ["fresh", "DEV", "", None])
def test_other_bank_cannot_masquerade_as_development(bank):
    with pytest.raises(ValueError, match="explicit"):
        chain_parent_profile(*fixture(), bank)


@pytest.mark.parametrize("bank,key", [("train", "old_cases"), ("dev", "DEV_cases")])
def test_no_course_may_be_dropped(bank, key):
    d, exam, candidate = fixture()
    d[key].pop()
    with pytest.raises(ValueError, match="sixteen"):
        chain_parent_profile(d, exam, candidate, bank)


def test_train_rows_cannot_be_relabelled_dev():
    d, exam, candidate = fixture()
    candidate["isolated"]["dev16"]["results"][0] = candidate["continuous"]["results"][0]["first"]
    with pytest.raises(ValueError, match="source-bound"):
        chain_parent_profile(d, exam, candidate, "dev")


def test_short_prefix_coverage_fails_before_physics():
    d, exam, candidate = fixture()
    candidate["isolated"]["dev16"]["results"].pop()
    with pytest.raises(ValueError, match="coverage"):
        chain_parent_profile(d, exam, candidate, "dev")
