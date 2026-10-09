"""Synthetic scheduling tests: no real private pool, labels or physics."""

from dataclasses import FrozenInstanceError

import pytest

from rosclaw_soccer.rsi.paired_first_touch_schedule import PairedFirstTouchSchedule
from rosclaw_soccer.sim.contracts import hash_json


def bindings():
    return {
        name: "sha256:" + str(index) * 64
        for index, name in enumerate(
            (
                "protocol_hash",
                "pool_hash",
                "train_split_hash",
                "physics_hash",
                "parent_hash",
                "candidate973_hash",
                "candidate974_hash",
            ),
            start=1,
        )
    }


def test_all40_courses_get_all_three_fixed_policies_without_reselection():
    args = bindings()
    schedule = PairedFirstTouchSchedule(**args)
    jobs = schedule.jobs()
    assert len(jobs) == 120
    assert [job.ordinal for job in jobs] == list(range(120))
    expected = [args[key] for key in ("parent_hash", "candidate973_hash", "candidate974_hash")]
    for case in range(40):
        triple = jobs[3 * case : 3 * case + 3]
        assert [job.case_index for job in triple] == [case] * 3
        assert [job.policy_hash for job in triple] == expected
        assert [job.policy_role for job in triple] == [
            "parent",
            "candidate973_primary0",
            "candidate974_primary0",
        ]
        assert all(schedule.job(job.ordinal) == job for job in triple)


@pytest.mark.parametrize("name", tuple(bindings()))
@pytest.mark.parametrize(
    "bad",
    [
        None,
        True,
        1,
        "",
        "0" * 64,
        "sha256:" + "A" * 64,
        "sha256:" + "0" * 63,
        "sha256:" + "0" * 64 + "\n",
    ],
)
def test_rejects_malformed_identity(name, bad):
    args = bindings()
    args[name] = bad
    with pytest.raises(ValueError, match="identities"):
        PairedFirstTouchSchedule(**args)


@pytest.mark.parametrize(
    "left,right",
    [
        ("parent_hash", "candidate973_hash"),
        ("parent_hash", "candidate974_hash"),
        ("candidate973_hash", "candidate974_hash"),
    ],
)
def test_rejects_same_model_as_two_policies(left, right):
    args = bindings()
    args[left] = args[right]
    with pytest.raises(ValueError, match="differ"):
        PairedFirstTouchSchedule(**args)


@pytest.mark.parametrize("index", [True, False, -1, 120, 1.0, "0", None])
def test_rejects_invalid_job_index(index):
    with pytest.raises(ValueError, match="index"):
        PairedFirstTouchSchedule(**bindings()).job(index)


def test_contract_is_complete_sealed_owned_and_never_execution_authority():
    schedule = PairedFirstTouchSchedule(**bindings())
    contract = schedule.contract()
    seal = contract.pop("schedule_hash")
    assert hash_json(contract) == seal
    assert contract["ceiling"] == "SCHEDULE_ONLY_NOT_EXECUTION_ADMISSION"
    assert contract["native_execution_count_if_admitted"] == 120
    assert all(
        value is False
        for key, value in contract.items()
        if key.endswith(("_authorized", "_read", "_here"))
    )
    contract["binding"]["parent_hash"] = "sha256:" + "0" * 64
    contract["jobs"].clear()
    contract["fresh_execution_authorized"] = True
    assert len(schedule.jobs()) == 120
    assert schedule.contract()["schedule_hash"] == seal
    assert schedule.contract()["fresh_execution_authorized"] is False


@pytest.mark.parametrize("name", tuple(bindings()))
def test_every_identity_is_bound_into_schedule_hash(name):
    args = bindings()
    original = PairedFirstTouchSchedule(**args).contract()["schedule_hash"]
    args[name] = "sha256:" + "a" * 64
    assert PairedFirstTouchSchedule(**args).contract()["schedule_hash"] != original


def test_job_is_frozen_and_caller_cannot_corrupt_stored_schedule():
    schedule = PairedFirstTouchSchedule(**bindings())
    job = schedule.job(0)
    with pytest.raises(FrozenInstanceError):
        job.case_index = 39
    object.__setattr__(job, "case_index", 39)
    assert schedule.job(0).case_index == 0
