"""Admission tests only: no real assets or physics are loaded."""

import pytest

from scripts import rsi_verify_shared_nn_forecast as runner


class ReadOnlyAdmissionReached(Exception):
    pass


def invoke(monkeypatch, flags):
    def stop_before_files_or_physics(head):
        raise ReadOnlyAdmissionReached

    monkeypatch.setattr("scripts.rsi_native_fresh_common.clean", stop_before_files_or_physics)
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualify",
            "--effect-review",
            "not-opened.json",
            "--reference-record",
            "not-opened.json",
            "--output",
            "not-created",
            "--effect-review-hash",
            "unused",
            "--expected-head",
            "unused",
            *flags,
        ],
    )
    runner.main()


@pytest.mark.parametrize("axis", [0, 3, 28])
def test_explicit_actuated_axis_reaches_only_read_only_admission(monkeypatch, axis):
    with pytest.raises(ReadOnlyAdmissionReached):
        invoke(
            monkeypatch,
            [
                "--executed-task-axis",
                str(axis),
                "--pre-proposal-128",
                "--target-feasible-axes",
            ],
        )


@pytest.mark.parametrize(
    "flags",
    [
        ["--executed-task-axis", "-1", "--pre-proposal-128", "--target-feasible-axes"],
        ["--executed-task-axis", "29", "--pre-proposal-128", "--target-feasible-axes"],
        ["--executed-task-axis", "3", "--target-feasible-axes"],
        ["--executed-task-axis", "3", "--pre-proposal-128"],
        ["--executed-task-axis", "3", "--pre-proposal-96", "--target-feasible-axes"],
        [
            "--executed-task-axis",
            "3",
            "--pre-proposal-128",
            "--target-feasible-axes",
            "--executed-synergy",
        ],
        [
            "--executed-task-axis",
            "3",
            "--pre-proposal-128",
            "--target-feasible-axes",
            "--synergy-model",
            "forbidden.json",
        ],
        [
            "--executed-task-axis",
            "3",
            "--pre-proposal-128",
            "--target-feasible-axes",
            "--synergy-model-hash",
            "forbidden",
        ],
        [
            "--executed-task-axis",
            "3",
            "--pre-proposal-128",
            "--target-feasible-axes",
            "--whole-joint-reference",
        ],
        [
            "--executed-task-axis",
            "3",
            "--pre-proposal-128",
            "--target-feasible-axes",
            "--aim-goal-y=-1.0",
        ],
    ],
)
def test_no_mixed_or_implicitly_expanded_mode(monkeypatch, flags):
    with pytest.raises(ValueError):
        invoke(monkeypatch, flags)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--pre-proposal-128", "--whole-joint-reference"],
        ["--pre-proposal-128", "--aim-goal-y=-1.0"],
        [
            "--pre-proposal-128",
            "--executed-synergy",
            "--synergy-model",
            "old.json",
            "--target-feasible-axes",
        ],
    ],
)
def test_old_explicit_qualification_modes_remain_separate(monkeypatch, flags):
    with pytest.raises(ReadOnlyAdmissionReached):
        invoke(monkeypatch, flags)
