"""Original transient history checks, not physical policy qualification."""

import copy

import pytest

from scripts.rsi_read_impact_torque_screen import require_terminal_journal


def events():
    return [
        dict(
            _SYSTEMD_USER_UNIT="test.service",
            _SYSTEMD_INVOCATION_ID="original",
            MESSAGE=f"IMPACT_CASE_COMPLETE {i} FORECAST_STEPS 140",
            __REALTIME_TIMESTAMP=str(100 + i),
        )
        for i in range(6)
    ] + [
        dict(
            USER_UNIT="test.service",
            USER_INVOCATION_ID="original",
            MESSAGE_ID="39f53479d3a045ac8e11786248231fbf",
            JOB_TYPE="start",
            JOB_RESULT="done",
            __REALTIME_TIMESTAMP="200",
        )
    ]


def test_original_completed_history():
    assert require_terminal_journal(events(), "test.service") == "original"


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "different_invocation",
        "failed",
        "missing_case",
        "wrong_steps",
        "late_case",
        "repeated_success",
    ],
)
def test_no_missing_recreated_partial_or_failed_unit_history(bad):
    rows = copy.deepcopy(events())
    if bad == "missing":
        rows.pop()
    elif bad == "different_invocation":
        rows[-1]["USER_INVOCATION_ID"] = "recreated"
    elif bad == "failed":
        rows[-1]["JOB_RESULT"] = "failed"
    elif bad == "missing_case":
        rows.pop(3)
    elif bad == "wrong_steps":
        rows[0]["MESSAGE"] = "IMPACT_CASE_COMPLETE 0 FORECAST_STEPS 150"
    elif bad == "late_case":
        rows[0]["__REALTIME_TIMESTAMP"] = "201"
    else:
        rows.append(copy.deepcopy(rows[-1]))
    with pytest.raises(ValueError):
        require_terminal_journal(rows, "test.service")
