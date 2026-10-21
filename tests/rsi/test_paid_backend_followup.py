import pytest

from scripts.rsi_follow_paid_backend import producer_finished


def state(**kwargs):
    return (
        dict(
            ActiveState="active",
            SubState="running",
            Result="success",
            MainPID="17",
            ExecMainCode="0",
            ExecMainStatus="0",
        )
        | kwargs
    )


def test_live_producer_is_not_complete_even_if_receipt_exists():
    assert not producer_finished(state())
    assert not producer_finished(state(SubState="exited"))
    assert producer_finished(state(MainPID="0", SubState="exited", ExecMainCode="1"))


@pytest.mark.parametrize(
    "s",
    [
        state(ActiveState="failed", Result="exit-code", ExecMainStatus="1"),
        state(MainPID="0", SubState="exited", ExecMainCode="2"),
        state(MainPID="0", SubState="exited", ExecMainCode="1", ExecMainStatus="1"),
    ],
)
def test_failures_never_launch_successor_or_retry_physics(s):
    with pytest.raises(ValueError):
        producer_finished(s)
