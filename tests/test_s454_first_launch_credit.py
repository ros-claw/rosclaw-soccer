"""A weak pass followed by a chase must not satisfy first-pass learning credit."""

import pytest

from rosclaw_soccer.training.first_launch_credit import advance_first_launch_credit

torch = pytest.importorskip("torch")


def initial():
    return dict(
        departed=torch.tensor([False, False]),
        closed=torch.tensor([False, False]),
        departure_time_sec=torch.tensor([-1.0, -1.0]),
        foot_contact=torch.tensor([True, True]),
        forward_speed_mps=torch.tensor([0.2, 0.5]),
        previous_elapsed_sec=0.0,
        elapsed_sec=0.002,
    )


def test_first_contiguous_contact_then_late_rekick_has_no_credit():
    args = initial()
    before = {k: v.clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
    first = advance_first_launch_credit(**args)
    assert first.departed.tolist() == [False, True]
    assert first.eligible_contact.all()
    assert all(torch.equal(args[k], v) for k, v in before.items())
    args.update(
        departed=first.departed,
        closed=first.closed,
        departure_time_sec=first.departure_time_sec,
        previous_elapsed_sec=0.002,
        elapsed_sec=0.004,
        foot_contact=torch.tensor([True, False]),
    )
    separated = advance_first_launch_credit(**args)
    assert separated.closed.tolist() == [False, True]
    args.update(
        departed=separated.departed,
        closed=separated.closed,
        departure_time_sec=separated.departure_time_sec,
        previous_elapsed_sec=0.004,
        elapsed_sec=0.006,
        foot_contact=torch.tensor([True, True]),
        forward_speed_mps=torch.tensor([1.2, 1.2]),
    )
    later = advance_first_launch_credit(**args)
    assert later.eligible_contact.tolist() == [True, False]
    assert later.late_source_contact.tolist() == [False, True]
    assert later.departure_time_sec[1] == first.departure_time_sec[1]


def test_continuous_contact_cannot_keep_window_open_forever():
    args = initial()
    args["forward_speed_mps"] = torch.tensor([0.5, 0.5])
    for i in range(1, 65):
        args.update(previous_elapsed_sec=(i - 1) * 0.002, elapsed_sec=i * 0.002)
        state = advance_first_launch_credit(**args)
        args.update(
            departed=state.departed,
            closed=state.closed,
            departure_time_sec=state.departure_time_sec,
        )
    assert state.closed.all() and state.late_source_contact.all()


@pytest.mark.parametrize(
    "changes",
    [
        dict(elapsed_sec=float("nan")),
        dict(elapsed_sec=0.006),
        dict(elapsed_sec=0.0),
        dict(forward_speed_mps=torch.tensor([float("inf"), 0.4])),
        dict(forward_speed_mps=torch.tensor([101.0, 0.4])),
        dict(forward_speed_mps=torch.tensor([0.4, 0.4], requires_grad=True)),
        dict(foot_contact=torch.tensor([1, 1])),
        dict(closed=torch.tensor([True, False])),
        dict(departure_time_sec=torch.tensor([-0.5, -1.0])),
        dict(departed=torch.tensor([True, False]), departure_time_sec=torch.tensor([0.001, -1.0])),
    ],
)
def test_bad_history_or_clock_is_rejected(changes):
    args = initial()
    args.update(changes)
    with pytest.raises(ValueError):
        advance_first_launch_credit(**args)
