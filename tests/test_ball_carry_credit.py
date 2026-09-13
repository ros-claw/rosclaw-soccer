from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from rosclaw_soccer.training.ball_carry_credit import (  # noqa: E402
    advance_ball_carry,
    begin_ball_carry,
    ground_carry_success,
)


def initial():
    return begin_ball_carry(
        ball_position=torch.tensor([[0.25, 0.0, 0.115]]),
        root_position=torch.tensor([[0.0, 0.0, 0.75]]),
        direction=torch.tensor([[1.0, 0.0]]),
    )


def step(history, *, contact=False, x=0.0, y=0.0, root_x=None, **overrides):
    args = dict(
        ball_position=torch.tensor([[0.25 + x, y, 0.115]]),
        root_position=torch.tensor([[x if root_x is None else root_x, 0.0, 0.75]]),
        foot_contact=torch.tensor([contact]),
        nonfoot_contact=torch.tensor([False]),
        body_valid=torch.tensor([True]),
        in_play=torch.tensor([True]),
        tick=history.tick + 1,
        elapsed_sec=(history.tick + 1) * 0.002,
    )
    args.update(overrides)
    return advance_ball_carry(history, **args)


def successful():
    h = step(initial(), contact=True)
    for i in range(20):
        h = step(h, x=(i + 1) / 20)
    return step(h, contact=True, x=1.0)


def test_distinct_touches_and_forward_following_succeed():
    h = successful()
    assert h.touches.item() == 2
    assert ground_carry_success(h).item()


def test_continuous_contact_and_short_flicker_do_not_count_as_dribbling():
    h = step(initial(), contact=True)
    for _ in range(40):
        h = step(h, contact=True)
    for _ in range(19):
        h = step(h)
    h = step(h, contact=True)
    assert h.touches.item() == 1
    assert not ground_carry_success(h).item()


@pytest.mark.parametrize("flag", ["body_valid", "in_play", "nonfoot_contact"])
def test_physical_failure_latches(flag):
    h = step(successful(), x=1.0, **{flag: torch.tensor([flag == "nonfoot_contact"])})
    h = step(h, x=1.0)
    assert h.failed.item()
    assert not ground_carry_success(h).item()


def test_stationary_robot_cannot_claim_a_long_kick_is_carrying():
    h = step(successful(), x=1.0, root_x=0.0)
    assert not ground_carry_success(h).item()
    h = step(h, x=1.0)
    assert not ground_carry_success(h).item()  # previous loss of close control retained


def test_high_ball_is_not_ground_carry_even_after_it_lands():
    h = step(successful(), x=1.0, ball_position=torch.tensor([[1.25, 0.0, 0.7]]))
    h = step(h, x=1.0)
    assert not ground_carry_success(h).item()


def test_sideways_ball_does_not_meet_forward_lane_goal():
    assert not ground_carry_success(step(successful(), x=1.0, y=0.3)).item()


@pytest.mark.parametrize(
    "overrides",
    [
        {"tick": 0},
        {"tick": 2},
        {"tick": True},
        {"elapsed_sec": 0.003},
        {"elapsed_sec": float("nan")},
    ],
)
def test_bad_clock_rejected(overrides):
    with pytest.raises(ValueError):
        step(initial(), **overrides)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1001.0])
def test_invalid_positions_rejected(value):
    with pytest.raises(ValueError):
        step(initial(), ball_position=torch.tensor([[value, 0.0, 0.115]]))


def test_inputs_and_prior_history_not_mutated_or_aliased():
    h = initial()
    old = h.initial_ball.clone()
    newer = step(h, contact=True)
    newer.initial_ball.fill_(9)
    assert torch.equal(h.initial_ball, old)
    assert h.touches.item() == 0
    assert h.tick == 0


@pytest.mark.parametrize("field", ["forward_progress", "maximum_separation", "maximum_ball_height"])
def test_nonfinite_terminal_history_not_silently_scored(field):
    with pytest.raises(ValueError):
        ground_carry_success(replace(successful(), **{field: torch.tensor([float("nan")])}))


def test_batch_and_tensor_grad_contracts():
    with pytest.raises(ValueError):
        step(initial(), root_position=torch.ones((2, 3)))
    with pytest.raises(ValueError):
        step(initial(), ball_position=torch.ones((1, 3), requires_grad=True))
    with pytest.raises(ValueError):
        step(initial(), foot_contact=torch.ones(1))


def test_no_touch_initial_history_is_not_success():
    assert not ground_carry_success(initial()).item()


def test_below_ground_ball_latches_failure():
    h = step(successful(), x=1.0, ball_position=torch.tensor([[1.25, 0.0, -0.01]]))
    assert not ground_carry_success(step(h, x=1.0)).item()


def test_extent_history_cannot_be_reset_to_hide_loss_of_control():
    with pytest.raises(ValueError):
        step(replace(initial(), maximum_separation=torch.tensor([-1.0])))


def test_one_metre_threshold_is_not_relaxed():
    h = successful()
    assert not ground_carry_success(replace(h, forward_progress=torch.tensor([1.0 - 1e-6]))).item()
