"""Measured ground-carry progress, separate from first-pass release credit.

Pure SIM_ONLY tensor bookkeeping. A caller must supply the owning player's
actual 500 Hz contacts and body/in-play verdicts; this is not a contact sensor,
possession claim, motor path, skill promotion or replacement for pass scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BallCarryCredit:
    initial_ball: Any
    initial_root: Any
    direction: Any
    clear_ticks: Any
    touches: Any
    failed: Any
    maximum_separation: Any
    maximum_ball_height: Any
    forward_progress: Any
    root_progress: Any
    lateral_error: Any
    separation: Any
    tick: int
    elapsed_sec: float


def _positions(ball: Any, root: Any, direction: Any) -> None:
    import torch

    if (
        not isinstance(ball, torch.Tensor)
        or ball.ndim != 2
        or ball.shape[1] != 3
        or not 1 <= len(ball) <= 4096
        or ball.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("bounded measured XYZ batch required")
    for value, shape in ((ball, ball.shape), (root, ball.shape), (direction, (len(ball), 2))):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != ball.dtype
            or value.device != ball.device
            or value.layout != torch.strided
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 1000).any())
        ):
            raise ValueError("aligned detached finite measured positions required")
    if bool((torch.linalg.vector_norm(direction, dim=1) - 1).abs().gt(1e-4).any()):
        raise ValueError("explicit unit ground-carry direction required")


def begin_ball_carry(*, ball_position: Any, root_position: Any, direction: Any) -> BallCarryCredit:
    """Start a new evaluation history, never reset a physical body or ball."""
    import torch

    _positions(ball_position, root_position, direction)
    n = len(ball_position)
    zero = ball_position.new_zeros(n)
    separation = torch.linalg.vector_norm(ball_position[:, :2] - root_position[:, :2], dim=1)
    return BallCarryCredit(
        initial_ball=ball_position.clone(),
        initial_root=root_position.clone(),
        direction=direction.clone(),
        clear_ticks=torch.full((n,), 20, dtype=torch.int64, device=ball_position.device),
        touches=torch.zeros(n, dtype=torch.int64, device=ball_position.device),
        failed=(ball_position[:, 2] < 0).clone(),
        maximum_separation=separation.clone(),
        maximum_ball_height=ball_position[:, 2].clone(),
        forward_progress=zero.clone(),
        root_progress=zero.clone(),
        lateral_error=zero.clone(),
        separation=separation,
        tick=0,
        elapsed_sec=0.0,
    )


def advance_ball_carry(
    history: BallCarryCredit,
    *,
    ball_position: Any,
    root_position: Any,
    foot_contact: Any,
    nonfoot_contact: Any,
    body_valid: Any,
    in_play: Any,
    tick: int,
    elapsed_sec: float,
) -> BallCarryCredit:
    """Count distinct touches only after >=40 ms of measured separation.

    Continuing contact counts once, including contact-force flicker shorter
    than 40 ms. Body, nonfoot and out-of-play failures latch for this history.
    Unlike first-pass credit, later legitimate own-foot touches are desirable.
    The caller must not reset a failed history to relabel the same trial.
    """
    import torch

    if not isinstance(history, BallCarryCredit):
        raise ValueError("explicit carry history required")
    if (
        type(history.tick) is not int
        or type(tick) is not int
        or not 1 <= tick <= 30000
        or tick != history.tick + 1
        or type(elapsed_sec) not in (int, float)
        or type(history.elapsed_sec) not in (int, float)
        or not math.isfinite(elapsed_sec)
        or not math.isfinite(history.elapsed_sec)
        or abs(history.elapsed_sec - history.tick * 0.002) > 1e-6
        or abs(elapsed_sec - tick * 0.002) > 1e-6
    ):
        raise ValueError("consecutive 500 Hz carry observations required")
    _positions(ball_position, root_position, history.direction)
    _positions(history.initial_ball, history.initial_root, history.direction)
    if (
        history.initial_ball.shape != ball_position.shape
        or history.initial_ball.device != ball_position.device
        or history.initial_ball.dtype != ball_position.dtype
    ):
        raise ValueError("carry history batch cannot change")
    n = len(ball_position)
    for value in (foot_contact, nonfoot_contact, body_valid, in_play, history.failed):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n,)
            or value.device != ball_position.device
            or value.dtype != torch.bool
            or value.layout != torch.strided
        ):
            raise ValueError("aligned measured boolean carry verdicts required")
    for value, maximum in ((history.clear_ticks, 20), (history.touches, history.tick)):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n,)
            or value.device != ball_position.device
            or value.dtype != torch.int64
            or value.layout != torch.strided
            or bool(((value < 0) | (value > maximum)).any())
        ):
            raise ValueError("bounded carry contact history required")
    for value in (history.maximum_separation, history.maximum_ball_height):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n,)
            or value.dtype != ball_position.dtype
            or value.device != ball_position.device
            or value.requires_grad
            or value.layout != torch.strided
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 3000).any())
        ):
            raise ValueError("finite detached carry extent history required")
    initial_separation = torch.linalg.vector_norm(
        history.initial_ball[:, :2] - history.initial_root[:, :2], dim=1
    )
    if bool(
        (
            (history.maximum_separation < initial_separation)
            | (history.maximum_ball_height < history.initial_ball[:, 2])
        ).any()
    ):
        raise ValueError("carry maxima cannot discard initial measurements")
    separation = torch.linalg.vector_norm(ball_position[:, :2] - root_position[:, :2], dim=1)
    delta = ball_position[:, :2] - history.initial_ball[:, :2]
    progress = (delta * history.direction).sum(1)
    lateral = (delta[:, 0] * history.direction[:, 1] - delta[:, 1] * history.direction[:, 0]).abs()
    return BallCarryCredit(
        initial_ball=history.initial_ball.clone(),
        initial_root=history.initial_root.clone(),
        direction=history.direction.clone(),
        clear_ticks=torch.where(foot_contact, 0, (history.clear_ticks + 1).clamp(max=20)),
        touches=history.touches + (foot_contact & (history.clear_ticks >= 20)).long(),
        failed=history.failed
        | nonfoot_contact
        | ~body_valid
        | ~in_play
        | (ball_position[:, 2] < 0),
        maximum_separation=torch.maximum(history.maximum_separation, separation),
        maximum_ball_height=torch.maximum(history.maximum_ball_height, ball_position[:, 2]),
        forward_progress=progress,
        root_progress=(
            (root_position[:, :2] - history.initial_root[:, :2]) * history.direction
        ).sum(1),
        lateral_error=lateral,
        separation=separation,
        tick=tick,
        elapsed_sec=float(elapsed_sec),
    )


def ground_carry_success(history: BallCarryCredit) -> Any:
    """Fixed local skill goal; not pass precision or full-match promotion.

    One metre forward ball progress, half a metre root progress, >=2 distinct
    touches, <=0.25 m lateral error, <=0.75 m peak root/ball separation and
    <=0.65 m final separation, ball centre never above0.5 m, no latched failure.
    Only pass freshly returned, independently recorded history to this scorer.
    """
    import torch

    if not isinstance(history, BallCarryCredit):
        raise ValueError("explicit carry history required")
    _positions(history.initial_ball, history.initial_root, history.direction)
    n = len(history.initial_ball)
    if (
        type(history.tick) is not int
        or not 0 <= history.tick <= 30000
        or type(history.elapsed_sec) not in (int, float)
        or not math.isfinite(history.elapsed_sec)
        or abs(history.elapsed_sec - history.tick * 0.002) > 1e-6
    ):
        raise ValueError("bounded carry history clock required")
    for value, dtype in ((history.failed, torch.bool), (history.touches, torch.int64)):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n,)
            or value.dtype != dtype
            or value.device != history.initial_ball.device
            or value.layout != torch.strided
        ):
            raise ValueError("aligned terminal carry history required")
    if bool(((history.touches < 0) | (history.touches > history.tick)).any()):
        raise ValueError("impossible carry touch count")
    for value in (
        history.forward_progress,
        history.root_progress,
        history.lateral_error,
        history.separation,
        history.maximum_separation,
        history.maximum_ball_height,
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n,)
            or value.dtype != history.initial_ball.dtype
            or value.device != history.initial_ball.device
            or value.layout != torch.strided
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 3000).any())
        ):
            raise ValueError("finite detached terminal carry measurements required")
    if bool(
        (
            (history.separation < 0)
            | (history.lateral_error < 0)
            | (history.maximum_separation < history.separation)
            | (history.maximum_ball_height < history.initial_ball[:, 2])
        ).any()
    ):
        raise ValueError("inconsistent carry extent history")
    return (
        ~history.failed
        & (history.forward_progress >= 1.0)
        & (history.root_progress >= 0.5)
        & (history.touches >= 2)
        & (history.lateral_error <= 0.25)
        & (history.maximum_separation <= 0.75)
        & (history.separation <= 0.65)
        & (history.maximum_ball_height <= 0.5)
    )


def ground_carry_potential(history: BallCarryCredit) -> Any:
    """Bounded [0,6] training signal, not a success label or motion proposal.

    Ball/root progress contribute at most5/1 respectively, with lateral and
    separation penalties. A recorded physical failure, loss of close control
    or high ball zeros the potential permanently within this history. A
    trainer must bind its discount/terminal/reward semantics separately and
    evaluate actual success with ``ground_carry_success``, not this value.
    """
    ground_carry_success(history)  # validate history without granting task success
    valid = (
        ~history.failed
        & (history.maximum_separation <= 0.75)
        & (history.maximum_ball_height <= 0.5)
    )
    value = (
        5 * history.forward_progress.clamp(0, 1)
        + 2 * history.root_progress.clamp(0, 0.5)
        - 2 * history.lateral_error.clamp(0, 1)
        - 2 * (history.separation - 0.45).clamp(0, 1)
    ).clamp(0, 6)
    return value * valid.to(value.dtype)
