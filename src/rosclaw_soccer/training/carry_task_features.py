"""Causal task context for carry learning, not a sensor or execution path.

All twelve features use the supplied current measurement and its accumulated
history. No future trajectory, final success label or episode identifier enters
the policy. Saturation is explicit; this is not a proof of Markov sufficiency.
"""

from typing import Any

from rosclaw_soccer.training.ball_carry_credit import BallCarryCredit, ground_carry_success

CARRY_TASK_FEATURE_NAMES = (
    "ball_progress_over_5m",
    "root_progress_over_5m",
    "signed_ball_lane_over_5m",
    "signed_root_lane_over_5m",
    "separation_over_1m",
    "peak_separation_over_1m",
    "peak_ball_height_over_1m",
    "distinct_touches_over_4",
    "contact_clear_ticks_over_20",
    "heading_error_sin",
    "heading_error_cos",
    "physical_failure_latched",
)


def carry_task_features(
    history: BallCarryCredit,
    *,
    ball_position: Any,
    root_position: Any,
    root_quaternion: Any,
) -> Any:
    """Return private float32/64 N×12 context, bounded to [-1,1].

    Quaternion order is MuJoCo wxyz. Measurements must agree with the history's
    current planar progress and range; the caller still owns sensor provenance.
    """
    import torch

    ground_carry_success(history)  # Validate history, never expose its success label.
    for item, width in ((ball_position, 3), (root_position, 3), (root_quaternion, 4)):
        if (
            not isinstance(item, torch.Tensor)
            or item.shape != (len(history.direction), width)
            or item.dtype != history.direction.dtype
            or item.device != history.direction.device
            or item.layout != torch.strided
            or item.requires_grad
            or not bool(torch.isfinite(item).all())
            or bool((item.abs() > 1000).any())
        ):
            raise ValueError("aligned finite detached current carry measurement required")
    if bool((torch.linalg.vector_norm(root_quaternion, dim=1) - 1).abs().gt(1e-4).any()):
        raise ValueError("unit wxyz root quaternion required")
    direction = history.direction
    lateral = torch.stack((-direction[:, 1], direction[:, 0]), 1)
    ball_delta = ball_position[:, :2] - history.initial_ball[:, :2]
    root_delta = root_position[:, :2] - history.initial_root[:, :2]
    ball_progress = (ball_delta * direction).sum(1)
    root_progress = (root_delta * direction).sum(1)
    ball_lane = (ball_delta * lateral).sum(1)
    root_lane = (root_delta * lateral).sum(1)
    separation = torch.linalg.vector_norm(ball_position[:, :2] - root_position[:, :2], dim=1)
    for current, recorded in (
        (ball_progress, history.forward_progress),
        (root_progress, history.root_progress),
        (ball_lane.abs(), history.lateral_error),
        (separation, history.separation),
    ):
        if not bool(torch.allclose(current, recorded, rtol=0, atol=1e-4)):
            raise ValueError("current measurement and carry history disagree")
    w, x, y, z = root_quaternion.unbind(1)
    forward = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z)), 1)
    norm = torch.linalg.vector_norm(forward, dim=1)
    if bool((norm < 1e-3).any()):
        raise ValueError("root heading is undefined near vertical forward axis")
    forward = forward / norm[:, None]
    sine = forward[:, 0] * direction[:, 1] - forward[:, 1] * direction[:, 0]
    cosine = (forward * direction).sum(1)
    return torch.stack(
        (
            ball_progress / 5,
            root_progress / 5,
            ball_lane / 5,
            root_lane / 5,
            separation,
            history.maximum_separation,
            history.maximum_ball_height,
            history.touches.to(direction.dtype) / 4,
            history.clear_ticks.to(direction.dtype) / 20,
            sine,
            cosine,
            history.failed.to(direction.dtype),
        ),
        1,
    ).clamp(-1, 1)
