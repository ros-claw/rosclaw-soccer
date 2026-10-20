"""Bind anatomical command semantics across online learning and evaluation."""

from typing import Any


def command_protocol(point: str, position_gain: float) -> dict[str, Any]:
    if (
        type(point) is not str
        or point not in ("torso", "pelvis")
        or type(position_gain) not in (int, float)
        or position_gain not in (0.0, 1.0)
    ):
        raise ValueError("explicit anatomical command point and fixed 0/1 gain required")
    return dict(
        velocity_command_anatomical_point=point,
        velocity_command_position_feedback_gain=float(position_gain),
    )


def require_same_command_protocol(report: dict[str, Any], point: str, gain: float) -> None:
    """Legacy records mean torso/feedforward, not a qualification for pelvis."""
    expected = command_protocol(point, gain)
    actual = command_protocol(
        report.get("velocity_command_anatomical_point", "torso"),
        report.get("velocity_command_position_feedback_gain", 0.0),
    )
    if actual != expected:
        raise ValueError("online course requires a smoke of the same command protocol")


def ball_aware_checkpoint_protocol(
    report: dict[str, Any], checkpoint: dict[str, Any]
) -> dict[str, Any]:
    """The first 177D report omitted protocol fields; never default it to torso.

    Caller must authenticate the checkpoint bytes before this metadata check.
    The old immutable report is not rewritten or retroactively promoted.
    """
    from rosclaw_soccer.rsi.ball_aware_inputs import BALL_AWARE_ABI

    if any(
        source.get("ball_aware_abi") != BALL_AWARE_ABI or source.get("actor_observation_dim") != 177
        for source in (report, checkpoint)
    ):
        raise ValueError("explicit matching ball-aware177 ABI required")
    require_same_command_protocol(checkpoint, "pelvis", 1.0)
    result = command_protocol("pelvis", 1.0)
    for key, value in result.items():
        if key in report and report[key] != value:
            raise ValueError("report/checkpoint command protocol mismatch")
    return result
