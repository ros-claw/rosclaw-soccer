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
