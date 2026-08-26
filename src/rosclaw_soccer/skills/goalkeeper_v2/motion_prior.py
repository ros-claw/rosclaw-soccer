"""Soccer specialization of the train-only conditional teacher contract."""

from __future__ import annotations

from rosclaw.continual.teacher_prior import ConditionalTeacherPriorContract


def goalkeeper_motion_prior_contract(
    *,
    artifact_hash: str,
    body_hash: str,
) -> ConditionalTeacherPriorContract:
    """Define the region/task vocabulary without bundling motion data or code."""

    return ConditionalTeacherPriorContract(
        prior_id="soccer.goalkeeper.motion_teacher.v2",
        artifact_hash=artifact_hash,
        body_hash=body_hash,
        observation_names=(
            "gravity_orientation",
            "angular_velocity",
            "joint_position",
            "joint_velocity",
        ),
        output_names=("target_position", "target_velocity"),
        condition_vocabulary={
            "task": ("ready", "shuffle", "save", "landing", "recovery"),
            "region": (
                "center",
                "upper_left",
                "upper_right",
                "lower_left",
                "lower_right",
            ),
        },
    )


__all__ = ["goalkeeper_motion_prior_contract"]
