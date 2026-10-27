"""Causally align recorded native two-player pass transitions for offline SIM."""

from collections.abc import Mapping

import numpy as np


def actual_pass_transitions(
    trace: Mapping[str, np.ndarray], *, teacher: bool
) -> dict[str, np.ndarray]:
    """Preserve before-action physical state and actually applied action.

    Contact forces and body failure are post-action measurements, never inputs
    to the same transition. This transformation does not establish success or
    certify global substep safety.
    """
    if type(teacher) is not bool:
        raise ValueError("explicit teacher mode required")
    fields = (
        "initial_qpos",
        "initial_qvel",
        "qpos",
        "qvel",
        "ball_forces",
        "time_before",
        "A_NN_action",
        "B_V_obs",
        "B_V_action",
        "B_V_command",
        "body_failed",
    )
    if any(k not in trace for k in fields):
        raise ValueError("complete recorded source arrays required")
    arrays = {key: np.asarray(trace[key]) for key in fields}
    qpos, qvel = arrays["qpos"], arrays["qvel"]
    if qpos.ndim != 3 or not 1 <= len(qpos) <= 300:
        raise ValueError("bounded native control timeline required")
    n = len(qpos)
    shapes = {
        "initial_qpos": (79,),
        "initial_qvel": (76,),
        "qpos": (n, 10, 79),
        "qvel": (n, 10, 76),
        "ball_forces": (n, 10, 2, 3),
        "time_before": (n,),
        "A_NN_action": (n, 29),
        "B_V_obs": (n, 98),
        "B_V_action": (n, 29),
        "B_V_command": (n, 3),
        "body_failed": (n, 2),
    }
    for key, shape in shapes.items():
        a = arrays[key]
        if a.shape != shape or a.dtype.kind not in "bfiu" or not np.isfinite(a).all():
            raise ValueError(f"invalid recorded numeric transition field: {key}")
    if arrays["body_failed"].dtype != np.bool_ or np.any(np.diff(arrays["time_before"]) <= 0):
        raise ValueError("boolean body failures and monotone control time required")
    if teacher:
        if "B_CAP_teacher_applied_action" not in trace or "B_CAP_teacher_active" not in trace:
            raise ValueError("actual teacher-applied action and activity required")
        applied = np.asarray(trace["B_CAP_teacher_applied_action"])
        active = np.asarray(trace["B_CAP_teacher_active"])
        if (
            applied.shape != (n, 29)
            or active.shape != (n,)
            or applied.dtype.kind not in "fiu"
            or active.dtype.kind not in "bfiu"
            or not np.isfinite(applied).all()
            or not np.isfinite(active).all()
            or not np.isin(active, [0, 1]).all()
        ):
            raise ValueError("finite actual teacher action and boolean activity required")
        active = active.astype(bool)
    else:
        applied, active = arrays["B_V_action"], np.zeros(n, dtype=bool)
    return {
        "time_before": arrays["time_before"].copy(),
        "qpos_before": np.concatenate((arrays["initial_qpos"][None], qpos[:-1, -1]), axis=0),
        "qvel_before": np.concatenate((arrays["initial_qvel"][None], qvel[:-1, -1]), axis=0),
        "passer_action": arrays["A_NN_action"].copy(),
        "receiver_observation": arrays["B_V_obs"].copy(),
        "receiver_action_applied": applied.copy(),
        "receiver_base_action": arrays["B_V_action"].copy(),
        "receiver_command": arrays["B_V_command"].copy(),
        "teacher_active": active.copy(),
        "ball_force_max_after": arrays["ball_forces"].max(axis=1),
        "body_failed_after": arrays["body_failed"].copy(),
    }
