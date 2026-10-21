"""Independent integration-state/fixture checks for continuing skill chains."""

from typing import Any

import numpy as np

INTEGRATION_FIELDS = frozenset(
    (
        "qpos",
        "qvel",
        "ctrl",
        "qacc_warmstart",
        "mocap_pos",
        "mocap_quat",
        "qfrc_applied",
        "xfrc_applied",
        "act",
        "time",
    )
)


def require_integration_fields(state: dict[str, np.ndarray]) -> None:
    if set(state) != INTEGRATION_FIELDS or any(not np.isfinite(x).all() for x in state.values()):
        raise ValueError("complete finite integration state, including warmstart/forces, required")


def identical(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    if set(before) != set(after) or any(not np.array_equal(before[k], after[k]) for k in before):
        raise ValueError("complete physical integration snapshot differs")


def snapshot_fields(raw: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {k.removeprefix(prefix): v for k, v in raw.items() if k.startswith(prefix)}


def audit_terminal(auditor: Any, raw: dict[str, np.ndarray], state: dict[str, np.ndarray]) -> None:
    require_integration_fields(state)
    q, v = auditor.canonical(state["qpos"], state["qvel"])
    if (
        not np.array_equal(q, raw["qpos"][-1, -1])
        or not np.array_equal(v, raw["qvel"][-1, -1])
        or not np.array_equal(state["ctrl"][auditor.layout.ai], raw["torque"][-1, -1])
        or not np.isclose(state["time"], raw["time_before"][-1] + 0.02, rtol=0, atol=1e-11)
    ):
        raise ValueError("physical terminal state/control/time differs")


def audit_practice_boundary(
    auditor: Any,
    *,
    previous: dict[str, np.ndarray],
    terminal: dict[str, np.ndarray],
    drop: dict[str, np.ndarray],
    hold: dict[str, np.ndarray] | None,
    boundary: dict[str, np.ndarray] | None,
    source: str,
) -> dict[str, Any]:
    before = snapshot_fields(drop, "before_")
    require_integration_fields(before)
    require_integration_fields(terminal)
    identical(before, terminal)
    q, _ = auditor.canonical(before["qpos"], before["qvel"])
    performed = bool(drop["performed"])
    if performed != bool(np.linalg.norm(q[36:38] - q[:2]) >= 2):
        raise ValueError("practice fixture ball-return eligibility differs")
    native = 0
    if performed:
        after = snapshot_fields(drop, "after_")
        require_integration_fields(after)
        modified = {k: v.copy() for k, v in after.items()}
        layout = auditor.layout
        modified["qpos"][layout.bq : layout.bq + 7] = before["qpos"][layout.bq : layout.bq + 7]
        modified["qvel"][layout.bv : layout.bv + 6] = before["qvel"][layout.bv : layout.bv + 6]
        identical(before, modified)
        direction = before["mocap_pos"][auditor.goal_mocap_id, :2] - q[:2]
        direction /= np.linalg.norm(direction)
        sign = 1 if "left" in source else -1
        point = q[:2] + 0.8 * direction + sign * 0.15 * np.array([-direction[1], direction[0]])
        if (
            not np.array_equal(
                after["qpos"][layout.bq : layout.bq + 7], np.r_[point, 0.4, 1.0, 0, 0, 0]
            )
            or after["qvel"][layout.bv : layout.bv + 6].any()
        ):
            raise ValueError("declared ball drop position/velocity differs")
        if hold is None:
            raise ValueError("performed ball drop requires actual settle physics")
        checked = auditor.audit_hold(hold, after, previous["target"][-1])
        native = checked["native_steps"]
        if checked["body_failed"] or not checked["settled"]:
            if boundary is not None:
                raise ValueError("failed hold cannot restart next skill")
            return dict(
                status="HOLD_BODY_FAILURE" if checked["body_failed"] else "BALL_NOT_SETTLED",
                native_steps=native,
                state=None,
            )
    elif hold is not None or snapshot_fields(drop, "after_"):
        raise ValueError("an unperformed return must not add physics or state writes")
    if boundary is None:
        raise ValueError("healthy settled practice must record its real handoff")
    left, right = snapshot_fields(boundary, "before_"), snapshot_fields(boundary, "after_")
    require_integration_fields(left)
    require_integration_fields(right)
    identical(left, right)
    if performed:
        if hold is None:
            raise ValueError("settled ball drop requires its physical hold record")
        audit_terminal(auditor, hold, left)
    else:
        identical(terminal, left)
    if not np.array_equal(boundary["previous_actual_action"], previous["NN_action"][-1]):
        raise ValueError("actual prior executed action was erased")
    return dict(status="READY", native_steps=native, state=right)
