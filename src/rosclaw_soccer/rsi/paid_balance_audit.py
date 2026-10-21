"""Read-only active-recovery proposal, continuity and every2ms motor audit."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_balance_recovery import body_feedback, feedback_delta


def audit_balance_recovery(
    auditor: Any,
    raw: dict[str, np.ndarray],
    *,
    entry: dict[str, np.ndarray],
    anchor: np.ndarray,
    anchor_action: np.ndarray,
    latent: np.ndarray,
) -> dict[str, Any]:
    z = raw
    n = len(z["target"])
    if (
        not 1 <= n <= 100
        or z["target"].shape != (n, 29)
        or z["neural_control"].any()
        or any(k.startswith(("NN_", "PLAN_")) for k in z)
        or any(v.dtype.kind not in "biuf" or not np.isfinite(v).all() for v in z.values())
    ):
        raise ValueError(
            "bounded finite NEW recovery controls, not neural/reference padding, required"
        )
    for key, value in entry.items():
        if not np.array_equal(z["initial_" + key], value):
            raise ValueError("recovery entry integration state changed")
    if (
        not np.array_equal(z["native_qpos_before"][0], entry["qpos"])
        or not np.array_equal(z["native_qvel_before"][0], entry["qvel"])
        or not np.isclose(z["time_before"][0], entry["time"], rtol=0, atol=1e-11)
        or not np.allclose(np.diff(z["time_before"]), 0.02, rtol=0, atol=1e-11)
        or not np.array_equal(z["RECOVERY_anchor_HG"], np.broadcast_to(anchor, (n, 29)))
    ):
        raise ValueError("active recovery reset body/clock or changed its fixed anchor")
    qb, vb = auditor.canonical(z["native_qpos_before"], z["native_qvel_before"])
    if not np.array_equal(qb[1:], z["qpos"][:-1, -1]) or not np.array_equal(
        vb[1:], z["qvel"][:-1, -1]
    ):
        raise ValueError("active recovery native body continuity differs")
    previous = np.zeros(29)
    for i in range(n):
        feature = body_feedback(qb[i], vb[i])
        requested = feedback_delta(feature, latent)
        delta = np.clip(requested, previous - 0.02, previous + 0.02)
        action = anchor_action.copy()
        if not delta.any():
            target = anchor.copy()
        else:
            action[auditor.order] = ((anchor + delta - auditor.offset) / auditor.scale).astype(
                np.float32
            )
            target = auditor.offset + action[auditor.order] * auditor.scale
        actual_delta = target - anchor
        if (
            not np.array_equal(feature, z["RECOVERY_body_feedback"][i])
            or not np.array_equal(requested, z["RECOVERY_requested_delta_HG"][i])
            or not np.array_equal(action, z["RECOVERY_actual_action"][i])
            or not np.array_equal(target, z["target"][i])
            or not np.array_equal(actual_delta, z["RECOVERY_actual_delta_HG"][i])
            or np.abs(actual_delta).max() > 0.2 + 1e-6
            or np.abs(actual_delta - previous).max() > 0.02 + 1e-6
        ):
            raise ValueError("bounded causal body feedback/action/slew reconstruction differs")
        previous = actual_delta
    jq = np.concatenate((qb[:, None, 7:36], z["qpos"][:, :-1, 7:36]), axis=1)
    jv = np.concatenate((vb[:, None, 6:35], z["qvel"][:, :-1, 6:35]), axis=1)
    if not np.array_equal(
        np.clip(
            auditor.kp * (z["target"][:, None] - jq) - auditor.kd * jv,
            -auditor.effort,
            auditor.effort,
        ),
        z["torque"],
    ):
        raise ValueError("actual every2ms recovery PD torque law differs")
    failed = (
        (z["qpos"][:, :, 2] < 0.65)
        | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
    ).any(axis=1)
    if (
        not np.array_equal(failed, z["body_failed"])
        or failed[:-1].any()
        or (not failed.any() and n != 100)
        or (z["forces"] < 0).any()
    ):
        raise ValueError("recovery body failure/termination contract differs")
    q, v = z["qpos"][-1, -1], z["qvel"][-1, -1]
    feature = body_feedback(q, v)
    upright = float(1 - 2 * (q[4] ** 2 + q[5] ** 2))
    healthy = not bool(failed.any())
    stable = bool(
        healthy
        and q[2] >= 0.72
        and upright >= 0.98
        and np.linalg.norm(feature[4:6]) <= 0.2
        and np.linalg.norm(v[3:6]) <= 0.3
    )
    cost = float(
        np.sum((feature[:2] / 0.2) ** 2)
        + np.sum((feature[2:4] / 0.3) ** 2)
        + np.sum((feature[4:6] / 0.2) ** 2)
        + (feature[6] / 0.1) ** 2
    )
    return dict(
        controls=n,
        native_steps=n * 10,
        healthy=healthy,
        recovery_stability_observed=stable,
        terminal_feature=feature.tolist(),
        terminal_upright=upright,
        cost=cost,
        next_skill_ready_proven=False,
        neural_optimizer_updates=0,
        hardware_authorized=False,
    )
