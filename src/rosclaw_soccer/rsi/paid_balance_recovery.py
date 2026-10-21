"""Explicit bounded SIM recovery primitive, not reference padding or a policy.

The teacher uses measured body feedback; gains must be physically optimized
and independently verified before producing learning labels. This adds a NEW
post-reference phase and never changes the original skill's score or horizon.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def body_feedback(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """[roll,pitch,wx,wy,local_vx,local_vy,height_error,local_vz].

    MuJoCo free-joint translational qvel is world-frame, rotational qvel is
    body-local: https://mujoco.readthedocs.io/en/stable/overview.html#floating-objects
    Use only the current PRE-control state, never a future trajectory sample.
    """
    if q.shape != (43,) or v.shape != (41,) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("finite actual canonical43/41 state required")
    w, x, y, z = q[3:7]
    if not np.isclose(np.dot(q[3:7], q[3:7]), 1, rtol=0, atol=1e-6):
        raise ValueError("actual unit body quaternion required; no pose repair")
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    up = rotation[2]
    local_v = rotation.T @ v[:3]
    return np.array(
        [
            np.arctan2(up[1], up[2]),
            np.arctan2(-up[0], up[2]),
            v[3],
            v[4],
            local_v[0],
            local_v[1],
            q[2] - 0.78,
            local_v[2],
        ]
    )


def feedback_delta(feature: np.ndarray, latent: np.ndarray) -> np.ndarray:
    if (
        feature.shape != (8,)
        or latent.shape != (9, 4)
        or not np.isfinite(feature).all()
        or not np.isfinite(latent).all()
        or np.abs(latent).max() > 1
    ):
        raise ValueError("finite causal8 features and bounded9x4 feedback gains required")
    sagittal = feature[[1, 3, 4, 6]] / np.array([0.3, 1.0, 0.5, 0.1])
    lateral = feature[[0, 2, 5, 6]] / np.array([0.3, 1.0, 0.5, 0.1])
    channels = (
        "hip_pitch",
        "knee",
        "ankle_pitch",
        "waist_pitch",
        "shoulder_pitch",
        "hip_roll",
        "ankle_roll",
        "waist_roll",
        "shoulder_roll",
    )
    result = np.zeros(29)
    for i, channel in enumerate(channels):
        offset = float(0.2 * np.tanh(2 * np.dot(latent[i], sagittal if i < 5 else lateral)))
        names = (
            (channel + "_joint",)
            if channel.startswith("waist")
            else tuple(side + "_" + channel + "_joint" for side in ("left", "right"))
        )
        for name in names:
            result[G1_DDS_JOINT_NAMES.index(name)] = offset
    return result


class BalanceRecovery:
    """At most 100 NEW 20ms controls; fixed anchor, 0.2rad/0.02rad-step bounds."""

    def __init__(self, ep: Any) -> None:
        if not ep.done or ep.termination_reason != "REFERENCE_COMPLETE":
            raise ValueError("healthy completed actual skill required for recovery")
        self.ep = ep
        self.anchor = ep.target.copy()
        self.anchor_action = ep.last_action.copy()
        self.delta = np.zeros(29)
        self.controls = 0

    def step(self, requested: np.ndarray) -> dict[str, Any]:
        import mujoco

        ep = self.ep
        if (
            self.controls >= 100
            or ep.termination_reason != "REFERENCE_COMPLETE"
            or requested.shape != (29,)
            or not np.isfinite(requested).all()
            or np.abs(requested).max() > 0.2 + 1e-12
        ):
            raise ValueError("bounded active recovery cannot restart failures or extend its budget")
        delta = np.clip(requested, self.delta - 0.02, self.delta + 0.02)
        if not delta.any():
            ep.last_action = self.anchor_action.copy()
            ep.target = self.anchor.copy()
        else:
            action = self.anchor_action.copy()
            action[ep.order] = ((self.anchor + delta - ep.offset) / ep.scale).astype(np.float32)
            ep.last_action = action
            ep.target = ep.offset + action[ep.order] * ep.scale
        self.delta = ep.target - self.anchor
        if np.abs(self.delta).max() > 0.2 + 1e-6:
            raise ValueError("actual rounded recovery target escaped the fixed anchor bound")
        mujoco.mj_forward(ep.env.model, ep.env.data)
        row: dict[str, Any] = ep._integrate_target(False)
        row.update(
            RECOVERY_requested_delta_HG=requested.copy(),
            RECOVERY_actual_delta_HG=self.delta.copy(),
            RECOVERY_actual_action=ep.last_action.copy(),
            RECOVERY_anchor_HG=self.anchor.copy(),
        )
        self.controls += 1
        if row["body_failed"]:
            ep.termination_reason = "BODY_FAILURE"
        return row


def collect_balance_recovery(
    ep: Any, latent: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    recovery = BalanceRecovery(ep)
    rows = []
    for _ in range(100):
        q, v = ep.layout.canonical(ep.env.data)
        feature = body_feedback(q, v)
        row = recovery.step(feedback_delta(feature, latent))
        row["RECOVERY_body_feedback"] = feature
        rows.append(row)
        if row["body_failed"]:
            break
    raw = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    q, v = raw["qpos"][-1, -1], raw["qvel"][-1, -1]
    feature = body_feedback(q, v)
    upright = float(1 - 2 * (q[4] ** 2 + q[5] ** 2))
    healthy = not bool(raw["body_failed"].any())
    certificate = bool(
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
    return raw, dict(
        controls=len(rows),
        native_steps=len(rows) * 10,
        healthy=healthy,
        recovery_stability_observed=certificate,
        terminal_feature=feature.tolist(),
        terminal_upright=upright,
        cost=cost,
        next_skill_ready_proven=False,
        neural_optimizer_updates=0,
        hardware_authorized=False,
    )
