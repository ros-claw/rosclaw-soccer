"""Independent kinematic-observation, own-network and PD standing audit.

Scratch mj_forward calls compute geometry only, never integrate physics.
Unlike the producer's observer, this reader reconstructs all160 fields itself.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_balance_recovery import body_feedback
from rosclaw_soccer.rsi.paid_continuity_audit import audit_terminal, identical, snapshot_fields


def _product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w, x, y, z = a
    v, i, j, k = b
    return np.array(
        [
            w * v - x * i - y * j - z * k,
            w * i + x * v + y * k - z * j,
            w * j - x * k + y * v + z * i,
            w * k + x * j - y * i + z * v,
        ],
        dtype=np.float64,
    )


class StandingObservationAuditor:
    def __init__(self, auditor: Any) -> None:
        import mujoco

        self.auditor = auditor
        self.data = mujoco.MjData(auditor.layout.model)
        self.evaluations = 0

    def observe(
        self, q: np.ndarray, v: np.ndarray, pose: np.ndarray, last: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        import mujoco

        a = self.auditor
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        mujoco.mj_forward(a.layout.model, self.data)
        self.evaluations += 1
        quat = self.data.xquat[a.layout.pelvis].astype(np.float64)
        quat = quat / np.linalg.norm(quat)
        w, x, y, z = quat
        gravity = np.array(
            [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)], np.float32
        )
        conjugate = np.array([w, -x, -y, -z])
        origin = self.data.xpos[a.layout.pelvis]

        def local(point: np.ndarray) -> np.ndarray:
            return _product(_product(conjugate, np.r_[0.0, point - origin]), quat)[1:]

        inverse = np.argsort(a.order)
        return np.concatenate(
            [
                pose.astype(np.float32),
                np.zeros(29, np.float32),
                gravity,
                np.zeros(3, np.float32),
                self.data.qvel[a.layout.rv + 3 : a.layout.rv + 6].astype(np.float32),
                (self.data.qpos[a.layout.qi] - a.offset)[inverse].astype(np.float32),
                self.data.qvel[a.layout.vi][inverse].astype(np.float32),
                last.astype(np.float32),
                local(self.data.xpos[a.layout.ball]).astype(np.float32),
                local(goal).astype(np.float32),
            ]
        )


def audit_neural_stand(
    auditor: Any,
    observer: Any,
    actor: Any,
    raw: dict[str, np.ndarray],
    row: dict[str, Any],
    *,
    entry: dict[str, np.ndarray],
    pose: np.ndarray,
    previous_action: np.ndarray,
    initial_memory: tuple[np.ndarray, np.ndarray],
    start_control: int,
    maximum_controls: int,
    count_inference: Callable[[], None] | None = None,
) -> dict[str, Any]:
    z = raw
    n = len(z["target"])
    if (
        not 1 <= n <= maximum_controls
        or n != row["controls"]
        or row["task"] != "static_neural_stand"
        or row["memory_mode"] != "carry"
        or row["native_steps"] != n * 10
        or row["neural_rows"] != n
        or row["old_exam_result"] is not False
        or row["no_motion_padding"] is not True
        or row["optimizer_updates"] != 0
        or row["hardware_authorized"] is not False
        or not z["neural_control"].all()
        or len(z["NN_action"]) != n
        or any(v.dtype.kind not in "biuf" or not np.isfinite(v).all() for v in z.values())
        or not np.array_equal(z["STAND_control"], np.arange(start_control, start_control + n))
        or not np.array_equal(z["STAND_reference_pose"], np.broadcast_to(pose, (n, 29)))
    ):
        raise ValueError("bounded independent static-standing task contract required")
    identical(snapshot_fields(z, "initial_"), entry)
    if (
        not np.array_equal(z["native_qpos_before"][0], entry["qpos"])
        or not np.array_equal(z["native_qvel_before"][0], entry["qvel"])
        or not np.isclose(z["time_before"][0], entry["time"], rtol=0, atol=1e-11)
        or not np.allclose(np.diff(z["time_before"]), 0.02, rtol=0, atol=1e-11)
    ):
        raise ValueError("standing body/clock reset detected")
    qb, vb = auditor.canonical(z["native_qpos_before"], z["native_qvel_before"])
    if not np.array_equal(qb[1:], z["qpos"][:-1, -1]) or not np.array_equal(
        vb[1:], z["qvel"][:-1, -1]
    ):
        raise ValueError("standing native control continuity differs")
    state = tuple(torch.from_numpy(t.copy()) for t in initial_memory)
    previous = previous_action.copy()
    goal = entry["mocap_pos"][auditor.goal_mocap_id]
    with torch.no_grad():
        for j in range(n):
            obs = observer.observe(
                z["native_qpos_before"][j], z["native_qvel_before"][j], pose, previous, goal
            )
            if not np.array_equal(obs, z["NN_obs"][j]):
                raise ValueError(
                    "standing observation does not match actual body/pose/goal/previousaction"
                )
            if any(
                not np.array_equal(t.numpy(), z[k][j])
                for t, k in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
            ):
                raise ValueError("standing private causal memory differs")
            mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
            if count_inference is not None:
                count_inference()
            if (
                not np.array_equal(mean[0, 0].numpy(), z["NN_action"][j])
                or not np.array_equal(mean[0, 0].numpy(), z["NN_nominal_action"][j])
                or any(
                    not np.array_equal(t.numpy(), z[k][j])
                    for t, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
                )
            ):
                raise ValueError("standing own frozen motor action/memory differs")
            previous = z["NN_action"][j]
    target = auditor.offset + z["NN_action"][:, auditor.order] * auditor.scale
    jq = np.concatenate((qb[:, None, 7:36], z["qpos"][:, :-1, 7:36]), axis=1)
    jv = np.concatenate((vb[:, None, 6:35], z["qvel"][:, :-1, 6:35]), axis=1)
    if not np.array_equal(target, z["target"]) or not np.array_equal(
        np.clip(
            auditor.kp * (target[:, None] - jq) - auditor.kd * jv, -auditor.effort, auditor.effort
        ),
        z["torque"],
    ):
        raise ValueError("standing original named29 joint target/every2ms torque law differs")
    failed = (
        (z["qpos"][:, :, 2] < 0.65)
        | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
    ).any(axis=1)
    if not np.array_equal(failed, z["body_failed"]) or failed[:-1].any() or (z["forces"] < 0).any():
        raise ValueError("standing actual body termination/forces differ")
    healthy = not bool(failed.any())
    q, v = z["qpos"][-1, -1], z["qvel"][-1, -1]
    feature = body_feedback(q, v)
    upright = float(1 - 2 * (q[4] ** 2 + q[5] ** 2))
    stable = bool(
        healthy
        and q[2] >= 0.72
        and upright >= 0.98
        and np.linalg.norm(feature[4:6]) <= 0.2
        and np.linalg.norm(v[3:6]) <= 0.3
    )
    settled = bool(q[38] <= 0.12 and abs(v[37]) <= 0.1)
    expected = dict(
        healthy=healthy,
        stability_observed=stable,
        terminal_feature=feature.tolist(),
        terminal_upright=upright,
        ball_settled=settled,
    )
    if any(row.get(k) != value for k, value in expected.items()):
        raise ValueError("standing body metrics differ")
    if healthy and n < maximum_controls and (start_control == 0 or not settled):
        raise ValueError("standing controls silently omitted")
    terminal = snapshot_fields(z, "terminal_")
    audit_terminal(auditor, z, terminal)
    for key in ("mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied", "act"):
        if not np.array_equal(terminal[key], entry[key]):
            raise ValueError("standing changed external goal/forces")
    return dict(
        native_steps=n * 10,
        neural_rows=n,
        healthy=healthy,
        stability_observed=stable,
        state=tuple(t.numpy().copy() for t in state),
        previous_action=previous,
        terminal=terminal,
    )
