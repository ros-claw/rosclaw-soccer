"""Read-only reconstruction of research motor-episode primitive evidence.

No neural inference, simulation stepping, search or promotion. Raw nonselected
contact forces are audited as recorded; independently replay physics separately.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_contact_window import native_contact_window


class PaidPrimitiveAuditor:
    def __init__(
        self,
        *,
        layout: Any,
        parameters: dict[str, np.ndarray],
        order: np.ndarray,
        effort: np.ndarray,
        goal_mocap_id: int,
    ) -> None:
        self.layout, self.order, self.effort = layout, order, effort
        self.offset, self.scale, self.kp, self.kd = [
            parameters[k][order]
            for k in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping")
        ]
        self.goal_mocap_id = goal_mocap_id

    def canonical(self, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.layout
        return (
            np.concatenate(
                (q[..., x.rq : x.rq + 7], q[..., x.qi], q[..., x.bq : x.bq + 7]), axis=-1
            ),
            np.concatenate(
                (v[..., x.rv : x.rv + 6], v[..., x.vi], v[..., x.bv : x.bv + 6]), axis=-1
            ),
        )

    def audit_hold(
        self, z: dict[str, np.ndarray], entry: dict[str, np.ndarray], target: np.ndarray
    ) -> dict[str, Any]:
        """Declared <=1s physical PD settling, not new NN/reference frames."""
        n = len(z["target"])
        if (
            not 1 <= n <= 50
            or any(k.startswith("NN_") for k in z)
            or any(not np.isfinite(v).all() for v in z.values())
            or z["neural_control"].any()
            or not z["post_reference_hold"].all()
            or not np.array_equal(z["target"], np.broadcast_to(target, z["target"].shape))
        ):
            raise ValueError("bounded held-target physics, never neural/ref padding, required")
        qb, vb = self.canonical(z["native_qpos_before"], z["native_qvel_before"])
        eq, ev = self.canonical(entry["qpos"], entry["qvel"])
        if (
            not np.array_equal(qb[0], eq)
            or not np.array_equal(vb[0], ev)
            or not np.array_equal(qb[1:], z["qpos"][:-1, -1])
            or not np.array_equal(vb[1:], z["qvel"][:-1, -1])
            or not np.isclose(z["time_before"][0], entry["time"], rtol=0, atol=1e-11)
            or not np.allclose(np.diff(z["time_before"]), 0.02, rtol=0, atol=1e-11)
        ):
            raise ValueError("hold body/time must remain physically continuous")
        jq = np.concatenate((qb[:, None, 7:36], z["qpos"][:, :-1, 7:36]), axis=1)
        jv = np.concatenate((vb[:, None, 6:35], z["qvel"][:, :-1, 6:35]), axis=1)
        if not np.array_equal(
            np.clip(
                self.kp * (z["target"][:, None] - jq) - self.kd * jv, -self.effort, self.effort
            ),
            z["torque"],
        ):
            raise ValueError("actual hold bounded PD torque differs")
        bad = (
            (z["qpos"][:, :, 2] < 0.65)
            | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
        ).any(axis=1)
        if not np.array_equal(bad, z["body_failed"]) or bad[:-1].any():
            raise ValueError("hold failure must terminate at its first control boundary")
        return dict(
            native_steps=n * 10,
            body_failed=bool(bad.any()),
            settled=bool(z["qpos"][-1, -1, 38] <= 0.12 and abs(z["qvel"][-1, -1, 37]) <= 0.1),
        )

    def audit(
        self,
        z: dict[str, np.ndarray],
        row: dict[str, Any],
        *,
        horizon: int,
        previous_action: np.ndarray,
        initial_memory: tuple[np.ndarray, np.ndarray] | None = None,
        intent: tuple[int, np.ndarray] | None = None,
        reference_prefix: bool = False,
    ) -> dict[str, Any]:
        from rosclaw_soccer.rsi.paid_foot_taskspace import phase_intent

        if any(not np.isfinite(x).all() for x in z.values()):
            raise ValueError("nonfinite primitive evidence")
        controls, n = len(z["target"]), len(z["NN_action"])
        if type(reference_prefix) is not bool or (
            reference_prefix
            and (
                controls != n
                or row.get("reference_prefix_only") is not True
                or row.get("official_exam_result") is not False
                or row.get("held_tail_controls_omitted") != 60
            )
        ):
            raise ValueError("new reference-prefix audit must not masquerade as old complete exam")
        if (
            not 1 <= horizon <= 300
            or n != min(controls, horizon)
            or n != row["neural_rows"]
            or controls * 10 != row["native_steps"]
            or not np.array_equal(z["neural_control"], np.arange(controls) < n)
            or not 0 <= controls - n <= 60
        ):
            raise ValueError("actual reference/neural/native clock differs")
        h, c = (
            (np.zeros((2, 1, 128), np.float32), np.zeros((2, 1, 128), np.float32))
            if initial_memory is None
            else initial_memory
        )
        if (
            not np.array_equal(z["NN_h_before"][0], h)
            or not np.array_equal(z["NN_c_before"][0], c)
            or not np.array_equal(z["NN_h_before"][1:], z["NN_h_after"][:-1])
            or not np.array_equal(z["NN_c_before"][1:], z["NN_c_after"][:-1])
            or not np.array_equal(z["NN_obs"][0, 125:154], previous_action)
            or not np.array_equal(z["NN_obs"][1:, 125:154], z["NN_action"][:-1])
        ):
            raise ValueError("actual per-player neural memory/action history differs")
        nominal = z["NN_nominal_action"]
        if intent is None:
            if any(k.startswith("PLAN_") for k in z) or not np.array_equal(nominal, z["NN_action"]):
                raise ValueError("unassisted neural action differs")
        else:
            center, latent = intent
            expected = np.stack([phase_intent(t, center, latent) for t in range(n)])
            delta = z["PLAN_joint_delta_HG"]
            if (
                not np.array_equal(expected, z["PLAN_pose_offset"])
                or np.abs(delta).max() > 0.2 + 1e-12
            ):
                raise ValueError("declared bounded Cartesian teacher differs")
            action = nominal.copy()
            action[:, self.order] = (action[:, self.order] + delta / self.scale).astype(np.float32)
            if not np.array_equal(action, z["NN_action"]):
                raise ValueError("teacher action not bound to neural proposal and actual IK delta")
        targets = self.offset + z["NN_action"][:, self.order] * self.scale
        if not np.array_equal(targets, z["target"][:n]) or not np.array_equal(
            z["target"][n:], np.broadcast_to(targets[-1], z["target"][n:].shape)
        ):
            raise ValueError("named joint target or held tail differs")
        qb, vb = self.canonical(z["native_qpos_before"], z["native_qvel_before"])
        initial_q, initial_v = self.canonical(z["initial_qpos"], z["initial_qvel"])
        if (
            not np.array_equal(qb[0], initial_q)
            or not np.array_equal(vb[0], initial_v)
            or not np.array_equal(qb[1:], z["qpos"][:-1, -1])
            or not np.array_equal(vb[1:], z["qvel"][:-1, -1])
            or not np.isclose(z["time_before"][0], z["initial_time"], rtol=0, atol=1e-11)
            or not np.allclose(np.diff(z["time_before"]), 0.02, rtol=0, atol=1e-11)
        ):
            raise ValueError("native body/time continuity differs")
        jq = np.concatenate((qb[:, None, 7:36], z["qpos"][:, :-1, 7:36]), axis=1)
        jv = np.concatenate((vb[:, None, 6:35], z["qvel"][:, :-1, 6:35]), axis=1)
        if not np.array_equal(
            np.clip(
                self.kp * (z["target"][:, None] - jq) - self.kd * jv, -self.effort, self.effort
            ),
            z["torque"],
        ):
            raise ValueError("actual every2ms bounded torque law differs")
        q, v, f = [z[k].reshape(-1, w) for k, w in (("qpos", 43), ("qvel", 41), ("forces", 3))]
        if (f < 0).any():
            raise ValueError("negative recorded force norm")
        goal = z["initial_mocap_pos"][self.goal_mocap_id]
        post = native_contact_window(
            force=f[:, :2],
            canonical_qpos=q,
            canonical_qvel=v,
            goal=goal,
            intended_foot=0 if "left" in row["source"] else 1,
        )
        bad = (
            (z["qpos"][:, :, 2] < 0.65)
            | (1 - 2 * (z["qpos"][:, :, 4] ** 2 + z["qpos"][:, :, 5] ** 2) < 0.9)
        ).any(axis=1)
        if bad[:-1].any() or (
            not bad.any() and controls != horizon + (0 if reference_prefix else 60)
        ):
            raise ValueError("body failure stop or uncensored reference tail differs")
        forbidden = bool((f[:, 2] > 1).any())
        impulse = float((f[:, 2] * 0.002).sum())
        reward = -12.0 if bad.any() else -8.0 - 6 * forbidden
        if not bad.any() and post["correct_foot_first"] and post["post_contact_quality_observed"]:
            first = post["first_foot_contact_substep"]
            retreat = post["maximum_pelvis_retreat_after_contact_m"]
            if not isinstance(first, int) or not isinstance(retreat, (int, float)):
                raise ValueError("observed numeric physical contact window required")
            distance = float(np.linalg.norm(q[first + 600, 36:39] - goal))
            reward = (
                2 * (6 - min(12, distance))
                - 6 * forbidden
                - min(3, 2 * impulse)
                + 4 * (not forbidden)
                - 0.5 * retreat
            )
        if (
            post != row["postcontact"]
            or not np.array_equal(bad, z["body_failed"])
            or bool(bad.any()) != row["body_failed"]
            or forbidden != row["forbidden"]
            or reward != row["terminal_reward"]
            or impulse != row["nonfoot_impulse_Ns"]
        ):
            raise ValueError("independent physical primitive outcome/reward differs")
        progress = post["goal_progress_1p2s_m"]
        lateral = post["goal_lateral_1p2s_m"]
        retreat = post["maximum_pelvis_retreat_after_contact_m"]
        quality = bool(
            not bad.any()
            and not forbidden
            and post["correct_foot_first"]
            and post["post_contact_quality_observed"]
            and isinstance(progress, (int, float))
            and progress >= 1
            and isinstance(lateral, (int, float))
            and abs(lateral) <= 0.75
            and not post["other_foot_contact_in_window"]
            and isinstance(retreat, (int, float))
            and retreat <= 0.05
        )
        return dict(
            native_steps=controls * 10, neural_rows=n, quality=quality, body_failed=bool(bad.any())
        )
