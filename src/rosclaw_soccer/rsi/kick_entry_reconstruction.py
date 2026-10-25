"""Reconstruct a persisted SIM entry observation; never advance physics or actuate.

The source engine stores post-integration qpos while torso/ball FK is cached
from the pre-integration state. Both clocks are retained, not conflated.
"""

import contextlib
import io
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.kick_measured_history import (
    KickMeasuredSample,
    install_measured_kick_history,
)
from rosclaw_soccer.providers.g1.kick_warmstart import rebase_kick_reference
from rosclaw_soccer.rsi.first_touch_outcome import remaining_task_time


def sample_from_numeric(values: Any) -> KickMeasuredSample:
    x = np.asarray(values)
    if (
        x.shape != (103,)
        or x.dtype != np.float64
        or not np.isfinite(x).all()
        or (any(float(x[k]) != int(x[k]) for k in (0, 2)))
    ):
        raise ValueError("lossless finite103 measured-history row with integer frames required")
    return KickMeasuredSample(
        frame=int(x[0]),
        time_sec=float(x[1]),
        previous_target_frame=int(x[2]),
        joint_position=tuple(map(float, x[3:32])),
        joint_velocity=tuple(map(float, x[32:61])),
        angular_velocity=tuple(map(float, x[61:64])),
        pelvis_position=tuple(map(float, x[64:67])),
        pelvis_quaternion=tuple(map(float, x[67:71])),
        ball_position=tuple(map(float, x[71:74])),
        previous_pd_target=tuple(map(float, x[74:103])),
    )


class KickEntryReconstructor:
    """One offline policy/resettable software workspace, not a live skill owner."""

    def __init__(self, model: Any, layout: Any, policy: Any, hg_mapping: Any) -> None:
        import mujoco

        mapping = np.asarray(hg_mapping)
        if (
            mapping.shape != (29,)
            or mapping.dtype.kind not in "iu"
            or sorted(mapping.tolist()) != list(range(29))
            or (policy.runtime_mode != "sim" or policy.use_body_frame_ball)
        ):
            raise ValueError("qualified HG29 permutation and SIM policy required")
        self.model, self.layout, self.policy, self.mapping = model, layout, policy, mapping
        self.data = mujoco.MjData(model)
        self.torso = int(model.body(layout.robot_prefix + "torso_link").id)

    def reconstruct(self, row: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
        import mujoco

        frame, phase = row["entry_control"], row["admitted_reference_phase"]
        if (
            type(frame) is not int
            or not 5 <= frame < min(300, row["controls"])
            or (type(phase) is not int or not 0 <= phase < self.policy.motion_total_steps)
        ):
            raise ValueError("actual admitted entry and available first neural target required")
        p, d, lo = self.policy, self.data, self.layout
        samples = tuple(sample_from_numeric(s) for s in row["entry_state"]["measured_history"])
        q, v = (np.asarray(row["entry_state"][k], dtype=np.float64) for k in ("qpos", "qvel"))
        if q.shape != (43,) or v.shape != (41,) or len(samples) != 5:
            raise ValueError("actual canonical entry state and allfive measured samples required")
        n = frame * 10 - 1
        native_q, native_v = arrays["qpos"][n], arrays["qvel"][n]
        expected_q = np.r_[
            native_q[lo.rq : lo.rq + 7], native_q[lo.qi], native_q[lo.bq : lo.bq + 7]
        ]
        expected_v = np.r_[
            native_v[lo.rv : lo.rv + 6], native_v[lo.vi], native_v[lo.bv : lo.bv + 6]
        ]
        if not np.array_equal(q, expected_q) or not np.array_equal(v, expected_v):
            raise ValueError("entry metadata differs from actual persisted native state")
        # Inspection only. No mj_step/mj_forward/torque write/physical clock.
        d.qpos[:] = arrays["qpos"][n - 1]
        mujoco.mj_kinematics(self.model, d)
        if not np.array_equal(d.xpos[lo.ball], samples[-1].ball_position):
            raise ValueError("pre-integration ball FK differs from measured observation")
        s = p.state_cmd
        s.q, s.dq = q[7:36].copy(), v[6:35].copy()
        s.tau_est = arrays["torque"][n, lo.ai].copy()
        s.root_lin_vel_b, s.root_ang_vel_b = v[:3].copy(), v[3:6].copy()
        s.torso_pos_w, s.torso_quat_w = d.xpos[self.torso].copy(), d.xquat[self.torso].copy()
        s.pelvis_pos_w, s.pelvis_quat_w = q[:3].copy(), q[3:7].copy()
        s.ball_pos_w, s.ball_vel_w = np.array(samples[-1].ball_position), v[35:38].copy()
        s.ball_valid, s.target_y_bias, s.ball_y_bias = True, 0.0, 0.0
        p.target_pos_w = np.array([8.0, 0.0, 0.11], np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            p.enter()
        install_measured_kick_history(p, samples, frame=frame, time_sec=samples[-1].time_sec)
        rebase_kick_reference(p, entry_frame=phase)
        p.time_step = p.WARMUP_STEPS + phase
        obs = p._build_obs()
        if obs.shape != (547,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError("finite actual547 observation required")
        raw_action = p.ort_session.run(["actions"], {"obs": obs[None]})[0][0]
        actions = np.clip(raw_action, p.action_clip_lo_il, p.action_clip_hi_il)
        target = p.default_q_mj + p.action_scale_mj * actions[self.mapping]
        if not np.array_equal(target.astype(np.float64), arrays["joint_targets"][frame]):
            delta = float(np.max(np.abs(target - arrays["joint_targets"][frame])))
            raise ValueError(f"reconstructed first HG29 target differs: max_error={delta}")
        if float(d.time) != 0.0:
            raise ValueError("inspection unexpectedly advanced physics")
        self.last_raw_action = raw_action.copy()
        self.last_clipped_action = actions.copy()
        return np.asarray(
            np.r_[
                obs,
                np.float32(phase / (p.motion_total_steps - 1)),
                np.float32(remaining_task_time(frame)),
            ],
            dtype=np.float32,
        )

    def reconstruct_trace(
        self, row: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Recover the entire original policy history, checking EVERY PD target.

        The first entry initializes from five real measured frames. Thereafter
        the frozen policy carries its own exact original clipped action/history;
        observed body states always come from the saved native trajectory.
        This is offline inference replay, not new physics or an RL rollout.
        """
        import mujoco

        first = self.reconstruct(row, arrays)
        entry = row["entry_control"]
        observations = [first[:547].copy()]
        raw = [self.last_raw_action.copy()]
        clipped = [self.last_clipped_action.copy()]
        p, d, lo = self.policy, self.data, self.layout
        for frame in range(entry + 1, row["controls"]):
            p.last_action_il = self.last_clipped_action.copy()
            p.time_step += 1
            n = frame * 10 - 1
            native_q, native_v = arrays["qpos"][n], arrays["qvel"][n]
            q = np.r_[native_q[lo.rq : lo.rq + 7], native_q[lo.qi], native_q[lo.bq : lo.bq + 7]]
            v = np.r_[native_v[lo.rv : lo.rv + 6], native_v[lo.vi], native_v[lo.bv : lo.bv + 6]]
            d.qpos[:] = arrays["qpos"][n - 1]
            mujoco.mj_kinematics(self.model, d)
            s = p.state_cmd
            s.q, s.dq = q[7:36].copy(), v[6:35].copy()
            s.tau_est = arrays["torque"][n, lo.ai].copy()
            s.root_lin_vel_b, s.root_ang_vel_b = v[:3].copy(), v[3:6].copy()
            s.torso_pos_w, s.torso_quat_w = d.xpos[self.torso].copy(), d.xquat[self.torso].copy()
            s.pelvis_pos_w, s.pelvis_quat_w = q[:3].copy(), q[3:7].copy()
            s.ball_pos_w, s.ball_vel_w = d.xpos[lo.ball].copy(), v[35:38].copy()
            obs = p._build_obs()
            if obs.shape != (547,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
                raise ValueError("finite actual547 full-history observation required")
            action = p.ort_session.run(["actions"], {"obs": obs[None]})[0][0]
            applied = np.clip(action, p.action_clip_lo_il, p.action_clip_hi_il)
            target = p.default_q_mj + p.action_scale_mj * applied[self.mapping]
            if not np.array_equal(target.astype(np.float64), arrays["joint_targets"][frame]):
                raise ValueError(f"full policy history failed exact HG29 target at frame{frame}")
            self.last_raw_action, self.last_clipped_action = action.copy(), applied.copy()
            observations.append(obs.copy())
            raw.append(action.copy())
            clipped.append(applied.copy())
        if float(d.time) != 0.0:
            raise ValueError("full-history inspection unexpectedly advanced physics")
        return dict(
            observations=np.stack(observations),
            original_raw_actions=np.stack(raw),
            original_clipped_actions=np.stack(clipped),
            control_frames=np.arange(entry, row["controls"], dtype=np.int64),
            original_pd_targets=arrays["joint_targets"][entry:].copy(),
        )
