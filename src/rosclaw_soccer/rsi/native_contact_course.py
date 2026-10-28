"""Synchronous24 SIM native contact course; no in-step resets or hardware.

Each reset allocates fresh native data ONLY after the previous batch is fully
terminal. Failed worlds stop integrating immediately, remain invalid samples,
and keep their final observations. The caller explicitly consumes the terminal
step and starts the next episode. Critic ABI is deliberately177, not legacy304.
"""

from typing import Any

import mujoco
import numpy as np
import torch
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.identity_kick_credit import IdentityKickCredit
from rosclaw_soccer.rsi.native_course_tracking import NativeCourseTracking
from rosclaw_soccer.rsi.native_physics_batch import NativePhysicsBatch
from rosclaw_soccer.rsi.native_receive_credit import NativeReceiveCredit, planar_ball_root_distance
from rosclaw_soccer.rsi.native_reference_observation import native_reference_observation177
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


def balanced_course_indices(
    bank: dict[str, NDArray[Any]], *, worlds: int, episode: int, lead: int | None
) -> NDArray[np.int64]:
    if (
        type(worlds) is not int
        or worlds not in (8, 16, 32, 64, 256)
        or type(episode) is not int
        or episode < 0
    ):
        raise ValueError("balanced bounded world count and episode required")
    if lead is not None and (type(lead) is not int or lead not in (10, 20, 40, 80)):
        raise ValueError("explicit measured lead10/20/40/80 or all required")
    ids = []
    for i in range(worlds):
        pool = np.flatnonzero(
            (bank["source_id"] == i % 8)
            & ((bank["lead_controls"] == lead) if lead is not None else True)
        )
        if len(pool) != (8 if lead is not None else 32):
            raise ValueError("entire balanced source pool required")
        ids.append(pool[(episode * (worlds // 8) + i // 8) % len(pool)])
    return np.array(ids, dtype=np.int64)


def shin_proximity_risk(gap_m: float) -> float:
    """Bounded measured shin/ball separation penalty, not a contact or safety label."""
    if not np.isfinite(gap_m):
        raise ValueError("finite native collision geometry distance required")
    return float(np.exp(-max(0.0, gap_m) / 0.08))


class NativeContactCourse:
    protocol = "native-contact24-v1"
    actor_observation_dim = critic_observation_dim = 177
    window = 24

    def __init__(
        self,
        *,
        model: Any,
        params: Any,
        bank: dict[str, NDArray[Any]],
        plans: list[dict[str, NDArray[Any]]],
        body_references: list[dict[str, NDArray[Any]]],
        worlds: int,
        lead: int | None = 10,
        threads: int = 8,
        receive_reward: bool = False,
        receive_window: int = 24,
        dense_shin_risk: bool = False,
    ) -> None:
        if (
            type(receive_reward) is not bool
            or type(receive_window) is not int
            or receive_window not in (24, 40, 60)
            or (not receive_reward and receive_window != 24)
            or type(dense_shin_risk) is not bool
            or (dense_shin_risk and (not receive_reward or receive_window != 60))
        ):
            raise ValueError("explicit bounded experimental receive reward mode required")
        self.receive_reward = receive_reward
        self.dense_shin_risk = dense_shin_risk
        self.window = receive_window
        self.receive_failure_penalty = 800 if receive_window == 60 else 600
        self.model, self.params = model, params
        self.bank = {k: v.copy() for k, v in bank.items()}
        self.plans = [{k: v.copy() for k, v in plan.items()} for plan in plans]
        if len(plans) != 8:
            raise ValueError("exact TRAIN8 plans required")
        for k, shape in {
            "qpos": (256, 43),
            "qvel": (256, 41),
            "qacc_warmstart": (256, 41),
            "previous_action": (256, 29),
            "phase_before": (256,),
            "observation": (256, 177),
            "origin": (256, 3),
            "goal_world": (256, 3),
        }.items():
            if self.bank[k].shape != shape or not np.isfinite(self.bank[k]).all():
                raise ValueError("complete finite measured native bank required")
        if not np.array_equal(self.bank["source_id"], self.bank["world"] % 8):
            raise ValueError("bound TRAIN identity required")
        for f, s in zip(bank["reference_frame"], bank["source_id"], strict=True):
            if int(f) + self.window >= len(plans[int(s)]["joint"]):
                raise ValueError("entire24-control measured reference window required")
        self.worlds, self.lead, self.threads = worlds, lead, threads
        balanced_course_indices(self.bank, worlds=worlds, episode=0, lead=lead)
        self.layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        self.tracking = NativeCourseTracking(model, self.layout, body_references)
        self.torso = model.body("robot/torso_link").id
        self.feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
        if dense_shin_risk:
            balls = np.flatnonzero(model.geom_bodyid == self.layout.ball)
            knees = [model.body("robot/" + s + "_knee_link").id for s in ("left", "right")]
            self.shin_geoms = [
                i
                for i in range(model.ngeom)
                if model.geom_bodyid[i] in knees
                and (model.geom_contype[i] or model.geom_conaffinity[i])
            ]
            if len(balls) != 1 or not self.shin_geoms:
                raise ValueError("one ball and actual knee collision geometries required")
            self.ball_geom = int(balls[0])
        self.batch: NativePhysicsBatch | None = None
        self.episode = -1
        self.closed_steps = 0
        self._closed = False
        self.faulted = False
        self.ended = np.ones(worlds, dtype=bool)
        self._observation = np.zeros((worlds, 177), dtype=np.float32)

    @property
    def native_steps(self) -> int:
        return self.closed_steps + (self.batch.native_steps if self.batch is not None else 0)

    @property
    def observation(self) -> NDArray[np.float32]:
        if self.batch is None or self._closed or self.faulted:
            raise ValueError("live native course required")
        return self._observation.copy()

    def reject_worlds(self, mask: NDArray[np.bool_]) -> None:
        """Freeze explicitly rejected SIM worlds before the next physics step.

        The caller must separately record why each world was rejected; this is
        not a physical stop, successful episode, or automatic reset.
        """
        if self.batch is None or self._closed or self.faulted or self.ended.all():
            raise ValueError("active native course required for rejection")
        if (
            not isinstance(mask, np.ndarray)
            or mask.shape != (self.worlds,)
            or mask.dtype != np.bool_
            or not mask.any()
            or np.any(mask & self.ended)
        ):
            raise ValueError("nonempty live-world rejection mask required")
        self.ended |= mask

    def reset(self) -> NDArray[np.float32]:
        if self._closed or self.faulted or (self.batch is not None and not self.ended.all()):
            raise ValueError("cannot reset active/faulted native course")
        if self.batch is not None:
            self.closed_steps += self.batch.native_steps
            self.batch.close()
        self.episode += 1
        self.index = balanced_course_indices(
            self.bank, worlds=self.worlds, episode=self.episode, lead=self.lead
        )
        b, ids = self.bank, self.index
        self.frame = b["reference_frame"][ids].astype(np.int64).copy()
        self.phase = b["phase_before"][ids].astype(np.float32).copy()
        self.last_action = b["previous_action"][ids].copy()
        self.ended = np.zeros(self.worlds, dtype=bool)
        self.elapsed = np.zeros(self.worlds, dtype=np.int64)
        self.control = 0
        self.credit = IdentityKickCredit(self.worlds, "cpu")
        self.receive_credit = NativeReceiveCredit(self.worlds) if self.receive_reward else None
        self.batch = NativePhysicsBatch(self.model, self.worlds, threads=self.threads)
        previous_ctrl = np.zeros((self.worlds, self.model.nu))
        previous_ctrl[:, self.layout.ai] = np.stack(
            [target29(x, self.params) for x in self.last_action]
        )
        try:
            self.batch.restore(
                qpos=b["qpos"][ids],
                qvel=b["qvel"][ids],
                previous_ctrl=previous_ctrl,
                warmstart=b["qacc_warmstart"][ids],
                time=self.frame * 0.02,
            )
            self._observation = np.zeros((self.worlds, 177), dtype=np.float32)
            self._next_phase = self.phase.copy()
            self._prepare_observation(np.arange(self.worlds))
            if not np.array_equal(self._observation, b["observation"][ids]):
                raise ValueError("actual native reset observation differs from qualified bank")
        except Exception:
            self.faulted = True
            raise
        return self.observation

    def _prepare_observation(self, ids: NDArray[np.int64]) -> None:
        assert self.batch is not None
        for i in ids:
            source = int(self.bank["source_id"][self.index[i]])
            obs, phase = native_reference_observation177(
                data=self.batch.data[i],
                layout=self.layout,
                params=self.params,
                plan=self.plans[source],
                frame=int(self.frame[i]),
                origin=self.bank["origin"][self.index[i]],
                goal=self.bank["goal_world"][self.index[i]],
                last_action=self.last_action[i],
                phase=self.phase[i],
                torso=self.torso,
            )
            self._observation[i], self._next_phase[i] = obs, phase

    def _forces(self, data: Any) -> NDArray[np.float64]:
        result = np.zeros(3)
        for ci in range(data.ncon):
            c = data.contact[ci]
            a, b = (int(self.model.geom_bodyid[g]) for g in c.geom)
            other = b if a == self.layout.ball else a if b == self.layout.ball else -1
            if other >= 0 and (self.model.body(other).name or "").startswith("robot/"):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, data, ci, wrench)
                k = self.feet.index(other) if other in self.feet else 2
                result[k] = max(result[k], np.linalg.norm(wrench[:3]))
        return result

    def step(self, action: NDArray[np.float32]) -> dict[str, NDArray[Any]]:
        if self.batch is None or self._closed or self.faulted or self.ended.all():
            raise ValueError("active native course step required")
        if (
            not isinstance(action, np.ndarray)
            or action.shape != (self.worlds, 29)
            or action.dtype != np.float32
            or not np.isfinite(action).all()
        ):
            raise ValueError("finite float32 actual actor29 required")
        active = ~self.ended.copy()
        before_nonfoot = (
            self.receive_credit.nonfoot_seen.copy() if self.receive_credit is not None else None
        )
        ids = np.flatnonzero(active).astype(np.int64)
        before_observation = self.observation
        before_phase, before_frame = self.phase.copy(), self.frame.copy()
        target = np.stack([target29(x, self.params) for x in action])
        ctrl = np.zeros((self.worlds, self.model.nu))
        ctrl[:, self.layout.ai] = target
        q_rows, v_rows, force_rows = [], [], []
        try:
            for _ in range(10):
                self.batch.advance(ctrl, substeps=1, world_ids=ids)
                q_rows.append(np.stack([d.qpos.copy() for d in self.batch.data]))
                v_rows.append(np.stack([d.qvel.copy() for d in self.batch.data]))
                f = np.zeros((self.worlds, 3))
                for i in ids:
                    d = self.batch.data[i]
                    f[i] = self._forces(d)
                    ai = self.layout.ai
                    expected = np.clip(
                        self.model.actuator_gainprm[ai, 0] * target[i]
                        + self.model.actuator_biasprm[ai, 0]
                        + self.model.actuator_biasprm[ai, 1] * d.actuator_length[ai]
                        + self.model.actuator_biasprm[ai, 2] * d.actuator_velocity[ai],
                        self.model.actuator_forcerange[ai, 0],
                        self.model.actuator_forcerange[ai, 1],
                    )
                    if not np.allclose(d.actuator_force[ai], expected, atol=1e-8, rtol=0):
                        raise ValueError("actual native PD law differs")
                force_rows.append(f)
            qs, vs, forces = (np.stack(x, axis=1) for x in (q_rows, v_rows, force_rows))
            tracking = np.zeros((self.worlds, 5))
            guards = np.zeros((self.worlds, 4), dtype=bool)
            proximity, limits = np.zeros(self.worlds), np.zeros(self.worlds)
            shin_risk = np.zeros(self.worlds)
            for i in ids:
                d = self.batch.data[i]
                mujoco.mj_forward(self.model, d)
                if self.dense_shin_risk:
                    assert before_nonfoot is not None
                if self.dense_shin_risk and before_nonfoot is not None and not before_nonfoot[i]:
                    gap = min(
                        float(
                            mujoco.mj_geomDistance(self.model, d, self.ball_geom, geom, 1.0, None)
                        )
                        for geom in self.shin_geoms
                    )
                    shin_risk[i] = shin_proximity_risk(gap)
                index = self.index[i]
                tracking[i], guards[i] = self.tracking.measure(
                    d,
                    source=int(self.bank["source_id"][index]),
                    frame=int(self.frame[i]),
                    origin=self.bank["origin"][index],
                )
                foot = self.feet[int(self.bank["expected_foot"][index])]
                proximity[i] = np.exp(
                    -np.square(d.xpos[foot] - d.xpos[self.layout.ball]).sum() / 0.3**2
                )
                ranges = self.model.jnt_range[self.layout.joints]
                center = ranges.mean(axis=1)
                half = (ranges[:, 1] - ranges[:, 0]) * 0.9 / 2
                position = d.qpos[self.layout.qi]
                limits[i] = (
                    np.maximum(center - half - position, 0)
                    + np.maximum(position - center - half, 0)
                ).sum()
            if self.receive_reward:
                assert self.receive_credit is not None
                credits = self.receive_credit.step(
                    step=self.control,
                    forces=forces,
                    active=active,
                    ball_root_planar_distance_m=planar_ball_root_distance(
                        qs[:, -1, self.layout.bq : self.layout.bq + 3],
                        qs[:, -1, self.layout.rq : self.layout.rq + 3],
                    ),
                    relative_speed_mps=np.linalg.norm(
                        vs[:, -1, self.layout.bv : self.layout.bv + 3]
                        - vs[:, -1, self.layout.rv : self.layout.rv + 3],
                        axis=1,
                    ),
                    healthy=~guards.any(axis=1),
                )
            else:
                scores = self.credit.step(
                    step=self.control,
                    forces=torch.from_numpy(forces[:, :, :2]),
                    expected=torch.from_numpy(self.bank["expected_foot"][self.index]),
                    ball_velocity=torch.from_numpy(vs[:, -1, self.layout.bv : self.layout.bv + 3]),
                    goal_delta=torch.from_numpy(
                        self.bank["goal_world"][self.index]
                        - qs[:, -1, self.layout.bq : self.layout.bq + 3]
                    ),
                    forbidden_forces=torch.from_numpy(forces[:, :, 2]),
                )
                credits = {k: v.numpy().copy() for k, v in scores.items()}
            for value in credits.values():
                value[~active] = 0
            terminated = guards.any(axis=1) & active
            self.elapsed[active] += 1
            reference_end = np.array(
                [
                    self.frame[i]
                    >= len(self.plans[int(self.bank["source_id"][self.index[i]])]["joint"]) - 1
                    for i in range(self.worlds)
                ]
            )
            truncated = active & ~terminated & ((self.elapsed >= self.window) | reference_end)
            action_rate = np.square(action - self.last_action).sum(axis=1).clip(max=100)
            if self.receive_reward:
                reward = 0.02 * (
                    tracking.sum(axis=1)
                    + proximity
                    - 0.1 * action_rate
                    - 10 * limits
                    + 50 * credits["first_clean_foot"]
                    - 300 * credits["first_nonfoot"]
                    + 200 * credits["sustained_capture"]
                    - self.receive_failure_penalty * terminated
                )
                if self.dense_shin_risk:
                    reward -= 0.1 * shin_risk
            else:
                reward = 0.02 * (
                    tracking.sum(axis=1)
                    + proximity
                    - 0.1 * action_rate
                    - 10 * limits
                    + 50 * credits["contact"]
                    - 10 * credits["wrong_contact"]
                    + 30 * credits["alignment"]
                    + 10 * credits["speed"]
                    - 300 * credits["forbidden_contact"]
                    - 400 * terminated
                )
            reward[~active] = 0
            if not np.isfinite(reward).all():
                raise ValueError("finite native rewards required")
            self.phase[active] = self._next_phase[active]
            self.last_action[active] = action[active]
            self.frame[active] += 1
            for i in ids:
                self.frame[i] = min(
                    self.frame[i],
                    len(self.plans[int(self.bank["source_id"][self.index[i]])]["joint"]) - 1,
                )
            self._prepare_observation(ids)
            self.ended |= terminated | truncated
            self.control += 1
            return dict(
                observation=self.observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                valid=active,
                ended=self.ended.copy(),
                observation_before=before_observation,
                phase_before=before_phase,
                frame_before=before_frame,
                target=target,
                qpos=qs,
                qvel=vs,
                forces=forces,
                tracking=tracking,
                guards=guards,
                proximity=proximity,
                limits=limits,
                action_rate=action_rate,
                **({"shin_risk": shin_risk} if self.dense_shin_risk else {}),
                **{"credit_" + k: v for k, v in credits.items()},
            )
        except Exception:
            self.faulted = True
            raise

    def close(self) -> None:
        if self.batch is not None:
            self.batch.close()
        self._closed = True
