"""Shared native research collection for physics teachers and later learning.

The immutable model is shared; EVERY episode owns a new mjData and its own
actor h/c. External reference modules are injected only after caller pinning.
No ROS/robot transport, physical-state correction or automatic model promotion.
"""

import copy
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_contact_window import native_contact_window
from rosclaw_soccer.rsi.paid_native_episode import PaidNativeEpisode


class PaidResearchBench:
    def __init__(
        self,
        *,
        xml: Path,
        contract: Any,
        clips: dict[str, Any],
        scene_factory: Any,
        build_observation: Any,
        quat_apply: Any,
        effort: np.ndarray,
        torso_index: int,
    ) -> None:
        self.template = scene_factory(xml)
        self.contract, self.clips = contract, clips
        self.build_observation, self.quat_apply = build_observation, quat_apply
        self.effort, self.torso_index = effort.copy(), torso_index
        self.counters: list[Any] = []
        self.neural_rows = 0
        self.lock = Lock()

    @property
    def native_steps(self) -> int:
        with self.lock:
            counters = self.counters.copy()
        return sum(c.native_steps for c in counters)

    def make(self, case: dict[str, Any]) -> PaidNativeEpisode:
        import mujoco

        name = case["source"]
        # Copy only the lightweight view/address tables; model remains const.
        # MuJoCo's supported data-parallel design is one model + per-episode
        # mjData. Do not change ANY model parameters after this factory exists.
        env = copy.copy(self.template)
        env.data = mujoco.MjData(env.model)
        ep = PaidNativeEpisode(
            env,
            self.clips[name],
            self.contract.parameters,
            self.contract.hg_from_policy,
            self.effort,
            self.build_observation,
            self.torso_index,
        )
        with self.lock:
            self.counters.append(ep.work)
        quat = self.clips[name].body_quat_w[0, 0].astype(np.float64)
        if "ball_local" in case:
            ball = self.quat_apply(quat, np.r_[case["ball_local"], 0.0])
            ball[2] = 0.11
            velocity = self.quat_apply(quat, np.r_[case["velocity_local"], 0.0])[:2]
            goal = self.quat_apply(quat, np.array([6.0, 0.0, 0.0]))
            goal[2] = 0.11
        else:
            ball, goal = np.array(case["ball_world_initial"]), np.array(case["goal_world"])
            local_velocity = (
                [0, 0] if case["case"] % 2 == 0 else [-0.2, 0.1 if "left" in name else -0.1]
            )
            velocity = self.quat_apply(quat, np.r_[local_velocity, 0.0])[:2]
        ep.reset(ball=ball, velocity=velocity, goal=goal)
        return ep

    def collect(
        self, case: dict[str, Any], actor: Any, *, transform: Any = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """One deterministic actor/composite-teacher rollout from frame zero.

        A transformed action is an explicit composite teacher, NOT the actor's
        stochastic sample. No policy likelihood is fabricated for these data.
        """
        import torch

        ep = self.make(case)
        state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
        controls, neural, plans = [], [], []
        while not ep.done:
            action = None
            if ep.frame < ep.clip.length:
                obs = ep.observation()
                before = [s.numpy().copy() for s in state]
                with torch.no_grad():
                    mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
                with self.lock:
                    self.neural_rows += 1
                nominal = mean[0, 0].numpy().copy()
                action = nominal.copy()
                if transform is not None:
                    action, plan = transform(ep, nominal)
                    plans.append(plan)
                neural.append(
                    dict(
                        obs=obs,
                        nominal_action=nominal,
                        action=action.copy(),
                        h_before=before[0],
                        c_before=before[1],
                        h_after=state[0].numpy().copy(),
                        c_after=state[1].numpy().copy(),
                    )
                )
            controls.append(ep.step(action))
        raw = {k: np.array([row[k] for row in controls]) for k in controls[0]}
        raw.update({"NN_" + k: np.array([row[k] for row in neural]) for k in neural[0]})
        raw.update({"initial_" + k: v for k, v in ep.initial.items()})
        if plans:
            raw.update({"PLAN_" + k: np.array([row[k] for row in plans]) for k in plans[0]})
        q, v, f = [
            raw[k].reshape(-1, width) for k, width in (("qpos", 43), ("qvel", 41), ("forces", 3))
        ]
        post = native_contact_window(
            force=f[:, :2],
            canonical_qpos=q,
            canonical_qvel=v,
            goal=ep.goal,
            intended_foot=0 if "left" in case["source"] else 1,
        )
        nonfoot = bool((f[:, 2] > 1).any())
        failed = bool(raw["body_failed"].any())
        impulse = float((f[:, 2] * 0.002).sum())
        if failed:
            reward = -12.0
        elif not (post["post_contact_quality_observed"] and post["correct_foot_first"]):
            reward = -8.0 - 6 * nonfoot
        else:
            first = post["first_foot_contact_substep"]
            retreat = post["maximum_pelvis_retreat_after_contact_m"]
            if not isinstance(first, int) or not isinstance(retreat, (int, float)):
                raise ValueError("observed numeric contact window required")
            index = first + 600
            distance = float(np.linalg.norm(q[index, 36:39] - ep.goal))
            reward = (
                2 * (6 - min(12, distance))
                - 6 * nonfoot
                - min(3, 2 * impulse)
                + 4 * (not nonfoot)
                - 0.5 * retreat
            )
        summary = dict(
            source=case["source"],
            native_steps=len(controls) * 10,
            neural_rows=len(neural),
            body_failed=failed,
            forbidden=nonfoot,
            nonfoot_impulse_Ns=impulse,
            terminal_reward=reward,
            postcontact=post,
            composite_teacher=transform is not None,
            policy_likelihood_provided=False,
        )
        return raw, summary
