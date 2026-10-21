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
            if "velocity_world_initial" in case:
                raise ValueError("world velocity cannot silently mix with local-frame case")
            ball = self.quat_apply(quat, np.r_[case["ball_local"], 0.0])
            ball[2] = 0.11
            velocity = self.quat_apply(quat, np.r_[case["velocity_local"], 0.0])[:2]
            goal = self.quat_apply(quat, np.array([6.0, 0.0, 0.0]))
            goal[2] = 0.11
        else:
            ball, goal = np.array(case["ball_world_initial"]), np.array(case["goal_world"])
            if "velocity_world_initial" in case:
                velocity = np.asarray(case["velocity_world_initial"], dtype=np.float64)
                if (
                    velocity.shape != (2,)
                    or not np.isfinite(velocity).all()
                    or np.linalg.norm(velocity) > 0.6
                ):
                    raise ValueError("finite explicitly bounded world velocity required")
            else:
                local_velocity = (
                    [0, 0] if case["case"] % 2 == 0 else [-0.2, 0.1 if "left" in name else -0.1]
                )
                velocity = self.quat_apply(quat, np.r_[local_velocity, 0.0])[:2]
        ep.reset(ball=ball, velocity=velocity, goal=goal)
        return ep

    def collect(
        self,
        case: dict[str, Any],
        actor: Any,
        *,
        transform: Any = None,
        causal_transform: Any = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """One deterministic actor/composite-teacher rollout from frame zero.

        A transformed action is an explicit composite teacher, NOT the actor's
        stochastic sample. No policy likelihood is fabricated for these data.
        """
        ep = self.make(case)
        return self.collect_current(
            ep, actor, transform=transform, causal_transform=causal_transform
        )

    def collect_current(
        self,
        ep: PaidNativeEpisode,
        actor: Any,
        *,
        transform: Any = None,
        causal_transform: Any = None,
        transition_controls: int = 0,
        initial_state: tuple[Any, Any] | None = None,
        stop_at_reference: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Run an already admitted live-body skill; NEVER reset simulator state.

        Neural h/c starts at zero unless the caller explicitly carries its own
        per-player state. Previous executed action is carried by the episode. Crossfade
        is an explicit composite transition, not a pure-neural policy claim.
        """
        import torch

        if (
            ep.done
            or ep.frame != 0
            or transition_controls not in (0, 12)
            or type(stop_at_reference) is not bool
        ):
            raise ValueError("fresh explicit skill boundary and declared transition required")
        if (
            transform is not None
            and causal_transform is not None
            or transition_controls
            and (transform is not None or causal_transform is not None)
        ):
            raise ValueError("transition and teacher transformations must be separate experiments")
        if causal_transform is not None and not callable(causal_transform):
            raise ValueError("explicit callable causal research transform required")
        previous = ep.last_action.copy()
        if initial_state is not None and (
            len(initial_state) != 2
            or any(
                not isinstance(s, torch.Tensor)
                or s.shape != (2, 1, 128)
                or s.dtype != torch.float32
                or s.device.type != "cpu"
                or not torch.isfinite(s).all()
                for s in initial_state
            )
        ):
            raise ValueError("explicit finite CPU float32 per-player h/c required")
        state = (
            (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
            if initial_state is None
            else tuple(s.detach().clone() for s in initial_state)
        )
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
                if transition_controls and ep.frame < transition_controls:
                    u = (ep.frame + 1) / transition_controls
                    weight = 10 * u**3 - 15 * u**4 + 6 * u**5
                    action = ((1 - weight) * previous + weight * nominal).astype(np.float32)
                if transform is not None:
                    action, plan = transform(ep, nominal)
                    plans.append(plan)
                if causal_transform is not None:
                    from rosclaw_soccer.rsi.paid_causal_context import CausalMotorContext

                    context = CausalMotorContext(
                        episode_key=id(ep),
                        frame=ep.frame,
                        observation=obs,
                        h_before=before[0],
                        c_before=before[1],
                        h_after=state[0].numpy(),
                        c_after=state[1].numpy(),
                    )
                    # A planner may propose an action but must not rewrite the
                    # actual SIM state or the actor proposal being recorded.
                    physical = {
                        k: getattr(ep.env.data, k).copy()
                        for k in (
                            "qpos",
                            "qvel",
                            "act",
                            "ctrl",
                            "qacc_warmstart",
                            "mocap_pos",
                            "mocap_quat",
                        )
                    }
                    clock = float(ep.env.data.time)
                    action, plan = causal_transform(ep, nominal.copy(), context)
                    context.assert_current(ep)
                    if float(ep.env.data.time) != clock or any(
                        not np.array_equal(getattr(ep.env.data, k), v) for k, v in physical.items()
                    ):
                        raise ValueError("causal teacher mutated native SIM state")
                    if (
                        not isinstance(action, np.ndarray)
                        or action.shape != (29,)
                        or action.dtype != np.float32
                        or not np.isfinite(action).all()
                    ):
                        raise ValueError("finite float32 policy-order29 teacher proposal required")
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
            if stop_at_reference and ep.frame == ep.clip.length and not ep.done:
                ep.admit_reference_recovery()
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
            intended_foot=0 if "left" in ep.clip.name else 1,
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
            source=ep.clip.path.name,
            native_steps=len(controls) * 10,
            neural_rows=len(neural),
            body_failed=failed,
            forbidden=nonfoot,
            nonfoot_impulse_Ns=impulse,
            terminal_reward=reward,
            postcontact=post,
            composite_teacher=transform is not None
            or causal_transform is not None
            or transition_controls > 0,
            policy_likelihood_provided=False,
        )
        if causal_transform is not None:
            summary["explicit_own_causal_teacher_context"] = True
        if stop_at_reference:
            summary.update(
                reference_prefix_only=True,
                official_exam_result=False,
                held_tail_controls_omitted=60,
                recovery_admitted=ep.termination_reason == "RECOVERY_ADMITTED",
            )
        return raw, summary
