"""Reuse the native SIM controller for own-policy causal head PPO experiments.

Numerical proposals are not activation: the campaign must physically check old
skills, then either commit or restore BOTH motor weights and Adam moments.
"""

import copy
import math
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.actor_trust_region import bounded_adam_proposal
from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp
from rosclaw_soccer.rsi.kick_event_reward import kick_event_rewards
from rosclaw_soccer.rsi.kick_prior_actor import KickPriorActor
from rosclaw_soccer.rsi.kick_prior_own_policy import OwnKickPlayer, OwnKickSession
from rosclaw_soccer.rsi.kick_prior_ppo import (
    KickValueHead,
    freeze_kick_features,
    own_ppo_objective,
    value_features,
)
from rosclaw_soccer.rsi.native_kick_reward import finite_returns


def copy_actor(actor: KickPriorActor) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in actor.state_dict().items()}


def snapshot_factory(original_factory: Any, state: dict[str, torch.Tensor]) -> Any:
    def create(*args: Any, **kwargs: Any) -> Any:
        policy = original_factory(*args, **kwargs)
        policy.ort_session.actor.load_state_dict(state, strict=True)
        return policy

    return create


def collect_episode(
    actor: KickPriorActor,
    original_factory: Any,
    *,
    seed: int,
    state_type: Any,
    output_type: Any,
    mapping: Any,
    **episode_args: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any] | None, torch.Tensor | None]:
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode

    snapshot = copy_actor(actor)
    current = snapshot_factory(original_factory, snapshot)
    sessions: list[OwnKickSession] = []

    def create(*args: Any, **kwargs: Any) -> Any:
        policy = current(*args, **kwargs)
        session = OwnKickSession(policy.ort_session.actor, seed=seed)
        policy.ort_session = session
        sessions.append(session)
        return policy

    row, arrays = evaluate_episode(
        policy_types=(state_type, output_type, create, mapping),
        player_type=OwnKickPlayer,
        **episode_args,
    )
    if any(not torch.equal(v, actor.state_dict()[k]) for k, v in snapshot.items()):
        raise ValueError("collector modified the current training actor")
    n = row["controls"]
    rq = row["robot_qpos_address"]
    events = kick_event_rewards(
        ball_after=arrays["ball_position_after_step_m"],
        pelvis_native=arrays["qpos"][:, rq : rq + 7].reshape(n, 10, 7),
        force_native=arrays["ball_forces"][:, 0].reshape(n, 10, 3),
    )
    arrays["causal_rewards"] = events["rewards"]
    row["causal_events"] = {
        k: v for k, v in events.items() if k not in ("components", "rewards", "component_names")
    }
    row["causal_total_return"] = float(events["rewards"].sum(dtype=np.float64))
    if row["entry_control"] is None:
        if sessions:
            raise ValueError("unadmitted episode unexpectedly created a motor actor")
        row["own_motor_samples"] = 0
        return row, arrays, None, None
    if len(sessions) != 1 or sessions[0].faulted or sessions[0].pending is not None:
        raise ValueError("one fault-free completely committed own-policy session required")
    session = sessions[0]
    records = session.samples
    frames = np.array([r["frame"] for r in records], np.int64)
    if not np.array_equal(frames, np.arange(row["entry_control"], n)) or session.calls != len(
        frames
    ):
        raise ValueError("every own neural call must match its actual control boundary")
    obs = torch.from_numpy(np.stack([r["observation"] for r in records]))
    old = torch.from_numpy(np.stack([r["mean"] for r in records]))
    actions = torch.from_numpy(np.stack([r["raw_action"] for r in records]))
    logp = torch.tensor([r["raw_logp"] for r in records], dtype=torch.float32)
    pd = np.stack([r["executed_pd_target"] for r in records])
    with torch.no_grad():
        if float((actor(obs) - old).abs().max()) > 1e-5 or (
            float((conditional_logp(old, actions, session.log_std, rho=0.9) - logp).abs().max())
            > 0.002
        ):
            raise ValueError("actual current-policy means or raw probability law failed replay")
    if not np.array_equal(pd, arrays["joint_targets"][frames]):
        raise ValueError("own raw draw was not committed to the actual encoded PD target")
    arrays.update(
        own_observations=obs.numpy(),
        own_old_means=old.numpy(),
        own_raw_actions=actions.numpy(),
        own_raw_logp=logp.numpy(),
        own_control_frames=frames,
        own_executed_pd_target=pd,
    )
    row["own_motor_samples"] = len(frames)
    episode = dict(
        observations=obs,
        old_means=old,
        raw_actions=actions,
        old_logp=logp,
        frames=torch.from_numpy(frames),
        returns=torch.from_numpy(
            finite_returns(events["rewards"], gamma=1.0)[frames].copy()  # type: ignore[no-untyped-call]
        ),
        actual_pd_commitment_verified=True,
        external_teacher=False,
    )
    return row, arrays, episode, session.generator.get_state()


class CausalKickLearner:
    def __init__(self, actor: KickPriorActor, anchors: torch.Tensor) -> None:
        self.actor = actor
        self.plastic = freeze_kick_features(actor)
        self.critic = KickValueHead()
        self.actor_optimizer = torch.optim.Adam(self.plastic, lr=1e-4)
        self.value_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.anchors = anchors
        with torch.no_grad():
            self.anchor_means = actor(anchors).detach()
        self.initial_features = {
            k: v.clone() for k, v in copy_actor(actor).items() if not k.startswith("actor.6.")
        }
        self.log_std = torch.full((29,), math.log(0.015), dtype=torch.float32)
        self.value_steps = 0
        self.actor_proposals = 0
        self.motor_commits = 0
        self.pending: dict[str, Any] | None = None

    def propose(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        if self.pending is not None or not episodes:
            raise ValueError(
                "physical decision for prior proposal and new own trajectories required"
            )
        self.actor_optimizer.zero_grad(set_to_none=True)
        for episode in episodes:
            if (
                episode.get("actual_pd_commitment_verified") is not True
                or episode.get("external_teacher") is not False
            ):
                raise ValueError(
                    "actual unassisted own experience required before any value update"
                )
            target = episode["returns"]
            if (
                target.shape != (len(episode["observations"]),)
                or not torch.isfinite(target).all()
                or target.requires_grad
            ):
                raise ValueError("finite detached causal returns at actual own frames required")
            with torch.no_grad():
                if (
                    float((self.actor(episode["observations"]) - episode["old_means"]).abs().max())
                    > 1e-5
                ):
                    raise ValueError("new batch must be generated by the current motor policy")
            episode["features"] = value_features(
                self.actor, episode["observations"], episode["frames"]
            )
            with torch.no_grad():
                episode["advantages"] = (
                    episode["returns"] - self.critic(episode["features"])
                ).detach()
        mean = torch.stack([e["advantages"].mean() for e in episodes]).mean()
        variance = torch.stack([(e["advantages"] - mean).square().mean() for e in episodes]).mean()
        for episode in episodes:
            episode["advantages"] = (
                (episode["advantages"] - mean) / (variance.sqrt() + 1e-8)
            ).detach()
        value_losses = []
        for _ in range(4):
            self.value_optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(
                [(self.critic(e["features"]) - e["returns"]).square().mean() for e in episodes]
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite causal value loss")
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, error_if_nonfinite=True)
            self.value_optimizer.step()
            self.value_steps += 1
            value_losses.append(float(loss.detach()))
        if any(p.grad is not None for p in self.actor.parameters()):
            raise ValueError("critic gradients leaked into the motor actor")
        old = dict(
            actor=copy_actor(self.actor), optimizer=copy.deepcopy(self.actor_optimizer.state_dict())
        )

        def closure() -> tuple[torch.Tensor, dict[str, Any]]:
            loss, stats = own_ppo_objective(
                self.actor, episodes, self.anchors, self.anchor_means, self.log_std
            )
            # NEW protocol: local trust relative to CURRENT behavior; the
            # original prior is a soft full-history regularizer, not a false
            # closed-loop safety certificate. Physical retention decides.
            stats["mean_kl"] = stats["own_conditional_kl"]
            return loss, stats

        decision = bounded_adam_proposal(self.plastic, self.actor_optimizer, closure)
        self.actor_proposals += 1
        if any(
            not torch.equal(v, self.actor.state_dict()[k]) for k, v in self.initial_features.items()
        ):
            raise ValueError("frozen motor features changed")
        if decision["accepted"]:
            self.pending = old
        return dict(
            actor_decision=decision,
            value_losses=value_losses,
            total_value_updates=self.value_steps,
            actor_proposals=self.actor_proposals,
        )

    def physical_decision(self, passed: bool) -> None:
        if type(passed) is not bool or self.pending is None:
            raise ValueError(
                "one explicit physical commit/rollback for a pending numerical proposal required"
            )
        if passed:
            self.motor_commits += 1
        else:
            self.actor.load_state_dict(self.pending["actor"], strict=True)
            self.actor_optimizer.load_state_dict(self.pending["optimizer"])
        self.pending = None
