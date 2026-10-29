"""Transactional SIM return-player adaptation on its own observed clock."""

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.passer_velocity_recovery import PasserVelocityRecovery
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
from rosclaw_soccer.rsi.receiver_exploration_stream import ReceiverExplorationStream
from rosclaw_soccer.rsi.return_player_adaptive_proposal import return_player_adaptive_proposal


class AdaptivePasserVelocityRecovery(PasserVelocityRecovery):
    """No reset or transport: preserve actual actions through a 20-control blend."""

    simulation_only = True

    def __init__(self, world, prefix, teammate_prefix, stream, actor, *, sampling_seed=None):
        if (
            teammate_prefix == prefix
            or teammate_prefix not in world.layouts
            or type(actor) is not PrecontactReceivingMotor
            or actor.training
            or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
        ):
            raise ValueError("distinct simulated teammate and frozen explicit actor required")
        super().__init__(world, prefix, stream)
        self.teammate_prefix, self.actor = teammate_prefix, actor
        self.sampler = (
            None
            if sampling_seed is None
            else ReceiverExplorationStream(
                np.asarray(stream.params.action_scale, dtype=np.float32), seed=sampling_seed
            )
        )
        self.sampled_controls = 0
        self.activation_control = None
        self.foot_contact_seen = self.other_body_contact_seen = False
        self.pending_adaptive = None
        self.last_adaptive_trace = None
        self.adaptive_calls = 0
        self.faulted = False

    def activate(self, *, motion_complete, passer_foot_seen):
        if self.faulted:
            raise ValueError("adaptive SIM player fault latched")
        super().activate(motion_complete=motion_complete, passer_foot_seen=passer_foot_seen)
        self.activation_control = self.world.control_steps

    def observe(self, command_body):
        if self.faulted:
            raise ValueError("adaptive SIM player fault latched")
        try:
            self.last_adaptive_trace = None
            original = super().observe(command_body)
            self.pending_adaptive = None
            if not self.active or self.world.control_steps - self.activation_control < 20:
                return original
            q, v = self.world.layouts[self.prefix].canonical(self.world.data)
            teammate_q, _ = self.world.layouts[self.teammate_prefix].canonical(self.world.data)
            proposal = return_player_adaptive_proposal(
                self.actor,
                self.stream.params,
                self.stream.last_observation,
                self.stream.last_action,
                q,
                v,
                np.asarray(teammate_q[:3], dtype=np.float64).copy(),
                foot_contact_seen=self.foot_contact_seen,
                other_body_contact_seen=self.other_body_contact_seen,
            )
            if self.sampler is not None and self.sampled_controls < 300:
                mean = proposal["action"].copy()
                action, sampling_trace = self.sampler.propose(
                    mean, proposal["reference_action"], control=self.sampled_controls
                )
                proposal.update(
                    sampling_mean=mean,
                    sampling_trace=sampling_trace,
                    action=action,
                    target=np.asarray(target29(action, self.stream.params)).copy(),
                )
            self.pending_adaptive = proposal
            self.adaptive_calls += 1
            self.pending = np.asarray(proposal["target"], dtype=np.float64).copy()
            return self.pending.copy()
        except Exception:
            self.faulted = True
            raise

    def commit(self, *, transition_target=None):
        if self.faulted:
            raise ValueError("adaptive SIM player fault latched")
        try:
            if self.pending_adaptive is not None and transition_target is not None:
                raise ValueError("adaptive action cannot be relabeled as a blend")
            result = super().commit(transition_target=transition_target)
            if self.pending_adaptive is not None:
                if "sampling_trace" in self.pending_adaptive:
                    self.sampler.commit(self.pending_adaptive["action"])
                    self.sampled_controls += 1
                self.stream.last_action = self.pending_adaptive["action"].copy()
                result["action_memory_after"] = self.stream.last_action.copy()
                self.last_adaptive_trace = self.pending_adaptive
            feet = {self.prefix + "left_ankle_roll_link", self.prefix + "right_ankle_roll_link"}
            for name, force in self.world.last_ball_contact_bodies[self.prefix].items():
                if not name.startswith(self.prefix) or not np.isfinite(force) or force <= 1:
                    raise ValueError("actual own contact witness required")
                self.foot_contact_seen |= name in feet
                self.other_body_contact_seen |= name not in feet
            self.pending_adaptive = None
            return result
        except Exception:
            self.faulted = True
            raise
