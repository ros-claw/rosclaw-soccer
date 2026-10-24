"""Learn receiving decisions inside the ORIGINAL velocity-command envelope.

This is a command-level neural role policy, not direct torque or a claim of
end-to-end motor learning. The fixed velocity/body backend still owns the
29-joint action codec and PD execution. No physics, teacher, reset or I/O here.
"""

import numpy as np
import torch
from torch import nn

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, observation98
from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
from rosclaw_soccer.rsi.receiver_body_context import receiver_context14


class ReceivingCommandActor(nn.Module):
    observation_mean: torch.Tensor
    observation_scale: torch.Tensor
    command_center: torch.Tensor
    command_radius: torch.Tensor
    simulation_only = True
    policy_domain = "research.receiver_neural_velocity_command.v1"

    def __init__(self, observation_mean: torch.Tensor, observation_scale: torch.Tensor) -> None:
        super().__init__()
        if (
            any(
                type(x) is not torch.Tensor
                or x.shape != (112,)
                or x.dtype != torch.float32
                or x.device.type != "cpu"
                or not torch.isfinite(x).all()
                or x.requires_grad
                for x in (observation_mean, observation_scale)
            )
            or (observation_scale < 1e-3).any()
        ):
            raise ValueError("fixed finite CPU112 TRAIN-only normalization required")
        self.register_buffer("observation_mean", observation_mean.clone())
        self.register_buffer("observation_scale", observation_scale.clone())
        self.register_buffer("command_center", torch.tensor([0.25, 0.0, 0.0], dtype=torch.float32))
        self.register_buffer("command_radius", torch.tensor([0.75, 0.5, 1.0], dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(112, 128, dtype=torch.float32),
            nn.ELU(),
            nn.Linear(128, 128, dtype=torch.float32),
            nn.ELU(),
            nn.Linear(128, 3, dtype=torch.float32),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if (
            observations.ndim != 2
            or observations.shape[1] != 112
            or not 1 <= len(observations) <= 8192
            or observations.dtype != torch.float32
            or observations.device != self.observation_mean.device
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("finite actual own112 receiving observations required")
        latent = self.net((observations - self.observation_mean) / self.observation_scale)
        command = self.command_center + self.command_radius * torch.tanh(latent)
        if not torch.isfinite(command).all():
            raise ValueError("nonfinite receiving command; never repair or execute")
        return command


def command_inputs(
    qpos: np.ndarray,
    qvel: np.ndarray,
    angular_velocity_body: np.ndarray,
    *,
    params: VelocityPolicyParameters,
    last_action: np.ndarray,
    phase: np.float32,
    home_xy: np.ndarray,
    pass_direction: np.ndarray,
    anchor_world: np.ndarray,
    launch_seen: bool,
    foot_contact_seen: bool,
    other_body_contact_seen: bool,
) -> tuple[np.ndarray, np.ndarray, np.float32, bool]:
    """Use a causal pursuit PROPOSAL observation, not the teacher's command.

    command-dependent gait masking is recomputed with the original codec; it
    cannot be repaired by replacing just channels6:9 in a teacher observation.
    The motor subsequently recomputes its own98 with the ACTUAL neural command.
    This function never advances a controller or writes action/history state.
    """
    cues = receiver_context14(
        qpos,
        qvel,
        anchor_world,
        foot_contact_seen=foot_contact_seen,
        other_body_contact_seen=other_body_contact_seen,
    )
    qw, x, y, z = qpos[3:7]
    yaw = float(np.arctan2(2 * (qw * z + x * y), 1 - 2 * (y * y + z * z)))
    intent = receiving_intent(
        root_xy=qpos[:2],
        yaw=yaw,
        ball_xy=qpos[36:38],
        ball_velocity_xy=qvel[35:37],
        home_xy=home_xy,
        pass_direction=pass_direction,
        launch_seen=launch_seen,
        ready_anchor_xy=anchor_world[:2],
        mode="pursuit-v1",
    )
    baseline = np.array(intent.command_body, dtype=np.float32)
    obs, next_phase = observation98(
        qpos, qvel, angular_velocity_body, baseline, last_action, phase, params
    )
    return np.r_[obs, cues], baseline, next_phase, intent.launch_seen
