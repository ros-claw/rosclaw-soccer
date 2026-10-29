"""Causal smooth contact-focused SIM motor adapter, not a safety controller."""

import torch

from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor


def precontact_gate(observations: torch.Tensor) -> torch.Tensor:
    """Smooth gate from current own-frame ball position and relative velocity.

    The chosen bounds are development hypotheses, not a learned predictor or
    guarantee of contact. There is no event label or future input here.
    """
    if (
        observations.ndim != 2
        or observations.shape[1] != 112
        or observations.dtype != torch.float32
        or not torch.isfinite(observations).all()
    ):
        raise ValueError("finite actual 112-channel observations required")
    position, velocity = observations[:, 98:101], observations[:, 101:104]
    distance = torch.linalg.vector_norm(position, dim=1)
    closing = -(position * velocity).sum(dim=1) / distance.clamp_min(1e-6)

    def smooth(x):
        x = x.clamp(0, 1)
        return x * x * (3 - 2 * x)

    return smooth((1.5 - distance) / 0.2) * smooth((closing - 0.5) / 0.2)


class PrecontactReceivingMotor(AnchoredReceivingMotor):
    """Train only a smooth causal residual; preserve the original base exactly."""

    policy_domain = "research.receiver_precontact_velocity_actor.v1"

    def forward_with_evidence(self, observations, reference):
        mean, evidence = super().forward_with_evidence(observations, reference)
        gate = precontact_gate(observations).unsqueeze(1)
        gated = reference + gate * (mean - reference)
        evidence["precontact_gate"] = gate[:, 0]
        evidence["realized_delta_radians"] = (
            gated.double() - reference.double()
        ) * self.action_scale.double()
        return gated, evidence
