"""SIM research prototype: initiate a qualified neural expert, then keep it.

Not yet admitted to native exam loaders. No learned termination, online RL,
new neural fitting or continuous multi-skill handoff is implemented here.
"""

import copy
import re
from typing import Any

import torch
from torch import nn

from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.motor_option_state import MotorOptionState
from rosclaw_soccer.rsi.neural_option_initiation import NeuralOptionInitiation
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter


class CommittedNeuralMotor(nn.Module):
    def __init__(
        self,
        parent: ConsolidatedMotorAdapter,
        experts: dict[int, RefinementMotorAdapter],
        initiation: NeuralOptionInitiation,
        *,
        generation_hash: str,
    ):
        super().__init__()
        if (
            type(generation_hash) is not str
            or re.fullmatch(r"sha256:[a-f0-9]{64}", generation_hash) is None
        ):
            raise ValueError("explicit model-generation commitment required, not authorization")
        self.generation_hash = generation_hash
        if (
            type(parent) is not ConsolidatedMotorAdapter
            or type(initiation) is not NeuralOptionInitiation
            or sorted(experts) != [1, 2, 3]
            or any(type(x) is not RefinementMotorAdapter for x in experts.values())
        ):
            raise ValueError("explicit same-parent3-expert neural option prototype required")
        reference = parent.state_dict()
        for expert in experts.values():
            other = expert.parent.state_dict()
            if other.keys() != reference.keys() or any(
                not torch.equal(v, other[k]) for k, v in reference.items()
            ):
                raise ValueError(
                    "neural options must share the EXACT whole learned recurrent parent"
                )
        self.parent = copy.deepcopy(parent).requires_grad_(False).eval()
        self.heads = nn.ModuleDict(
            {str(k): copy.deepcopy(v.head).requires_grad_(False).eval() for k, v in experts.items()}
        )
        self.initiation = copy.deepcopy(initiation).requires_grad_(False).eval()
        self.eval()

    def train(self, mode: bool = True) -> "CommittedNeuralMotor":
        if mode:
            raise ValueError("read-only option composition; no optimizer interface")
        super().train(False)
        return self

    def initial_state(self, batch: int = 1) -> MotorOptionState:
        if type(batch) is not int or batch != 1:
            raise ValueError("one player's private option history required")
        return MotorOptionState(self.parent.initial_state(), self.generation_hash)

    def forward_with_evidence(
        self, obs: torch.Tensor, state: MotorOptionState
    ) -> tuple[torch.Tensor, MotorOptionState, dict[str, Any]]:
        if type(state) is not MotorOptionState:
            raise ValueError("explicit private option state required")
        state.validate()
        if state.generation_hash != self.generation_hash:
            raise ValueError("private option history belongs to a different model generation")
        if (
            state.controls >= 512
            or obs.shape != (1, 1, 160)
            or obs.dtype != torch.float32
            or obs.device.type != "cpu"
        ):
            raise ValueError("bounded single-player own CPU160 proposal required")
        with torch.no_grad():
            parent_mean, recurrent, e = self.parent.forward_with_evidence(obs, state.recurrent)
            option = (
                state.option
                if state.option is not None
                else int(self.initiation(e["critic_features"][0]).item())
            )
            if option == 0:
                mean = parent_mean
                total = e["final_mean_delta_radians"]
            else:
                mix = torch.tanh(self.heads[str(option)](e["critic_features"]))
                if not torch.isfinite(mix).all():
                    raise ValueError("nonfinite neural expert rejected")
                total = torch.clamp(
                    e["final_mean_delta_radians"] * (1 - mix.abs()) + 0.2 * mix, -0.2, 0.2
                )
                mean = e["reference_nominal"] + total / self.parent.memory.parent.get_buffer(
                    "action_scale"
                )
        after = MotorOptionState(recurrent, self.generation_hash, option, state.controls + 1)
        return (
            mean,
            after,
            dict(
                **{
                    k: v
                    for k, v in e.items()
                    if k not in ("final_mean_delta_radians", "physical_retention_proven")
                },
                selected_neural_option=option,
                model_generation_hash=self.generation_hash,
                option_controls=after.controls,
                option_initiated=state.option is None,
                refinement_parent_mean=parent_mean,
                final_mean_delta_radians=total,
                physical_retention_proven=False,
                learned_termination=False,
            ),
        )

    def forward(
        self, obs: torch.Tensor, state: MotorOptionState
    ) -> tuple[torch.Tensor, MotorOptionState]:
        mean, after, _ = self.forward_with_evidence(obs, state)
        return mean, after
