"""Trainable research copy of the one admitted external PAiD actor.

Not a generic ONNX importer or a deployable policy. The external CC-BY-NC
weights and any derived weights remain research-only. State is caller-owned:
independent players must never share recurrent history. No simulator imports.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy

PUBLISHED_HASH = "sha256:489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1"


def onnx_to_torch_gates(array: np.ndarray) -> torch.Tensor:
    """ONNX IOFC -> PyTorch IFGO, preserving the two biases separately."""
    if array.shape[0] != 512 or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("finite float32 four128-gate array required")
    return torch.from_numpy(np.concatenate([np.split(array, 4)[i] for i in (0, 2, 3, 1)]))


class PaidTrainableActor(nn.Module):
    mean: torch.Tensor
    denominator: torch.Tensor

    def __init__(self, path: Path, *, expected_hash: str) -> None:
        import onnx

        super().__init__()
        if expected_hash != PUBLISHED_HASH:
            raise ValueError("only the exact independently admitted research actor is supported")
        # Validate the complete ONNX contract, metadata, clock and digest before
        # interpreting any weights; the reader's state is not used for inference.
        PaidResearchPolicy(path, expected_hash=expected_hash, horizon=1)
        graph = onnx.load(path, load_external_data=False)
        arrays = {x.name: onnx.numpy_helper.to_array(x).copy() for x in graph.graph.initializer}
        self.register_buffer("mean", torch.from_numpy(arrays["parent.normalizer._mean"]))
        self.register_buffer("denominator", torch.from_numpy(arrays["onnx::Div_199"]))
        if self.mean.shape != (1, 160) or self.denominator.shape != (1, 160):
            raise ValueError("exact admitted normalizer shape required")
        if not torch.isfinite(self.mean).all() or not (self.denominator > 0).all():
            raise ValueError("finite mean and positive normalizer denominator required")
        self.rnn = nn.LSTM(160, 128, num_layers=2)
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 29),
        )
        lstms = [n for n in graph.graph.node if n.op_type == "LSTM"]
        if len(lstms) != 2:
            raise ValueError("two exact admitted recurrent layers required")
        with torch.no_grad():
            for layer, node in enumerate(lstms):
                w, r, b = [arrays[name][0] for name in node.input[1:4]]
                for name, data in (
                    ("weight_ih", w),
                    ("weight_hh", r),
                    ("bias_ih", b[:512]),
                    ("bias_hh", b[512:]),
                ):
                    parameter = getattr(self.rnn, f"{name}_l{layer}")
                    converted = onnx_to_torch_gates(data)
                    if parameter.shape != converted.shape:
                        raise ValueError("exact recurrent weight shape required")
                    parameter.copy_(converted)
            for index in (0, 2, 4, 6):
                for name in ("weight", "bias"):
                    getattr(self.actor[index], name).copy_(
                        torch.from_numpy(arrays[f"parent.actor.{index}.{name}"])
                    )

    def forward(
        self, observation: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if (
            observation.ndim != 3
            or observation.shape[-1] != 160
            or observation.shape[0] < 1
            or observation.shape[1] < 1
            or observation.dtype != torch.float32
            or len(state) != 2
        ):
            raise ValueError("explicit nonempty time,batch,160 float32 observation required")
        if any(
            x.shape != (2, observation.shape[1], 128)
            or x.dtype != torch.float32
            or x.device != observation.device
            for x in state
        ) or not all(torch.isfinite(x).all() for x in (observation, *state)):
            raise ValueError("finite explicit per-player recurrent states required")
        # A per-player slice can have the right values/shape but noncontiguous
        # strides. cuDNN requires contiguous h/c; copy layout, never crop/pad or
        # share another player's memory. CPU accepts the same explicit layout.
        output, state = self.rnn(
            (observation - self.mean) / self.denominator,
            (state[0].contiguous(), state[1].contiguous()),
        )
        return self.actor(output), state


class PaidTorchResearchPolicy(PaidResearchPolicy):
    """CPU inference bridge for a new, separately measured native experiment."""

    def __init__(self, path: Path, *, expected_hash: str, horizon: int) -> None:
        super().__init__(path, expected_hash=expected_hash, horizon=horizon)
        self.actor = PaidTrainableActor(path, expected_hash=expected_hash).eval()

    def step(self, observation: np.ndarray, *, frame: int) -> dict[str, Any]:
        if self.faulted:
            raise ValueError("teacher inference fault latched; new episode object required")
        try:
            if (
                type(frame) is not int
                or frame != self.frame
                or not 0 <= frame < self.horizon
                or observation.shape != (160,)
                or observation.dtype != np.float32
                or not np.isfinite(observation).all()
            ):
                raise ValueError(
                    "consecutive actual finite160 input, no crop/pad/reference extension"
                )
            before = dict(
                h_before=self.h.copy(), c_before=self.c.copy(), observation=observation.copy()
            )
            self.attempted_neural_calls += 1
            with torch.no_grad():
                ac, (h, c) = self.actor(
                    torch.from_numpy(observation.copy()).reshape(1, 1, 160),
                    (torch.from_numpy(self.h), torch.from_numpy(self.c)),
                )
            self.neural_calls += 1
            action = ac[0, 0].numpy().copy()
            h, c = h.numpy().copy(), c.numpy().copy()
            if (
                action.shape != (29,)
                or h.shape != (2, 1, 128)
                or c.shape != (2, 1, 128)
                or any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (action, h, c))
            ):
                raise ValueError("finite exact trainable actor outputs required")
            self.h, self.c = h, c
            self.frame += 1
            return dict(**before, action=action, h_after=h.copy(), c_after=c.copy())
        except Exception:
            self.faulted = True
            raise
