"""Differentiable copy of the exact research kick prior, not a trained successor.

Raw outputs are 29 ISAAC-ordered action coordinates, NOT joint torques or a
hardware interface. The existing PD mapping, clips and execution gates remain
outside this model. Import only in optional Torch research workflows.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from rosclaw_soccer.providers.g1.velocity_policy import file_hash

KICK_PRIOR_HASH = "sha256:b952af481c47402c40a883ea214635368b44ee36de356780258d04601eaa3774"


class KickPriorActor(nn.Module):
    simulation_only = True
    hardware_authorized = False

    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        super().__init__()
        dimensions = (547, 512, 256, 128, 29)
        shapes: dict[str, tuple[int, ...]] = {}
        for layer, (before, after) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
            shapes[f"actor.{layer * 2}.weight"] = (after, before)
            shapes[f"actor.{layer * 2}.bias"] = (after,)
        if set(tensors) != set(shapes) or any(
            x.shape != shapes[k] or x.dtype != np.float32 or not np.isfinite(x).all()
            for k, x in tensors.items()
        ):
            raise ValueError("exact finite547-512-256-128-29 ELU tensors required")
        self.actor = nn.Sequential(
            nn.Linear(547, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )
        with torch.no_grad():
            for index in (0, 2, 4, 6):
                linear = self.actor[index]
                assert isinstance(linear, nn.Linear)
                linear.weight.copy_(torch.from_numpy(tensors[f"actor.{index}.weight"].copy()))
                linear.bias.copy_(torch.from_numpy(tensors[f"actor.{index}.bias"].copy()))

    @classmethod
    def from_onnx(cls, path: Path, *, expected_hash: str) -> "KickPriorActor":
        import onnx

        if (
            path.is_symlink()
            or expected_hash != KICK_PRIOR_HASH
            or file_hash(path) != expected_hash
        ):
            raise ValueError("exact frozen research kick ONNX required")
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
        nodes = list(model.graph.node)
        if (
            model.functions
            or len(model.graph.input) != 1
            or len(model.graph.output) != 1
            or [n.op_type for n in nodes] != ["Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]
            or model.graph.input[0].name != "obs"
            or model.graph.output[0].name != "actions"
            or any(n.domain for n in nodes)
            or any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
        ):
            raise ValueError(
                "exact feedforward graph without normalization or external data required"
            )
        for value, width in ((model.graph.input[0], 547), (model.graph.output[0], 29)):
            tensor = value.type.tensor_type
            if tensor.elem_type != onnx.TensorProto.FLOAT or (
                [d.dim_value for d in tensor.shape.dim] != [1, width]
            ):
                raise ValueError("exact float32 ONNX input/output contract required")
        previous = "obs"
        for index, node in enumerate(nodes):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            expected = (
                {"alpha": 1.0, "beta": 1.0, "transB": 1} if index % 2 == 0 else {"alpha": 1.0}
            )
            inputs = [previous]
            if index % 2 == 0:
                inputs += [f"actor.{index}.weight", f"actor.{index}.bias"]
            if list(node.input) != inputs or len(node.output) != 1 or attrs != expected:
                raise ValueError("exact graph connectivity and arithmetic required")
            previous = node.output[0]
        if previous != "actions":
            raise ValueError("exact action output required")
        tensors = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
        if len(tensors) != len(model.graph.initializer):
            raise ValueError("unique weight names required")
        return cls(tensors)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if (
            observation.ndim != 2
            or observation.shape[1] != 547
            or not 1 <= len(observation) <= 8192
        ):
            raise ValueError("nonempty bounded batch of actual547 observations required")
        parameter = self.actor[0].weight
        if (
            observation.dtype != torch.float32
            or observation.device != parameter.device
            or (not torch.isfinite(observation).all())
        ):
            raise ValueError("finite float32 observation on the actor device required")
        return cast(torch.Tensor, self.actor(observation))


@dataclass(frozen=True)
class KickTensorContract:
    name: str
    shape: tuple[int, int]
    type: str = "tensor(float)"


class TorchKickSession:
    """Exact batch1 inference ABI for a separately measured SIM policy instance."""

    def __init__(self, actor: KickPriorActor) -> None:
        self.actor = actor.eval()
        self.calls = 0

    def get_inputs(self) -> list[KickTensorContract]:
        return [KickTensorContract("obs", (1, 547))]

    def get_outputs(self) -> list[KickTensorContract]:
        return [KickTensorContract("actions", (1, 29))]

    def run(self, outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        if outputs != ["actions"] or set(feed) != {"obs"} or feed["obs"].shape != (1, 547):
            raise ValueError("exact frozen kick inference ABI required")
        with torch.no_grad():
            result = self.actor(torch.from_numpy(feed["obs"].copy())).numpy().copy()
        if result.shape != (1, 29) or not np.isfinite(result).all():
            raise ValueError("finite raw29 prior actions required")
        self.calls += 1
        return [result]


def torch_kick_policy_type(original: Any, model: Path) -> Any:
    """Fresh network per native episode; no shared player state or hardware IO."""

    class TorchKickPolicy(original):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.runtime_mode != "sim":
                raise ValueError("research Torch kick inference is SIM_ONLY")
            self.ort_session = TorchKickSession(
                KickPriorActor.from_onnx(model, expected_hash=KICK_PRIOR_HASH)
            )

    return TorchKickPolicy
