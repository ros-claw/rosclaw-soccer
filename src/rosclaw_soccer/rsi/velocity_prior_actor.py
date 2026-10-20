"""Optional Torch actor initialized from the qualified SIM velocity ONNX.

The first 98 inputs retain their exact deployment semantics. Extra reference
inputs start with zero influence, but their weights remain trainable. This is
an initialization mechanism, not a trained skill or a simulator controller.
"""

from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from rosclaw_soccer.providers.g1.velocity_policy import file_hash


class VelocityPriorActor(nn.Module):
    prior_mean: torch.Tensor
    prior_denominator: torch.Tensor

    def __init__(self, tensors: dict[str, np.ndarray], *, reference_features: int = 70):
        super().__init__()
        if type(reference_features) is not int or not 1 <= reference_features <= 1024:
            raise ValueError("bounded explicit reference feature count required")
        shapes = {
            "obs_normalizer._mean": (1, 98),
            "onnx::Div_24": (1, 98),
            "mlp.0.weight": (512, 98),
            "mlp.0.bias": (512,),
            "mlp.2.weight": (256, 512),
            "mlp.2.bias": (256,),
            "mlp.4.weight": (128, 256),
            "mlp.4.bias": (128,),
            "mlp.6.weight": (29, 128),
            "mlp.6.bias": (29,),
        }
        if set(tensors) != set(shapes) or any(
            x.shape != shapes[k] or x.dtype != np.float32 or not np.isfinite(x).all()
            for k, x in tensors.items()
        ):
            raise ValueError("exact finite float32 98-512-256-128-29 prior required")
        if np.any(tensors["onnx::Div_24"] <= 0):
            raise ValueError("strictly positive fixed normalization denominator required")
        self.reference_features = reference_features
        self.register_buffer("prior_mean", torch.from_numpy(tensors["obs_normalizer._mean"].copy()))
        self.register_buffer("prior_denominator", torch.from_numpy(tensors["onnx::Div_24"].copy()))
        self.mlp = nn.Sequential(
            nn.Linear(98 + reference_features, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )
        with torch.no_grad():
            for index in (0, 2, 4, 6):
                layer = self.mlp[index]
                assert isinstance(layer, nn.Linear)
                layer.bias.copy_(torch.from_numpy(tensors[f"mlp.{index}.bias"].copy()))
                weight = torch.from_numpy(tensors[f"mlp.{index}.weight"].copy())
                if index == 0:
                    layer.weight.zero_()
                    layer.weight[:, :98].copy_(weight)
                else:
                    layer.weight.copy_(weight)

    @classmethod
    def from_onnx(
        cls, path: Path, *, expected_hash: str, reference_features: int = 70
    ) -> "VelocityPriorActor":
        import onnx

        if file_hash(path) != expected_hash:
            raise ValueError("qualified immutable ONNX hash required before loading")
        model = onnx.load(str(path), load_external_data=False)
        nodes = list(model.graph.node)
        if (
            model.functions
            or len(model.graph.input) != 1
            or model.graph.input[0].name != "obs"
            or len(model.graph.output) != 1
            or model.graph.output[0].name != "actions"
            or [n.op_type for n in nodes]
            != ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]
            or any(n.domain for n in nodes)
            or any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
            or nodes[0].input != ["obs", "obs_normalizer._mean"]
            or nodes[1].input != [nodes[0].output[0], "onnx::Div_24"]
            or nodes[-1].output != ["actions"]
        ):
            raise ValueError("exact qualified feedforward ONNX graph required")
        for i, node in enumerate(nodes):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            expected = (
                {"alpha": 1.0, "beta": 1.0, "transB": 1}
                if node.op_type == "Gemm"
                else {"alpha": 1.0}
                if node.op_type == "Elu"
                else {}
            )
            if attrs != expected or (i and node.input[0] != nodes[i - 1].output[0]):
                raise ValueError("qualified graph connectivity and arithmetic required")
        for i, layer in zip((2, 4, 6, 8), (0, 2, 4, 6), strict=True):
            if list(nodes[i].input[1:]) != [f"mlp.{layer}.weight", f"mlp.{layer}.bias"]:
                raise ValueError("qualified layer tensor identities required")
        tensors = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
        if len(tensors) != len(model.graph.initializer):
            raise ValueError("unique qualified tensor names required")
        return cls(tensors, reference_features=reference_features)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != 98 + self.reference_features:
            raise ValueError("explicit velocity98 plus reference features required")
        if observations.dtype != torch.float32 or not torch.isfinite(observations).all():
            raise ValueError("finite float32 actor observations required")
        velocity = (observations[:, :98] - self.prior_mean) / self.prior_denominator
        return cast(torch.Tensor, self.mlp(torch.cat((velocity, observations[:, 98:]), dim=1)))
