"""Frozen, batchable SONIC G1 branch, numerically checked against source ONNX.

Only G1 reference mode is supported. Teleoperation/SMPL branches are not
silently approximated. No pickle, planner, robot transport or weight update.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicModelVariant, qualify_g1_sonic
from rosclaw_soccer.sim.contracts import hash_bytes


class FrozenSonicG1Torch:
    def __init__(
        self, model_root: Path, *, variant: G1SonicModelVariant = "sonic_v1_1", device: str = "cpu"
    ) -> None:
        import onnx
        import onnxruntime as ort
        import torch
        from onnx import numpy_helper

        self.qualification = qualify_g1_sonic(model_root, variant)
        self.qualification.require_eligible()
        self._torch = torch
        self.device = torch.device(device)
        self.variant = variant
        self._encoder: list[tuple[Any, Any]] = []
        self._decoder: list[tuple[Any, Any]] = []
        paths = [model_root / variant / f"model_{name}.onnx" for name in ("encoder", "decoder")]
        graphs = []
        for path in paths:
            if path.stat().st_size > 256_000_000:
                raise ValueError("SONIC graph exceeds bounded inline model size")
            graph = onnx.load(path, load_external_data=False).graph
            if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.initializer):
                raise ValueError("SONIC requires inline numeric model tensors")
            graphs.append(graph)

        def tensor(value: Any) -> Any:
            array = np.asarray(value, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite SONIC numeric tensor")
            return torch.tensor(array, dtype=torch.float32, device=self.device)

        enc = {t.name: numpy_helper.to_array(t) for t in graphs[0].initializer}
        for layer in range(0, 9, 2):
            prefix = f"module.encoders.g1.module.{layer}"
            self._encoder.append((tensor(enc[prefix + ".weight"].T), tensor(enc[prefix + ".bias"])))
        constants = {
            n.output[0]: numpy_helper.to_array(n.attribute[0].t)
            for n in graphs[0].node
            if n.op_type == "Constant" and n.name.startswith("/quantizer/")
        }
        self._quant = [
            tensor(constants[f"/quantizer/Constant_{i}_output_0"]).repeat(2) for i in range(1, 5)
        ]
        if any(v.shape != (64,) for v in self._quant) or bool((self._quant[3] <= 0).any()):
            raise ValueError("unsupported G1 scalar quantizer contract")
        dec = {t.name: numpy_helper.to_array(t) for t in graphs[1].initializer}
        matrices = [n.input[1] for n in graphs[1].node if n.op_type == "MatMul"]
        if len(matrices) != 9:
            raise ValueError("unsupported SONIC decoder graph")
        for i, name in enumerate(matrices):
            self._decoder.append(
                (tensor(dec[name]), tensor(dec[f"module.decoders.g1_dyn.module.{2 * i}.bias"]))
            )
        for layers, first, last in ((self._encoder, 640, 64), (self._decoder, 994, 29)):
            width = first
            for weight, bias in layers:
                if weight.ndim != 2 or weight.shape[0] != width or bias.shape != (weight.shape[1],):
                    raise ValueError("SONIC linear layer identity differs")
                width = weight.shape[1]
            if width != last:
                raise ValueError("SONIC output dimension differs")
        if [hash_bytes(p.read_bytes()) for p in paths] != [
            self.qualification.encoder_hash,
            self.qualification.decoder_hash,
        ]:
            raise ValueError("SONIC source changed during numeric extraction")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        sessions = [
            ort.InferenceSession(str(p), sess_options=options, providers=["CPUExecutionProvider"])
            for p in paths
        ]
        rng = np.random.default_rng(233)
        encoder_errors, decoder_errors = [], []
        with torch.no_grad():
            for _ in range(4):
                x = rng.normal(0, 0.2, (1, self.qualification.encoder_input_size)).astype(
                    np.float32
                )
                x[:, :4] = 0
                expected = sessions[0].run(None, {sessions[0].get_inputs()[0].name: x})[0]
                actual = self.encode_g1(tensor(x[:, 4:644])).cpu().numpy()
                encoder_errors.append(float(np.max(np.abs(actual - expected))))
                y = rng.normal(0, 0.2, (1, 994)).astype(np.float32)
                expected = (
                    sessions[1].run(None, {sessions[1].get_inputs()[0].name: y})[0].reshape(1, 29)
                )
                actual = self.decode(tensor(y)).cpu().numpy()
                decoder_errors.append(float(np.max(np.abs(actual - expected))))
        self.parity = dict(
            encoder_max_abs=max(encoder_errors), decoder_max_abs=max(decoder_errors), samples=4
        )
        if self.parity["encoder_max_abs"] != 0 or self.parity["decoder_max_abs"] > 1e-4:
            raise ValueError(f"SONIC G1 reconstruction differs from source ONNX: {self.parity}")

    def _input(self, value: Any, width: int) -> Any:
        torch = self._torch
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if (
            value.ndim != 2
            or value.shape[1] != width
            or not 1 <= value.shape[0] <= 4096
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 1000).any())
        ):
            raise ValueError("bounded finite SONIC batch required")
        return value

    def _mlp(self, value: Any, layers: list[tuple[Any, Any]]) -> Any:
        for i, (weight, bias) in enumerate(layers):
            value = value @ weight + bias
            if i + 1 < len(layers):
                value = value * self._torch.sigmoid(value)
        if not bool(self._torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite SONIC MLP output")
        return value

    def encode_g1(self, features: Any) -> Any:
        features = self._input(features, 640)
        # Reproduce the exported graph's reshape/concat, not an assumed
        # frame-interleaved native sensor layout. Native input is two 290-value
        # position/velocity blocks, then the 60-value orientation block.
        packed = self._torch.cat(
            (features[:, :580].reshape(-1, 10, 58), features[:, 580:].reshape(-1, 10, 6)),
            dim=2,
        ).reshape(-1, 640)
        value = self._mlp(packed, self._encoder)
        shift, scale, offset, divisor = self._quant
        bounded = self._torch.tanh(value + shift) * scale - offset
        # Preserve the source straight-through arithmetic in frozen inference.
        quantized = bounded + (self._torch.round(bounded) - bounded)
        return quantized / divisor

    def decode(self, observation: Any) -> Any:
        return self._mlp(self._input(observation, 994), self._decoder)
