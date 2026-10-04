"""Independently reconstruct executed CPU Torch arithmetic from ONNX weights.

The executed Torch policy is NOT universally bit-identical to original ONNX
across quantizer boundaries. Retain every original-ONNX token difference.
This separate contract never changes the historical strict ONNX audit.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.foundation_observation_capture import PREFIX
from rosclaw_soccer.sim.contracts import hash_bytes


class TorchCallAuditor:
    """Read-only numeric component; no simulator, runtime, training or IO transport."""

    def __init__(self, model_root: Path) -> None:
        import onnx
        import onnxruntime as ort
        import torch
        from onnx import numpy_helper

        self._torch = torch
        self._paths = [
            model_root / "low_latency" / f"model_{k}.onnx" for k in ("encoder", "decoder")
        ]
        if any(p.stat().st_size > 256000000 for p in self._paths):
            raise ValueError("bounded source graph required")
        self._hashes = [hash_bytes(p.read_bytes()) for p in self._paths]
        graphs = [onnx.load(p, load_external_data=False) for p in self._paths]
        if any(
            t.data_location == onnx.TensorProto.EXTERNAL
            for g in graphs
            for t in g.graph.initializer
        ):
            raise ValueError("bounded inline source weights required")
        concat = [n for n in graphs[0].graph.node if n.name == "/Concat_2"]
        if len(concat) != 1 or list(concat[0].input) != [
            "/Reshape_1_output_0",
            "/Reshape_3_output_0",
        ]:
            raise ValueError("exact low-latency G1 encoder layout required")
        enc, dec = [{t.name: numpy_helper.to_array(t) for t in g.graph.initializer} for g in graphs]

        def tensor(value: Any) -> Any:
            array = np.asarray(value, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError("finite source weights required")
            return torch.tensor(array, dtype=torch.float32, device="cpu")

        self._enc = [
            (
                tensor(enc[f"module.encoders.g1.module.{i}.weight"].T),
                tensor(enc[f"module.encoders.g1.module.{i}.bias"]),
            )
            for i in range(0, 9, 2)
        ]
        names = [n.input[1] for n in graphs[1].graph.node if n.op_type == "MatMul"]
        if len(names) != 9:
            raise ValueError("exact exported G1 decoder layer count required")
        self._dec = [
            (tensor(dec[n]), tensor(dec[f"module.decoders.g1_dyn.module.{2 * i}.bias"]))
            for i, n in enumerate(names)
        ]
        constants = {
            n.output[0]: numpy_helper.to_array(n.attribute[0].t)
            for n in graphs[0].graph.node
            if n.op_type == "Constant" and n.name.startswith("/quantizer/")
        }
        self._quant = [
            tensor(constants[f"/quantizer/Constant_{i}_output_0"]).repeat(2) for i in range(1, 5)
        ]
        for layers, width, output in ((self._enc, 640, 64), (self._dec, 994, 29)):
            for weight, bias in layers:
                if weight.ndim != 2 or weight.shape[0] != width or bias.shape != (weight.shape[1],):
                    raise ValueError("exact numeric layer dimensions required")
                width = weight.shape[1]
            if width != output:
                raise ValueError("exact G1 network output dimensions required")
        if any(v.shape != (64,) for v in self._quant) or bool((self._quant[3] <= 0).any()):
            raise ValueError("exact finite scalar quantization contract required")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self._original = ort.InferenceSession(
            str(self._paths[0]), options, providers=["CPUExecutionProvider"]
        )
        self._decoder = ort.InferenceSession(
            str(self._paths[1]), options, providers=["CPUExecutionProvider"]
        )
        graphs[0].graph.output.append(
            onnx.helper.make_tensor_value_info(
                "/quantizer/Sub_output_0", onnx.TensorProto.FLOAT, None
            )
        )
        self._instrumented = ort.InferenceSession(
            graphs[0].SerializeToString(), options, providers=["CPUExecutionProvider"]
        )

    def _forward(self, values: Any, layers: Any) -> Any:
        for i, (weight, bias) in enumerate(layers):
            values = self._torch.matmul(values, weight) + bias
            if i < len(layers) - 1:
                values = values * self._torch.sigmoid(values)
        if not bool(self._torch.isfinite(values).all()):
            raise ValueError("finite executed numeric network required")
        return values

    def recompute(self, arrays: dict[str, Any]) -> dict[str, Any]:
        torch = self._torch
        if torch.get_num_threads() != 1:
            raise ValueError("explicit source CPU one-thread arithmetic required")
        features, tokens, observations, actions = [
            np.asarray(arrays[PREFIX + k])
            for k in ("encoder_features", "latent_token", "decoder_input", "raw_action_isaac")
        ]
        if features.ndim != 3:
            raise ValueError("recorded features must have frame, lane and feature axes")
        frames, lanes = features.shape[:2]
        if not 1 <= frames <= 20000 or not 1 <= lanes <= 4096 or frames * lanes > 200000:
            raise ValueError("bounded recorded numeric calls required")
        for value, width in ((features, 640), (tokens, 64), (observations, 994), (actions, 29)):
            if (
                value.shape != (frames, lanes, width)
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                raise ValueError("aligned finite float32 actual network calls required")
        differences, before_error, original_decoder_error = [], 0.0, 0.0
        with torch.no_grad():
            for frame in range(frames):
                for lane in range(lanes):
                    f = features[frame, lane][None]
                    value = torch.tensor(f, dtype=torch.float32)
                    packed = torch.cat(
                        (value[:, :580].reshape(1, 10, 58), value[:, 580:].reshape(1, 10, 6)), dim=2
                    ).reshape(1, 640)
                    latent = self._forward(packed, self._enc)
                    shift, scale, offset, divisor = self._quant
                    bounded = torch.tanh(latent + shift) * scale - offset
                    quantized = (bounded + (torch.round(bounded) - bounded)) / divisor
                    actual = self._forward(torch.tensor(observations[frame, lane][None]), self._dec)
                    if not np.array_equal(
                        quantized.numpy()[0], tokens[frame, lane]
                    ) or not np.array_equal(actual.numpy()[0], actions[frame, lane]):
                        raise ValueError(
                            "recorded call differs from independent executed-Torch reconstruction"
                        )
                    original_input = np.zeros((1, 1247), dtype=np.float32)
                    original_input[:, 4:584], original_input[:, 590:650] = f[:, :580], f[:, 580:]
                    inputs = {self._original.get_inputs()[0].name: original_input}
                    original = self._original.run(None, inputs)[0].reshape(64)
                    tapped = self._instrumented.run(None, inputs)
                    if not np.array_equal(tapped[0].reshape(64), original):
                        raise ValueError("instrumentation changed original quantization output")
                    continuous = tapped[1].reshape(64)
                    if not np.isfinite(original).all() or not np.isfinite(continuous).all():
                        raise ValueError("finite original encoder inference required")
                    error = float(np.max(np.abs(bounded.numpy()[0] - continuous)))
                    if error > 1e-4:
                        raise ValueError(
                            "before-round Torch/ONNX difference exceeds independent profile"
                        )
                    before_error = max(before_error, error)
                    decoded = self._decoder.run(
                        None, {self._decoder.get_inputs()[0].name: observations[frame, lane][None]}
                    )[0].reshape(29)
                    if not np.isfinite(decoded).all():
                        raise ValueError("finite original decoder inference required")
                    d = float(np.max(np.abs(decoded - actions[frame, lane])))
                    if d > 1e-4:
                        raise ValueError("original decoder given recorded token differs")
                    original_decoder_error = max(original_decoder_error, d)
                    changed = np.flatnonzero(original != tokens[frame, lane])
                    if len(changed):
                        differences.append(
                            dict(
                                frame=frame,
                                lane=lane,
                                token_indices=changed.tolist(),
                                actual_token=tokens[frame, lane, changed].tolist(),
                                original_token=original[changed].tolist(),
                                actual_before_round=bounded.numpy()[0, changed].tolist(),
                                original_before_round=continuous[changed].tolist(),
                            )
                        )
        if [hash_bytes(p.read_bytes()) for p in self._paths] != self._hashes:
            raise ValueError("original numeric weights changed during audit")
        return dict(
            schema="soccer.rsi.executed_torch_foundation_audit.v1",
            recomputed_executed_calls=frames * lanes,
            torch_encoder_tolerance=0.0,
            torch_decoder_tolerance=0.0,
            original_encoder_bit_parity_passed=not differences,
            original_token_differences=differences,
            before_round_cross_framework_max_abs_error=before_error,
            before_round_tolerance=1e-4,
            original_decoder_given_recorded_token_max_abs_error=original_decoder_error,
            original_decoder_tolerance=1e-4,
            original_weight_hashes=self._hashes,
            audit_source_hash=hash_bytes(Path(__file__).read_bytes()),
            qualification="EXECUTED_CPU_TORCH_ONLY_NOT_UNIVERSAL_ORIGINAL_ONNX_ENCODER_PARITY",
            activation_ceiling="SIM_ONLY",
            planner_reference_reconstructed=False,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )
