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

from rosclaw_soccer.providers.g1.velocity_policy import (
    VelocityPolicyParameters,
    file_hash,
    observation98,
    phase_delta,
    target29,
)
from rosclaw_soccer.sim.contracts import hash_json


def phase_table(length: int = 4096) -> np.ndarray:
    """Pure lookup: reset primes one tick, inference advances one more tick."""
    if type(length) is not int or not 1 <= length <= 100000:
        raise ValueError("bounded explicit phase horizon required")
    phase = phase_delta()
    values = []
    for _ in range(length):
        phase = np.float32(np.fmod(np.float32(phase + phase_delta()), np.float32(1)))
        values.append(float(phase))
    return np.array(values, dtype=np.float32)


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


class LearnedPriorNativeSession:
    """SIM research inference with explicit learned identity, not old ONNX proof.

    Reference suffix is zero for a retention test; no reference-motion or
    football skill is inferred. State/motor execution stays in the simulator.
    """

    def __init__(
        self,
        model: Path,
        config: Path,
        *,
        expected_model_hash: str,
        expected_config_hash: str,
        checkpoint: Path,
        expected_checkpoint_hash: str,
        arm: str,
        reference_mode: str = "zero70",
    ) -> None:
        if (
            arm not in ("initial", "final")
            or reference_mode not in ("zero70", "declared70", "zero79", "declared79")
            or file_hash(checkpoint) != expected_checkpoint_hash
        ):
            raise ValueError("explicit arm and immutable learned checkpoint required")
        self.params = VelocityPolicyParameters.load(config, expected_hash=expected_config_hash)
        self.reference_features = 79 if reference_mode.endswith("79") else 70
        self.model = VelocityPriorActor.from_onnx(
            model, expected_hash=expected_model_hash, reference_features=self.reference_features
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            state.get("stable_prior_rehearsal") is not True
            or state.get("actor_observation_dim") != 98 + self.reference_features
        ):
            raise ValueError("actual stable-prior learner checkpoint required")
        if self.reference_features == 79:
            from rosclaw_soccer.rsi.ball_aware_inputs import BALL_AWARE_ABI

            if state.get("ball_aware_abi") != BALL_AWARE_ABI:
                raise ValueError(
                    "explicit new ball-aware ABI required; no old certificate inheritance"
                )
        actor_state = (
            state["initial_actor_state_dict"]
            if arm == "initial"
            else state["algorithm"]["actor_state_dict"]
        )
        expected_keys = set(self.model.state_dict())
        actual_keys = {k.removeprefix("mlp.") for k in actor_state if k.startswith("mlp.")}
        if actual_keys != expected_keys:
            raise ValueError("complete exact learned actor architecture required")
        self.model.load_state_dict({k: actor_state["mlp." + k] for k in expected_keys}, strict=True)
        self.model.eval()
        self.contract_hash = hash_json(
            dict(
                schema=(
                    "soccer.rsi.learned_prior_native_inference.v3"
                    if self.reference_features == 79
                    else "soccer.rsi.learned_prior_native_inference.v2"
                ),
                checkpoint_hash=expected_checkpoint_hash,
                arm=arm,
                parameter_hash=self.params.contract_hash,
                reference_features=reference_mode,
                hardware_authorized=False,
            )
        )
        self.phase = phase_delta()
        self.last_action = np.zeros(29, dtype=np.float32)
        self.last_observation = np.zeros(98, dtype=np.float32)
        self.next_frame = 0
        self.neural_calls = 0
        self.attempted_neural_calls = 0
        self.faulted = False
        self.reference_mode = reference_mode
        self.last_reference = np.zeros(70, dtype=np.float32)

    def propose(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        angular_velocity_body: np.ndarray,
        command_body: np.ndarray,
        *,
        frame: int,
    ) -> np.ndarray:
        if self.reference_mode not in ("zero70", "zero79"):
            self.faulted = True
            raise ValueError("declared reference session requires explicit reference features")
        return self._propose(
            qpos,
            qvel,
            angular_velocity_body,
            command_body,
            np.zeros(self.reference_features, dtype=np.float32),
            frame=frame,
        )

    def propose_reference(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        angular_velocity_body: np.ndarray,
        command_body: np.ndarray,
        reference: np.ndarray,
        *,
        frame: int,
    ) -> np.ndarray:
        if self.reference_mode != "declared70":
            self.faulted = True
            raise ValueError("reference features cannot enter a zero70 retention session")
        return self._propose(
            qpos, qvel, angular_velocity_body, command_body, reference, frame=frame
        )

    def _propose(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        angular_velocity_body: np.ndarray,
        command_body: np.ndarray,
        reference: np.ndarray,
        *,
        frame: int,
    ) -> np.ndarray:
        if self.faulted:
            raise ValueError("learned prior fault latched; no implicit restart")
        try:
            if type(frame) is not int or frame != self.next_frame or not 0 <= frame < 300:
                raise ValueError("consecutive H300 actual retention observations required")
            if (
                not isinstance(reference, np.ndarray)
                or reference.shape != (self.reference_features,)
                or reference.dtype != np.float32
                or not np.isfinite(reference).all()
            ):
                raise ValueError("finite float32 features of the declared inference ABI required")
            obs, phase = observation98(
                qpos,
                qvel,
                angular_velocity_body,
                command_body,
                self.last_action,
                self.phase,
                self.params,
            )
            inputs = np.concatenate((obs, reference))
            self.attempted_neural_calls += 1
            with torch.no_grad():
                action = self.model(torch.from_numpy(inputs[None])).numpy()[0].copy()
            self.neural_calls += 1
            target = target29(action, self.params)
            self.last_observation = obs.copy()
            self.last_action = action
            self.last_reference = reference.copy()
            self.phase = phase
            self.next_frame += 1
            return target
        except Exception:
            self.faulted = True
            raise

    def propose_ball_aware(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        angular_velocity_body: np.ndarray,
        command_body: np.ndarray,
        reference70: np.ndarray,
        perception9: np.ndarray,
        *,
        frame: int,
    ) -> np.ndarray:
        if self.reference_mode != "declared79":
            self.faulted = True
            raise ValueError("explicit declared79 SIM session required for ball perception")
        for value, shape in ((reference70, (70,)), (perception9, (9,))):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                self.faulted = True
                raise ValueError(
                    "separate finite float32 reference70 and measured perception9 required"
                )
        return self._propose(
            qpos,
            qvel,
            angular_velocity_body,
            command_body,
            np.concatenate((reference70, perception9)),
            frame=frame,
        )
