"""Source-bound learned body/ball dynamics predictions, never a controller."""

import copy
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth import context_prediction_mlp as reference

from rosclaw_soccer.rsi import body_ball_response_features
from rosclaw_soccer.rsi.body_ball_response_features import measured_body_ball_features
from rosclaw_soccer.rsi.body_response_features import _matrix
from rosclaw_soccer.sim.contracts import hash_bytes


class BodyBallResponseField:
    """38 velocity channels, explicit full29 control and 12/12/5 partitions.

    Fixed half nonlinear transition/half local matrix effect; no context or
    outcome chooses an expert. Predictions confer no physics qualification,
    action selection, runtime execution authority or hardware capability.
    """

    def __init__(
        self, baseline_model: Any, matrix_models: Any, *, implementation: str = "reference"
    ) -> None:
        if (
            implementation not in ("reference", "compiled")
            or type(matrix_models) is not list
            or len(matrix_models) != 3
        ):
            raise ValueError("explicit complete body-ball response representation required")
        self._models = copy.deepcopy([baseline_model, *matrix_models])
        for model, output in zip(self._models, (38, 456, 456, 190), strict=True):
            if reference.predict(model, np.zeros((1, 112))).shape != (1, output):
                raise ValueError("complete source-bound 112-feature body-ball models required")
        self._implementation = implementation
        self._compiled: Any = None
        if implementation == "compiled":
            from rosclaw.growth.compiled_context_prediction import CompiledContextPrediction

            self._compiled = [CompiledContextPrediction(m) for m in self._models]
        self._pins = {
            str(p): hash_bytes(p.read_bytes())
            for p in (
                Path(__file__),
                Path(body_ball_response_features.__file__),
                Path(reference.__file__),
            )
        }

    def _features(self, qpos: Any, qvel: Any, target: Any) -> np.ndarray[Any, Any]:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("body-ball response dependency source changed")
        features: np.ndarray[Any, Any] = measured_body_ball_features(qpos, qvel, target)
        return features

    def _predict(self, index: int, features: Any) -> np.ndarray[Any, Any]:
        result: np.ndarray[Any, Any] = (
            reference.predict(self._models[index], features)
            if self._compiled is None
            else self._compiled[index].predict(features)
        )
        return result

    def predict_next_velocity(
        self, *, qpos: Any, qvel: Any, nominal_target: Any
    ) -> np.ndarray[Any, Any]:
        features = self._features(qpos, qvel, nominal_target)
        v = _matrix(qvel, 41)
        result: np.ndarray[Any, Any] = v[:, :38] + self._predict(0, features)
        if not np.isfinite(result).all():
            raise ValueError("finite body-ball next velocity prediction required")
        result.flags.writeable = False
        return result

    def predict_effect(
        self, *, qpos: Any, qvel: Any, nominal_target: Any, target_increment: Any
    ) -> np.ndarray[Any, Any]:
        target, delta = _matrix(nominal_target, 29), _matrix(target_increment, 29)
        if len(target) != len(delta) or np.max(np.abs(delta)) > 0.02:
            raise ValueError("aligned bounded local full29 increment required")
        base = self._features(qpos, qvel, target)
        altered = self._features(qpos, qvel, target + delta)
        transitions = self._predict(0, np.concatenate((base, altered), axis=0))
        matrix = np.concatenate(
            [self._predict(i, base).reshape(-1, 38, n) for i, n in enumerate((12, 12, 5), 1)],
            axis=2,
        )
        result: np.ndarray[Any, Any] = 0.5 * (
            transitions[len(base) :] - transitions[: len(base)]
        ) + 0.5 * np.einsum("nij,nj->ni", matrix, delta)
        if not np.isfinite(result).all():
            raise ValueError("finite body-ball local effect prediction required")
        result.flags.writeable = False
        return result

    def contract(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.body_ball_response_field.v1",
            model_hashes=[m["model_hash"] for m in self._models],
            source_pins={Path(p).name: h for p, h in self._pins.items()},
            implementation=self._implementation,
            input_dimensions=112,
            effect_dimensions=38,
            target_increment_dimensions=29,
            matrix_partition_action_widths=[12, 12, 5],
            maximum_target_increment_rad=0.02,
            fixed_expert_weights=[0.5, 0.5],
            ball_velocity_reference="CURRENT_CANONICAL_WORLD_LINEAR_AXES",
            outcome_conditioned_selection=False,
            global_validity_guaranteed=False,
            prediction_only=True,
            motor_policy=False,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )
