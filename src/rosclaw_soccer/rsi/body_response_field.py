"""Fixed dual-representation local response predictions, never a controller."""

import copy
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth import context_prediction_mlp as reference

from rosclaw_soccer.rsi import body_response_features
from rosclaw_soccer.rsi.body_response_features import (
    _matrix,
    local_velocity_response,
    measured_response_features,
)
from rosclaw_soccer.sim.contracts import hash_bytes


class BodyResponseField:
    """35D measured-body effect from bounded leg or full-body increments.

    The fixed 0.5/0.5 blend never selects an expert by a future outcome or a
    scenario label. Physics qualification belongs to external evidence. This
    object predicts effects only; it cannot choose or execute a motor action.
    Full-body matrices use explicit 12/12/5 column partitions so every neural
    predictor stays inside Core's existing 512-output validation limit.
    """

    def __init__(
        self,
        next_velocity_model: dict[str, Any],
        local_matrix_model: dict[str, Any] | list[dict[str, Any]],
        *,
        implementation: str = "reference",
        action_dimensions: int = 12,
    ) -> None:
        if implementation not in ("reference", "compiled"):
            raise ValueError("explicit reference or compiled prediction implementation required")
        if type(action_dimensions) is not int or action_dimensions not in (12, 29):
            raise ValueError("explicit canonical leg or full-body response dimensions required")
        self._action_dimensions = action_dimensions
        self._old = copy.deepcopy(next_velocity_model)
        self._new = copy.deepcopy(local_matrix_model)
        self._partition_widths = (12,) if action_dimensions == 12 else (12, 12, 5)
        if action_dimensions == 12 and type(self._new) is dict:
            self._matrix_models = [self._new]
        elif action_dimensions == 29 and type(self._new) is list and len(self._new) == 3:
            self._matrix_models = self._new
        else:
            raise ValueError(
                "complete twelve-joint model or three full-body matrix partitions required"
            )
        checks = [(self._old, 1064, 35)] + [
            (m, 103, 35 * n)
            for m, n in zip(self._matrix_models, self._partition_widths, strict=True)
        ]
        for model, width, output in checks:
            if reference.predict(model, np.zeros((1, width))).shape != (1, output):
                raise ValueError("complete source-bound body response model dimensions required")
        self._implementation = implementation
        self._compiled: Any = None
        if implementation == "compiled":
            # Optional experimental Core API. Reference remains available in
            # installations without this compilation surface.
            from rosclaw.growth.compiled_context_prediction import CompiledContextPrediction

            self._compiled = (
                CompiledContextPrediction(self._old),
                *(CompiledContextPrediction(m) for m in self._matrix_models),
            )
        self._pins = {
            str(p): hash_bytes(p.read_bytes())
            for p in (
                Path(__file__),
                Path(body_response_features.__file__),
                Path(reference.__file__),
            )
        }

    def predict_effect(
        self,
        *,
        qpos: Any,
        qvel: Any,
        nominal_target: Any,
        relative_ball: Any,
        foundation_input: Any,
        target_increment: Any,
    ) -> np.ndarray[Any, Any]:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("body response field dependency source changed")
        q, v, target, ball, foundation, delta = (
            _matrix(value, width)
            for value, width in (
                (qpos, 43),
                (qvel, 41),
                (nominal_target, 29),
                (relative_ball, 6),
                (foundation_input, 994),
                (target_increment, self._action_dimensions),
            )
        )
        if not len(q) == len(v) == len(target) == len(ball) == len(foundation) == len(delta):
            raise ValueError("aligned current body response rows required")
        if np.max(np.abs(delta)) > 0.02:
            raise ValueError("bounded local target increment required")
        features = measured_response_features(q, v, target, ball)
        base = np.concatenate((foundation, ball, target, v[:, :35]), axis=1)
        candidate = base.copy()
        candidate[:, 1000 : 1000 + self._action_dimensions] += delta
        combined = np.concatenate((base, candidate), axis=0)
        if self._compiled is None:
            old_result = reference.predict(self._old, combined)
            predictions = [reference.predict(m, features) for m in self._matrix_models]
        else:
            old_result = self._compiled[0].predict(combined)
            predictions = [m.predict(features) for m in self._compiled[1:]]
        matrix = np.concatenate(
            [
                p.reshape(-1, 35, n)
                for p, n in zip(predictions, self._partition_widths, strict=True)
            ],
            axis=2,
        ).reshape(-1, 35 * self._action_dimensions)
        old_effect = old_result[len(q) :] - old_result[: len(q)]
        new_effect = local_velocity_response(
            matrix, delta, action_dimensions=self._action_dimensions
        )
        result: np.ndarray[Any, Any] = 0.5 * old_effect + 0.5 * new_effect
        if not np.isfinite(result).all():
            raise ValueError("finite local response prediction required")
        result.flags.writeable = False
        return result

    def contract(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.body_response_field.v1",
            next_velocity_model_hash=self._old["model_hash"],
            local_matrix_model_hash=(
                self._matrix_models[0]["model_hash"]
                if self._action_dimensions == 12
                else [m["model_hash"] for m in self._matrix_models]
            ),
            matrix_partition_action_widths=list(self._partition_widths),
            implementation=self._implementation,
            source_pins=dict(self._pins),
            fixed_expert_weights=[0.5, 0.5],
            target_increment_dimensions=self._action_dimensions,
            effect_dimensions=35,
            maximum_target_increment_rad=0.02,
            outcome_conditioned_expert_selection=False,
            global_validity_guaranteed=False,
            physics_qualified_by_this_function=False,
            prediction_only=True,
            activation_ceiling="SIM_ONLY",
            motor_policy=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
