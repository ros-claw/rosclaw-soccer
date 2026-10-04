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
    """35D measured-body effect from a bounded 12D target increment.

    The fixed 0.5/0.5 blend never selects an expert by a future outcome or a
    scenario label. Physics qualification belongs to external evidence. This
    object predicts effects only; it cannot choose or execute a motor action.
    """

    def __init__(
        self,
        next_velocity_model: dict[str, Any],
        local_matrix_model: dict[str, Any],
        *,
        implementation: str = "reference",
    ) -> None:
        if implementation not in ("reference", "compiled"):
            raise ValueError("explicit reference or compiled prediction implementation required")
        self._old = copy.deepcopy(next_velocity_model)
        self._new = copy.deepcopy(local_matrix_model)
        for model, width, output in ((self._old, 1064, 35), (self._new, 103, 420)):
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
                CompiledContextPrediction(self._new),
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
                (target_increment, 12),
            )
        )
        if not len(q) == len(v) == len(target) == len(ball) == len(foundation) == len(delta):
            raise ValueError("aligned current body response rows required")
        if np.max(np.abs(delta)) > 0.02:
            raise ValueError("bounded local target increment required")
        features = measured_response_features(q, v, target, ball)
        base = np.concatenate((foundation, ball, target, v[:, :35]), axis=1)
        candidate = base.copy()
        candidate[:, 1000:1012] += delta
        combined = np.concatenate((base, candidate), axis=0)
        if self._compiled is None:
            old_result = reference.predict(self._old, combined)
            matrix = reference.predict(self._new, features)
        else:
            old_result = self._compiled[0].predict(combined)
            matrix = self._compiled[1].predict(features)
        old_effect = old_result[len(q) :] - old_result[: len(q)]
        new_effect = local_velocity_response(matrix, delta)
        result: np.ndarray[Any, Any] = 0.5 * old_effect + 0.5 * new_effect
        if not np.isfinite(result).all():
            raise ValueError("finite local response prediction required")
        result.flags.writeable = False
        return result

    def contract(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.body_response_field.v1",
            next_velocity_model_hash=self._old["model_hash"],
            local_matrix_model_hash=self._new["model_hash"],
            implementation=self._implementation,
            source_pins=dict(self._pins),
            fixed_expert_weights=[0.5, 0.5],
            target_increment_dimensions=12,
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
