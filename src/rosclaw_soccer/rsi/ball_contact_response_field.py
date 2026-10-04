"""Nonlinear action-conditioned ball response, with exact zero-action effect."""

import copy
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth import context_prediction_mlp as reference

from rosclaw_soccer.rsi import body_ball_response_features
from rosclaw_soccer.rsi.body_ball_response_features import measured_body_ball_features
from rosclaw_soccer.rsi.body_response_features import _matrix
from rosclaw_soccer.sim.contracts import hash_bytes


class BallContactResponseField:
    """Predict ball linear velocity changes from bounded full29 target changes.

    No local Jacobian linearity assumption at contact. A paired zero-action
    prediction removes learned offsets, without applying a force to the ball.
    This object predicts only; it neither chooses nor executes any action.
    """

    def __init__(
        self, model: Any, *, implementation: str = "reference", contact_geometry: bool = False
    ) -> None:
        if implementation not in ("reference", "compiled") or type(contact_geometry) is not bool:
            raise ValueError("explicit ball response implementation required")
        self._model = copy.deepcopy(model)
        self._contact_geometry = contact_geometry
        self._input_dimensions = 197 if contact_geometry else 141
        if reference.predict(self._model, np.zeros((1, self._input_dimensions))).shape != (1, 3):
            raise ValueError("source-bound explicit three-channel ball representation required")
        self._compiled: Any = None
        if implementation == "compiled":
            from rosclaw.growth.compiled_context_prediction import CompiledContextPrediction

            self._compiled = CompiledContextPrediction(self._model)
        self._implementation = implementation
        self._pins = {
            str(p): hash_bytes(p.read_bytes())
            for p in (
                Path(__file__),
                Path(body_ball_response_features.__file__),
                Path(reference.__file__),
            )
        }

    def predict_effect(
        self,
        *,
        qpos: Any,
        qvel: Any,
        nominal_target: Any,
        target_increment: Any,
        foot_contact_features: Any = None,
    ) -> np.ndarray[Any, Any]:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("ball response dependency source changed")
        features = measured_body_ball_features(qpos, qvel, nominal_target)
        if self._contact_geometry:
            geometry = _matrix(foot_contact_features, 56)
            if len(geometry) != len(features):
                raise ValueError("aligned current measured foot collision features required")
            features = np.concatenate((features, geometry), axis=1)
        elif foot_contact_features is not None:
            raise ValueError("geometry features require an explicitly geometry-trained model")
        delta = _matrix(target_increment, 29)
        if len(delta) != len(features) or np.max(np.abs(delta)) > 0.02:
            raise ValueError("aligned bounded local full29 increment required")
        inputs = np.concatenate(
            (
                np.concatenate((features, delta), axis=1),
                np.concatenate((features, np.zeros_like(delta)), axis=1),
            ),
            axis=0,
        )
        output = (
            reference.predict(self._model, inputs)
            if self._compiled is None
            else self._compiled.predict(inputs)
        )
        result: np.ndarray[Any, Any] = output[: len(features)] - output[len(features) :]
        if not np.isfinite(result).all():
            raise ValueError("finite ball response required")
        result.flags.writeable = False
        return result

    def contract(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.ball_contact_response_field.v1",
            model_hash=self._model["model_hash"],
            source_pins={Path(p).name: h for p, h in self._pins.items()},
            implementation=self._implementation,
            input_dimensions=self._input_dimensions,
            current_collision_point_features=56 if self._contact_geometry else 0,
            output_dimensions=3,
            target_increment_dimensions=29,
            maximum_target_increment_rad=0.02,
            zero_action_effect_exact=True,
            contact_jacobian_linearity_assumed=False,
            future_contact_label_input=False,
            ball_velocity_reference="CURRENT_CANONICAL_WORLD_LINEAR_AXES",
            prediction_only=True,
            motor_policy=False,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )
