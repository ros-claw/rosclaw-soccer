"""Frozen measured-foot, SIM_ONLY navigation chooser; no actuator or world handle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.team_adaptive_intercept_navigation import TeamAdaptiveInterceptNavigation
from rosclaw_soccer.rsi.team_contextual_nav_policy import measured_entry_features
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


def _forward_network(
    feature: np.ndarray[Any, Any], network: dict[str, Any]
) -> np.ndarray[Any, Any]:
    value = feature
    if feature.shape not in {(12,), (16,)}:
        raise ValueError("invalid bounded measured-foot feature dimension")
    shapes = ((24, feature.shape[0]), (24, 24), (18, 24))
    if len(network.get("layers", ())) != 3:
        raise ValueError("invalid bounded chooser layers")
    for index, (layer, shape) in enumerate(zip(network["layers"], shapes, strict=True)):
        weight = np.asarray(layer["weight"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        if (
            weight.shape != shape
            or bias.shape != (shape[0],)
            or not np.all(np.isfinite(weight))
            or not np.all(np.isfinite(bias))
            or np.max(np.abs(weight)) > 100
            or np.max(np.abs(bias)) > 100
        ):
            raise ValueError("non-finite or unbounded chooser weights")
        value = weight @ value + bias
        if index < 2:
            value = np.tanh(value)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


class TeamFootVelocityChooser:
    """Select once at frame 30 from measured bilateral foot state and sealed weights."""

    activation_ceiling = "SIM_ONLY"
    needs_effector_velocities = True

    def __init__(
        self,
        *,
        agent_id: str,
        foundation_hash: str,
        foundation_config_hash: str,
        model_path: Path,
    ) -> None:
        if not model_path.is_absolute() or model_path.stat().st_size > 2_000_000:
            raise ValueError("bounded absolute SIM_ONLY foot model required")
        model = json.loads(model_path.read_text(encoding="utf-8"))
        commitment = model.pop("model_hash", None)
        schema = model.get("schema")
        feature_count = 12 if schema == "rsi_team_foot_velocity_chooser_model_v33" else 16
        if (
            schema
            not in {
                "rsi_team_foot_velocity_chooser_model_v33",
                "rsi_team_large_context_chooser_model_v39",
            }
            or model.get("activation_ceiling") != "SIM_ONLY"
            or model.get("promotion_authorized") is not False
            or commitment != hash_json(model)
            or model.get("feature_count") != feature_count
            or schema == "rsi_team_large_context_chooser_model_v39"
            and model.get("feature_set") != "relative_feet_16"
            or model.get("safety_threshold") != 0.65
            or model.get("safety_quantile") != 0.2
            or len(model.get("arm_names", ())) != 6
            or len(model.get("networks", ())) != 3
            or set(model.get("arm_names", ())) != set(model.get("arm_parameters", {}))
        ):
            raise ValueError("sealed unpromoted measured-foot model required")
        self.agent_id = agent_id
        self.foundation_hash = foundation_hash
        self.foundation_config_hash = foundation_config_hash
        self.model_hash = str(commitment)
        self.model = model
        self.contract_hash = str(
            hash_json(
                {
                    "schema": (
                        "rsi_team_large_context_navigation_v39"
                        if schema == "rsi_team_large_context_chooser_model_v39"
                        else "rsi_team_foot_velocity_navigation_v33"
                    ),
                    "agent_id": agent_id,
                    "foundation_hash": foundation_hash,
                    "foundation_config_hash": foundation_config_hash,
                    "model_hash": commitment,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )
        self.selected_arm: str | None = None
        self.selected_scores: list[float] = []
        self._selected = False
        self._delegate: (
            TeamInterceptNavigation
            | TeamPhaseInterceptNavigation
            | TeamAdaptiveInterceptNavigation
            | None
        ) = None

    def propose(self, observation: NavigationObservation) -> NavigationDelta:
        if observation.agent_id != self.agent_id:
            raise ValueError("foreign player's navigation observation")
        if observation.frame == 30:
            if self._selected:
                raise ValueError("repeated measured-foot entry decision")
            feet = dict((name, (x, y, z)) for name, x, y, z in observation.effector_positions)
            velocity = dict((name, (x, y, z)) for name, x, y, z in observation.effector_velocities)
            if set(feet) != {"left_foot", "right_foot"} or set(velocity) != set(feet):
                raise ValueError("bilateral measured foot position and velocity required")
            foot_values = np.asarray(
                (
                    *feet["left_foot"],
                    *feet["right_foot"],
                    *velocity["left_foot"],
                    *velocity["right_foot"],
                ),
                dtype=np.float64,
            )
            values = (
                np.concatenate(
                    (
                        np.asarray(
                            measured_entry_features(
                                ball=observation.ball_position,
                                ball_vx=observation.ball_velocity[0],
                                feet=(feet["left_foot"], feet["right_foot"]),
                            ),
                            dtype=np.float64,
                        ),
                        foot_values,
                    )
                )
                if self.model["schema"] == "rsi_team_large_context_chooser_model_v39"
                else foot_values
            )
            mean = np.asarray(self.model["mean"], dtype=np.float64)
            scale = np.asarray(self.model["scale"], dtype=np.float64)
            feature_count = self.model["feature_count"]
            if (
                values.shape != (feature_count,)
                or mean.shape != (feature_count,)
                or scale.shape != (feature_count,)
                or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(scale))
                or np.any(scale < 0.03)
            ):
                raise ValueError("invalid causal measured-foot feature normalization")
            feature = (values - mean) / scale
            predictions = np.stack(
                [
                    _forward_network(feature, network).reshape(6, 3)
                    for network in self.model["networks"]
                ]
            )
            scores = predictions.mean(axis=0)
            scores[:, 0] = np.quantile(predictions[:, :, 0], 0.2, axis=0)
            eligible = scores[:, 0] >= self.model["safety_threshold"]
            value = np.where(eligible, scores[:, 2] + 0.25 * scores[:, 1], -np.inf)
            self._selected = True
            self.selected_scores = [float(item) for item in scores[:, 2]]
            if eligible.any():
                self.selected_arm = self.model["arm_names"][int(np.argmax(value))]
                arm = self.model["arm_parameters"][self.selected_arm]
                if arm["foot_selection"] == "adaptive":
                    self._delegate = TeamAdaptiveInterceptNavigation(
                        agent_id=self.agent_id,
                        foundation_hash=self.foundation_hash,
                        foundation_config_hash=self.foundation_config_hash,
                        **self.model["adaptive_model_parameters"],
                    )
                else:
                    nav_type = (
                        TeamPhaseInterceptNavigation
                        if arm["foot_selection"] == "phase"
                        else TeamInterceptNavigation
                    )
                    self._delegate = nav_type(
                        agent_id=self.agent_id,
                        foundation_hash=self.foundation_hash,
                        foundation_config_hash=self.foundation_config_hash,
                        forward_gain=float(arm["forward_gain"]),
                        lateral_gain=float(arm["lateral_gain"]),
                    )
        if observation.frame > 30 and not self._selected:
            raise ValueError("missing measured-foot entry decision")
        if self._delegate is None:
            return NavigationDelta(
                observation.agent_id, observation.frame, observation.time_sec, (0.0, 0.0, 0.0)
            )
        return self._delegate.propose(observation)
