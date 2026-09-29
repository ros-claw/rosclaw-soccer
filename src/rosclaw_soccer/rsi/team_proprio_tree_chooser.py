"""Sealed NumPy-only SIM_ONLY navigation selection from measured G1 proprioception."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.team_adaptive_intercept_navigation import TeamAdaptiveInterceptNavigation
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


def measured_full_proprio_feature(observation: NavigationObservation) -> np.ndarray[Any, Any]:
    """Match the audited 89-value parent action trace at decision frame 30."""
    feet = dict((name, (x, y, z)) for name, x, y, z in observation.effector_positions)
    velocity = dict((name, (x, y, z)) for name, x, y, z in observation.effector_velocities)
    if (
        observation.frame != 30
        or set(feet) != {"left_foot", "right_foot"}
        or set(velocity) != set(feet)
        or observation.body_angular_velocity is None
        or len(observation.joint_positions_rad) != 29
        or len(observation.joint_velocities_radps) != 29
    ):
        raise ValueError("complete same-frame G1 body, feet and joint state required")
    feature = np.asarray(
        (
            *observation.body_pose[:3],
            *observation.body_velocity,
            *observation.ball_position,
            *observation.ball_velocity,
            *feet["left_foot"],
            *feet["right_foot"],
            *velocity["left_foot"],
            *velocity["right_foot"],
            *observation.body_pose[3:7],
            *observation.body_angular_velocity,
            *observation.joint_positions_rad,
            *observation.joint_velocities_radps,
        ),
        dtype=np.float64,
    )
    if feature.shape != (89,) or not np.isfinite(feature).all():
        raise ValueError("invalid measured full-proprio feature")
    return feature


class TeamProprioTreeChooser:
    """Choose one frozen navigation arm; no world, driver or actuator handle."""

    activation_ceiling = "SIM_ONLY"
    needs_effector_velocities = True
    needs_full_proprioception = True

    def __init__(
        self,
        *,
        agent_id: str,
        foundation_hash: str,
        foundation_config_hash: str,
        model_path: Path,
    ) -> None:
        if not model_path.is_absolute() or model_path.stat().st_size > 500_000:
            raise ValueError("bounded absolute SIM_ONLY tree manifest required")
        model = json.loads(model_path.read_text(encoding="utf-8"))
        commitment = model.pop("model_hash", None)
        if (
            model.get("schema") != "rsi_team_proprio_tree_chooser_model_v47"
            or model.get("activation_ceiling") != "SIM_ONLY"
            or model.get("promotion_authorized") is not False
            or commitment != hash_json(model)
            or model.get("feature_count") != 89
            or model.get("feature_set") != "full_proprio_89"
            or model.get("forest_file") != "forest.npz"
            or model.get("safety_threshold") != 0.75
            or model.get("safety_quantile") != 0.2
            or model.get("estimators_per_forest") != 120
            or model.get("ensemble_seeds") != [11, 19]
            or len(model.get("arm_names", ())) != 6
            or set(model["arm_names"]) != set(model.get("arm_parameters", {}))
        ):
            raise ValueError("unsealed or unpromoted G1 proprio tree model")
        forest_path = model_path.parent / "forest.npz"
        if (
            forest_path.stat().st_size > 30_000_000
            or hash_bytes(forest_path.read_bytes()) != model["forest_hash"]
        ):
            raise ValueError("proprio tree weights missing or changed")
        with np.load(forest_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "tree_offsets",
                "forest_offsets",
                "feature",
                "threshold",
                "left",
                "right",
                "positive_probability",
            }:
                raise ValueError("unexpected tree weight fields")
            self.tree_offsets = np.asarray(archive["tree_offsets"], dtype=np.int64)
            self.forest_offsets = np.asarray(archive["forest_offsets"], dtype=np.int64)
            self.feature = np.asarray(archive["feature"], dtype=np.int16)
            self.threshold = np.asarray(archive["threshold"], dtype=np.float64)
            self.left = np.asarray(archive["left"], dtype=np.int64)
            self.right = np.asarray(archive["right"], dtype=np.int64)
            self.probability = np.asarray(archive["positive_probability"], dtype=np.float64)
        node_count = len(self.feature)
        if (
            self.tree_offsets.shape != (721,)
            or self.forest_offsets.shape != (7,)
            or self.threshold.shape != (node_count,)
            or self.left.shape != (node_count,)
            or self.right.shape != (node_count,)
            or self.probability.shape != (node_count, 6)
            or not 720 <= node_count <= 250_000
            or self.tree_offsets[0] != 0
            or self.tree_offsets[-1] != node_count
            or np.any(np.diff(self.tree_offsets) <= 0)
            or not np.array_equal(self.forest_offsets, np.arange(7) * 120)
            or not np.isfinite(self.threshold).all()
            or not np.isfinite(self.probability).all()
            or np.any((self.probability < 0) | (self.probability > 1))
        ):
            raise ValueError("invalid bounded tree shape or probability")
        for tree in range(720):
            start, end = self.tree_offsets[tree : tree + 2]
            feature = self.feature[start:end]
            left, right = self.left[start:end], self.right[start:end]
            leaves = feature == -2
            if (
                np.any(~leaves & ((feature < 0) | (feature >= 89)))
                or np.any(leaves & ((left != -1) | (right != -1)))
                or np.any(
                    ~leaves & ((left <= np.arange(start, end)) | (right <= np.arange(start, end)))
                )
                or np.any(~leaves & ((left >= end) | (right >= end)))
            ):
                raise ValueError("invalid or cyclic tree child indices")
        for value in (
            self.tree_offsets,
            self.forest_offsets,
            self.feature,
            self.threshold,
            self.left,
            self.right,
            self.probability,
        ):
            value.setflags(write=False)
        self.agent_id = agent_id
        self.foundation_hash = foundation_hash
        self.foundation_config_hash = foundation_config_hash
        self.model_hash = str(commitment)
        self.model = model
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rsi_team_proprio_tree_navigation_v47",
                    "agent_id": agent_id,
                    "foundation_hash": foundation_hash,
                    "foundation_config_hash": foundation_config_hash,
                    "model_hash": commitment,
                    "activation_ceiling": self.activation_ceiling,
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

    def predict_scores(self, feature: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if feature.shape != (89,) or not np.isfinite(feature).all():
            raise ValueError("finite 89-value measured body input required")
        forest_scores = np.zeros((6, 6), dtype=np.float64)
        for forest in range(6):
            for tree in range(
                int(self.forest_offsets[forest]), int(self.forest_offsets[forest + 1])
            ):
                node = int(self.tree_offsets[tree])
                for _ in range(11):
                    field = int(self.feature[node])
                    if field == -2:
                        forest_scores[forest] += self.probability[node]
                        break
                    node = int(
                        self.left[node]
                        if feature[field] <= self.threshold[node]
                        else self.right[node]
                    )
                else:
                    raise ValueError("tree exceeded frozen maximum depth")
            forest_scores[forest] /= 120
        return np.stack(
            (
                np.quantile(forest_scores[:2], 0.2, axis=0),
                forest_scores[2:4].mean(axis=0),
                forest_scores[4:6].mean(axis=0),
            ),
            axis=1,
        )

    def choose_feature(self, feature: np.ndarray[Any, Any]) -> str | None:
        scores = self.predict_scores(feature)
        eligible = scores[:, 0] >= self.model["safety_threshold"]
        value = np.where(eligible, scores[:, 2] + 0.25 * scores[:, 1], -np.inf)
        return self.model["arm_names"][int(np.argmax(value))] if eligible.any() else None

    def propose(self, observation: NavigationObservation) -> NavigationDelta:
        if observation.agent_id != self.agent_id:
            raise ValueError("foreign player's navigation observation")
        if observation.frame == 30:
            if self._selected:
                raise ValueError("repeated measured-body entry decision")
            feature = measured_full_proprio_feature(observation)
            scores = self.predict_scores(feature)
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
            raise ValueError("missing measured-body entry decision")
        if self._delegate is None:
            return NavigationDelta(
                observation.agent_id, observation.frame, observation.time_sec, (0.0, 0.0, 0.0)
            )
        return self._delegate.propose(observation)
