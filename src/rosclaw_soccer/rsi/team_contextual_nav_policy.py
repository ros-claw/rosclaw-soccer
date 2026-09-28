"""Evidence-bound, causal SIM_ONLY action selection for a shared-world G1 player."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


def measured_entry_features(
    *,
    ball: tuple[float, float, float],
    ball_vx: float,
    feet: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    values = np.asarray((*ball, ball_vx, *feet[0], *feet[1]), dtype=np.float64)
    if values.shape != (10,) or not np.all(np.isfinite(values)):
        raise ValueError("finite same-frame ball and bilateral foot measurement required")
    if feet[0][2] - feet[1][2] >= 0.02:
        side = 0
    elif feet[1][2] - feet[0][2] >= 0.02:
        side = 1
    else:
        side = int(np.argmin(np.abs(np.asarray((feet[0][1], feet[1][1])) - ball[1])))
    selected = feet[side]
    return (
        ball[0] - selected[0],
        ball[1] - selected[1],
        selected[2] - feet[1 - side][2],
        ball_vx,
    )


def select_evidence_arm(
    *,
    feature: tuple[float, float, float, float],
    model: dict[str, Any],
) -> tuple[str | None, list[int]]:
    x = np.asarray(feature, dtype=np.float64)
    train = np.asarray(model["training_features"], dtype=np.float64)
    if x.shape != (4,) or train.shape != (24, 4) or not np.all(np.isfinite(x)):
        raise ValueError("bounded four-feature model required")
    scale = np.maximum(np.std(train, axis=0), 0.05)
    distance = np.linalg.norm((train - x) / scale, axis=1)
    nearest = np.argsort(distance, kind="stable")[:3]
    weights = 1.0 / np.maximum(distance[nearest], 0.1)
    weights /= weights.sum()
    estimates: list[tuple[float, str]] = []
    for name in model["arm_names"]:
        outcomes = model["training_outcomes"][name]
        rewards = []
        safe = True
        for index in nearest:
            row = outcomes[int(index)]
            safe &= row["safe"] is True
            rewards.append(
                -5.0
                if not row["safe"]
                else 2.0
                if row["useful_pass"]
                else 1.0
                if row["foot_contact"]
                else 0.0
            )
        value = float(weights @ np.asarray(rewards))
        if safe and value > 0.0:
            estimates.append((value, name))
    selected = sorted(estimates, key=lambda item: (-item[0], item[1]))
    return (selected[0][1] if selected else None, [int(index) for index in nearest])


class TeamContextualNavigationMemory:
    """One player's frozen bounded navigation; no simulator or actuator handle."""

    activation_ceiling = "SIM_ONLY"

    def __init__(
        self,
        *,
        agent_id: str,
        foundation_hash: str,
        foundation_config_hash: str,
        model_path: Path,
    ) -> None:
        if not model_path.is_absolute() or model_path.stat().st_size > 2_000_000:
            raise ValueError("bounded absolute SIM_ONLY model required")
        model = json.loads(model_path.read_text(encoding="utf-8"))
        commitment = model.pop("model_hash", None)
        if (
            model.get("schema") != "rsi_team_contextual_navigation_memory_v25"
            or model.get("activation_ceiling") != "SIM_ONLY"
            or model.get("promotion_authorized") is not False
            or commitment != hash_json(model)
            or len(model.get("arm_names", [])) != 10
            or len(model.get("training_features", [])) != 24
            or any(len(rows) != 24 for rows in model.get("training_outcomes", {}).values())
        ):
            raise ValueError("sealed unpromoted SIM_ONLY navigation memory required")
        self.agent_id = agent_id
        self.foundation_hash = foundation_hash
        self.foundation_config_hash = foundation_config_hash
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rsi_team_contextual_navigation_policy_v25",
                    "agent_id": agent_id,
                    "foundation_hash": foundation_hash,
                    "foundation_config_hash": foundation_config_hash,
                    "model_hash": commitment,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )
        self.model = model
        self.model_hash = str(commitment)
        self.selected_arm: str | None = None
        self.neighbor_indices: list[int] = []
        self._selected = False
        self._delegate: TeamInterceptNavigation | TeamPhaseInterceptNavigation | None = None

    def propose(self, observation: NavigationObservation) -> NavigationDelta:
        if observation.agent_id != self.agent_id:
            raise ValueError("foreign player's navigation observation")
        if observation.frame == 30:
            if self._selected:
                raise ValueError("entry decision repeated")
            feet = dict((name, (x, y, z)) for name, x, y, z in observation.effector_positions)
            if set(feet) != {"left_foot", "right_foot"}:
                raise ValueError("both measured feet required for causal decision")
            feature = measured_entry_features(
                ball=observation.ball_position,
                ball_vx=observation.ball_velocity[0],
                feet=(feet["left_foot"], feet["right_foot"]),
            )
            self.selected_arm, self.neighbor_indices = select_evidence_arm(
                feature=feature, model=self.model
            )
            self._selected = True
            if self.selected_arm is not None:
                arm = self.model["arm_parameters"][self.selected_arm]
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
            raise ValueError("entry decision missing")
        if self._delegate is None:
            return NavigationDelta(
                self.agent_id, observation.frame, observation.time_sec, (0.0, 0.0, 0.0)
            )
        return self._delegate.propose(observation)
