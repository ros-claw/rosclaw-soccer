"""Task descriptions for Soccer Academy simulation examinations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosclaw.simforge.models import SimForgeTaskSpec


@dataclass(frozen=True)
class SoccerTaskProvider:
    """Describe football tasks without constructing MuJoCo or a robot runtime."""

    provider_id: str = "soccer.academy"
    task_ids: tuple[str, ...] = (
        "soccer.age04_regulation",
        "soccer.first_touch",
        "soccer.three_role_league",
    )

    def task_spec(self, task_id: str) -> SimForgeTaskSpec:
        if task_id not in self.task_ids:
            raise KeyError(f"unknown soccer task: {task_id}")
        from rosclaw.simforge.models import EvidenceRequirements, SimForgeTaskSpec

        if task_id == "soccer.age04_regulation":
            return SimForgeTaskSpec(
                task_id=task_id,
                suite_id="soccer.academy.age04",
                body_id="unitree.g1.sim",
                required_capabilities=("locomotion", "whole_body_contact"),
                discovery_backends=("mujoco",),
                evaluation_backends=("mujoco",),
                differential_backends=(),
                scenario_distribution_ref="soccer://age04/regulation-v1",
                success_spec=(
                    ("goal_plane_target_error_m.max", 0.10),
                    ("ball_retained_in_goal", True),
                ),
                safety_spec=(
                    ("post_kick_fall", False),
                    ("torque_limit_violation", False),
                ),
                candidate_allowed_paths=(
                    "/contact_actor",
                    "/phase_conditioned_residual",
                ),
                evidence_requirements=EvidenceRequirements(
                    physics_executed=True,
                    strict_replay=True,
                    artifact_hashes=True,
                    minimum_seeds=8,
                    holdout_required=True,
                ),
            )
        if task_id == "soccer.first_touch":
            return SimForgeTaskSpec(
                task_id=task_id,
                suite_id="soccer.academy.age05",
                body_id="unitree.g1.sim",
                required_capabilities=("locomotion", "whole_body_contact", "ball_tracking"),
                discovery_backends=("mujoco",),
                evaluation_backends=("mujoco",),
                differential_backends=(),
                scenario_distribution_ref="soccer://age05/first-touch-v1",
                success_spec=(
                    ("continuous_episode_sec.min", 30.0),
                    ("controlled_first_touch", True),
                ),
                safety_spec=(
                    ("fall_count.max", 0),
                    ("torque_limit_violation", False),
                ),
                candidate_allowed_paths=(
                    "/intercept_policy",
                    "/touch_policy",
                    "/recovery_policy",
                ),
                evidence_requirements=EvidenceRequirements(
                    physics_executed=True,
                    strict_replay=True,
                    artifact_hashes=True,
                    minimum_seeds=20,
                    holdout_required=True,
                ),
            )
        return SimForgeTaskSpec(
            task_id=task_id,
            suite_id="soccer.academy.three-role-league",
            body_id="unitree.g1.sim",
            required_capabilities=(
                "locomotion",
                "whole_body_contact",
                "ball_tracking",
                "multi_agent_coordination",
            ),
            discovery_backends=("mujoco",),
            evaluation_backends=("mujoco",),
            differential_backends=(),
            scenario_distribution_ref="soccer://league/pass-shot-save-v1",
            success_spec=(
                ("all_role_growth_gates", True),
                ("rolling_authenticity", True),
                ("counterfactual_credit_bound", True),
            ),
            safety_spec=(
                ("fall_count.max", 0),
                ("torque_limit_violation", False),
            ),
            candidate_allowed_paths=(
                "/roles/passer/policy",
                "/roles/shooter/policy",
                "/roles/goalkeeper/policy",
            ),
            evidence_requirements=EvidenceRequirements(
                physics_executed=True,
                strict_replay=True,
                artifact_hashes=True,
                minimum_seeds=8,
                holdout_required=True,
            ),
        )


SOCCER_TASK_PROVIDER = SoccerTaskProvider()

__all__ = ["SOCCER_TASK_PROVIDER", "SoccerTaskProvider"]
