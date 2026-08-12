"""Football-specific Growth adapters and candidate builders.

ROSClaw Core owns task-neutral evidence and safety contracts.  This package
owns the football semantics that populate those contracts.  Importing it has
no simulator, runtime, ROS, or hardware side effects.
"""

from rosclaw_soccer.growth.adapter import SOCCER_GROWTH_ADAPTER, SoccerGrowthAdapter
from rosclaw_soccer.growth.joint_policy_search import (
    JointPolicySearchConfig,
    JointPolicySearchDecision,
    MirroredRoleProbe,
    RolePolicySearchSpace,
    RolePolicySearchUpdate,
    RolePolicyVector,
    default_three_role_search_spaces,
    learn_joint_policy_generation,
)
from rosclaw_soccer.growth.role_learning import (
    JointGrowthDecision,
    JointGrowthGateConfig,
    JointGrowthRoundDecision,
    RoleEpisodeOutcome,
    RoleGrowthMetrics,
    RolePolicyBinding,
    SharedWorldTeamEpisode,
    SoccerRole,
    SoccerSide,
    evaluate_joint_growth,
    evaluate_joint_growth_round,
)

__all__ = [
    "JointGrowthDecision",
    "JointGrowthGateConfig",
    "JointGrowthRoundDecision",
    "JointPolicySearchConfig",
    "JointPolicySearchDecision",
    "MirroredRoleProbe",
    "RoleEpisodeOutcome",
    "RoleGrowthMetrics",
    "RolePolicySearchSpace",
    "RolePolicySearchUpdate",
    "RolePolicyBinding",
    "RolePolicyVector",
    "SOCCER_GROWTH_ADAPTER",
    "SharedWorldTeamEpisode",
    "SoccerGrowthAdapter",
    "SoccerRole",
    "SoccerSide",
    "evaluate_joint_growth",
    "evaluate_joint_growth_round",
    "default_three_role_search_spaces",
    "learn_joint_policy_generation",
]
