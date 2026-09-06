"""Football worlds, fields, goals, nets, and regulation geometry."""

from rosclaw_soccer.world.field import (
    G1CompliantGoalNetState,
    G1TrainingGoalSpec,
    apply_g1_compliant_goal_net_force,
    build_g1_coupled_stadium_model,
    build_g1_four_player_two_ball_stadium_model,
    build_g1_stadium_model,
    build_g1_three_player_stadium_model,
    g1_ball_inside_goal_mouth,
    g1_goal_net_contact_plane_x,
    g1_stadium_scene_hash,
)
from rosclaw_soccer.world.multi_player import (
    G1PitchPlayerSpec,
    build_g1_multi_player_stadium_model,
)

__all__ = [
    "G1CompliantGoalNetState",
    "G1PitchPlayerSpec",
    "G1TrainingGoalSpec",
    "apply_g1_compliant_goal_net_force",
    "build_g1_coupled_stadium_model",
    "build_g1_four_player_two_ball_stadium_model",
    "build_g1_multi_player_stadium_model",
    "build_g1_stadium_model",
    "build_g1_three_player_stadium_model",
    "g1_ball_inside_goal_mouth",
    "g1_goal_net_contact_plane_x",
    "g1_stadium_scene_hash",
]
