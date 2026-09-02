"""Evidence-downstream media generation."""

from rosclaw_soccer.media.aerial_curriculum_video import (
    render_aerial_curriculum_video,
    validate_aerial_curriculum_video_manifest,
)
from rosclaw_soccer.media.alternating_growth_video import (
    render_alternating_growth_video,
    validate_alternating_growth_video_manifest,
)
from rosclaw_soccer.media.dynamic_aerial_lunge_video import (
    render_dynamic_aerial_lunge_video,
    validate_dynamic_aerial_lunge_video_manifest,
)
from rosclaw_soccer.media.dynamic_takeoff_video import (
    render_dynamic_takeoff_video,
    validate_dynamic_takeoff_video_manifest,
)
from rosclaw_soccer.media.free_kick_video import (
    G1FreeKickVideoResult,
    render_g1_free_kick_showcase_video,
)
from rosclaw_soccer.media.full_body_tactical_video import (
    render_full_body_tactical_video,
    validate_full_body_tactical_video_manifest,
)
from rosclaw_soccer.media.goalkeeper_showcase_video import (
    render_collision_faithful_goalkeeper_video,
    render_goalkeeper_showcase_video,
    validate_collision_faithful_goalkeeper_manifest,
    validate_goalkeeper_showcase_manifest,
)
from rosclaw_soccer.media.goalkeeper_v2_video import (
    GoalkeeperV2DevelopmentVideoResult,
    render_goalkeeper_v2_development_video,
)
from rosclaw_soccer.media.physics_goalkeeper_video import (
    PhysicsGoalkeeperVideoResult,
    render_physics_goalkeeper_champion_video,
)
from rosclaw_soccer.media.regulation_dead_corner_video import (
    render_regulation_dead_corner_video,
    validate_regulation_dead_corner_video_manifest,
)
from rosclaw_soccer.media.rolling_comparison_video import (
    RollingComparisonVideoResult,
    render_rolling_comparison_video,
)
from rosclaw_soccer.media.three_player_video import (
    ThreePlayerVideoResult,
    render_three_player_showcase_video,
)
from rosclaw_soccer.media.three_role_aerial_save_video import (
    render_three_role_aerial_save_video,
)
from rosclaw_soccer.media.three_role_save_portfolio_video import (
    render_three_role_save_portfolio_video,
)

__all__ = [
    "G1FreeKickVideoResult",
    "GoalkeeperV2DevelopmentVideoResult",
    "PhysicsGoalkeeperVideoResult",
    "RollingComparisonVideoResult",
    "ThreePlayerVideoResult",
    "render_alternating_growth_video",
    "render_aerial_curriculum_video",
    "render_collision_faithful_goalkeeper_video",
    "render_dynamic_aerial_lunge_video",
    "render_dynamic_takeoff_video",
    "render_g1_free_kick_showcase_video",
    "render_full_body_tactical_video",
    "render_goalkeeper_v2_development_video",
    "render_goalkeeper_showcase_video",
    "render_physics_goalkeeper_champion_video",
    "render_rolling_comparison_video",
    "render_regulation_dead_corner_video",
    "render_three_player_showcase_video",
    "render_three_role_aerial_save_video",
    "render_three_role_save_portfolio_video",
    "validate_alternating_growth_video_manifest",
    "validate_aerial_curriculum_video_manifest",
    "validate_collision_faithful_goalkeeper_manifest",
    "validate_dynamic_aerial_lunge_video_manifest",
    "validate_dynamic_takeoff_video_manifest",
    "validate_full_body_tactical_video_manifest",
    "validate_goalkeeper_showcase_manifest",
    "validate_regulation_dead_corner_video_manifest",
]
