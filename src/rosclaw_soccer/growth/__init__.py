"""Football-specific Growth adapters and candidate builders.

ROSClaw Core owns task-neutral evidence and safety contracts.  This package
owns the football semantics that populate those contracts.  Importing it has
no simulator, runtime, ROS, or hardware side effects.
"""

from rosclaw_soccer.growth.adapter import SOCCER_GROWTH_ADAPTER, SoccerGrowthAdapter

__all__ = ["SOCCER_GROWTH_ADAPTER", "SoccerGrowthAdapter"]
