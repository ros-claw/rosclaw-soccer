"""Shooting skills kept downstream from ROSClaw Core.

The free-kick controller owns a three-player integration and therefore imports
team modules.  Keep those exports lazy so importing the standalone loft
teacher from a team controller cannot re-enter a partially initialized team
package.
"""

from __future__ import annotations

from typing import Any

from rosclaw_soccer.skills.shoot.loft_teacher import (
    G1LoftTeacherConfig,
    G1LoftTeacherEffect,
    g1_loft_teacher_effect,
    project_g1_vertical_foot_force,
)

_FREE_KICK_EXPORTS = {
    "G1FootballEventPhase",
    "G1FrontDuelConfig",
    "G1FrontDuelSummary",
    "G1FreeKickEvidence",
    "G1FreeKickFlowConfig",
    "G1FreeKickResult",
    "run_g1_free_kick_showcase",
}


def __getattr__(name: str) -> Any:
    if name not in _FREE_KICK_EXPORTS:
        raise AttributeError(name)
    from rosclaw_soccer.skills.shoot import free_kick

    return getattr(free_kick, name)


__all__ = [
    "G1FootballEventPhase",
    "G1FrontDuelConfig",
    "G1FrontDuelSummary",
    "G1FreeKickEvidence",
    "G1FreeKickFlowConfig",
    "G1FreeKickResult",
    "G1LoftTeacherConfig",
    "G1LoftTeacherEffect",
    "g1_loft_teacher_effect",
    "project_g1_vertical_foot_force",
    "run_g1_free_kick_showcase",
]
