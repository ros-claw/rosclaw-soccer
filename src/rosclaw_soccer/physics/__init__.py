"""Versioned football physics and regulation geometry."""

from rosclaw_soccer.physics.reality_pack import (
    RealityPackReport,
    run_reality_pack,
)
from rosclaw_soccer.physics.rolling_authenticity import (
    RollingAuditResult,
    RollingAuthenticityMetrics,
    RollingAuthenticityThresholds,
    audit_pass_rolling_physics,
    measure_rolling_authenticity,
)
from rosclaw_soccer.physics.standards import IFABRegulationSpec

__all__ = [
    "IFABRegulationSpec",
    "RealityPackReport",
    "RollingAuditResult",
    "RollingAuthenticityMetrics",
    "RollingAuthenticityThresholds",
    "audit_pass_rolling_physics",
    "measure_rolling_authenticity",
    "run_reality_pack",
]
