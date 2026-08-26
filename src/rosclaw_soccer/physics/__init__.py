"""Versioned football physics and regulation geometry."""

from rosclaw_soccer.physics.reality_pack import (
    RealityPackReport,
    run_reality_pack,
)
from rosclaw_soccer.physics.standards import IFABRegulationSpec

__all__ = ["IFABRegulationSpec", "RealityPackReport", "run_reality_pack"]
