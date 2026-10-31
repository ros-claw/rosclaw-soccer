"""Read-only contact fingerprints for simulation transfer diagnostics.

An observed match is a necessary local comparison, not proof that two models
or trajectories are dynamically equivalent. Callers must refresh MuJoCo's
contact data for the state they intend to inspect before calling this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContactSignature:
    """Effective contact values observed from one active MuJoCo contact."""

    dimension: int
    active_friction: tuple[float, ...]
    solref: tuple[float, ...]
    solref_friction: tuple[float, ...]
    solimp: tuple[float, ...]
    include_margin: float

    def __post_init__(self) -> None:
        if self.dimension not in (1, 3, 4, 6):
            raise ValueError("unsupported contact dimension")
        if len(self.active_friction) != self.dimension - 1:
            raise ValueError("active friction does not match contact dimension")
        if len(self.solref) != 2 or len(self.solref_friction) != 2 or len(self.solimp) != 5:
            raise ValueError("invalid contact solver parameter shape")
        values = (
            self.active_friction
            + self.solref
            + self.solref_friction
            + self.solimp
            + (self.include_margin,)
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("contact parameters must be finite")
        if any(value < 0.0 for value in self.active_friction):
            raise ValueError("contact friction must be non-negative")


@dataclass(frozen=True)
class ContactComparison:
    """Local contact comparison; never a motion or transfer authorization."""

    matches: bool
    mismatches: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"


def observe_contact(model: Any, data: Any, geom_a: int, geom_b: int) -> ContactSignature:
    """Return the effective parameters of exactly one observed contact pair.

    Missing or multiple contacts are rejected, not interpreted as matching.
    The caller owns ``model`` and must call ``mj_forward`` or step the simulator
    before inspection; this function never steps or changes simulator state.
    Geom IDs are local to this model and are not compared between models.
    """

    if (
        type(geom_a) is not int
        or type(geom_b) is not int
        or geom_a == geom_b
        or not 0 <= geom_a < int(model.ngeom)
        or not 0 <= geom_b < int(model.ngeom)
    ):
        raise ValueError("two distinct valid geometry IDs are required")
    matches = [
        contact
        for contact in data.contact[: data.ncon]
        if {int(contact.geom1), int(contact.geom2)} == {geom_a, geom_b}
        and int(contact.efc_address) >= 0
    ]
    if len(matches) != 1:
        raise ValueError("exactly one observed contact is required")
    contact = matches[0]
    dimension = int(contact.dim)
    if dimension not in (1, 3, 4, 6):
        raise ValueError("unsupported contact dimension")
    return ContactSignature(
        dimension=dimension,
        active_friction=tuple(float(value) for value in contact.friction[: dimension - 1]),
        solref=tuple(float(value) for value in contact.solref),
        solref_friction=tuple(float(value) for value in contact.solreffriction),
        solimp=tuple(float(value) for value in contact.solimp),
        include_margin=float(contact.includemargin),
    )


def compare_contacts(
    source: ContactSignature, target: ContactSignature, *, atol: float = 1e-12
) -> ContactComparison:
    """Compare active contact properties without claiming full-model parity."""

    if not isinstance(source, ContactSignature) or not isinstance(target, ContactSignature):
        raise TypeError("observed ContactSignature inputs are required")
    if not isfinite(atol) or not 0.0 <= atol <= 1e-6:
        raise ValueError("contact comparison tolerance must be finite and at most 1e-6")
    mismatches: list[str] = []
    if source.dimension != target.dimension:
        mismatches.append("dimension")
        common = min(len(source.active_friction), len(target.active_friction))
        if common and not np.allclose(
            source.active_friction[:common], target.active_friction[:common], rtol=0.0, atol=atol
        ):
            mismatches.append("shared_active_friction")
    elif not np.allclose(source.active_friction, target.active_friction, rtol=0.0, atol=atol):
        mismatches.append("active_friction")
    if not np.allclose(source.solref, target.solref, rtol=0.0, atol=atol):
        mismatches.append("solref")
    if not np.allclose(source.solref_friction, target.solref_friction, rtol=0.0, atol=atol):
        mismatches.append("solref_friction")
    if not np.allclose(source.solimp, target.solimp, rtol=0.0, atol=atol):
        mismatches.append("solimp")
    if abs(source.include_margin - target.include_margin) > atol:
        mismatches.append("include_margin")
    return ContactComparison(matches=not mismatches, mismatches=tuple(mismatches))
