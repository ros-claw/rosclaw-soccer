"""Explicit full-joint SIM reference experiment using disjoint bounded old blocks.

No direct joint target override: the unchanged original NN consumes each
conditioned reference on its own measured body/history. Not a learned policy.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motion_synergy_reference import (
    MotionSynergyPlan,
    condition_synergy_reference,
)


@dataclass(frozen=True)
class WholeJointReferencePlan:
    start_frame: int
    hg_to_isaac: tuple[int, ...]
    coefficients: tuple[tuple[float, ...], tuple[float, ...]]

    def blocks(self) -> tuple[MotionSynergyPlan, ...]:
        if (
            type(self.hg_to_isaac) is not tuple
            or len(self.hg_to_isaac) != 29
            or any(type(i) is not int for i in self.hg_to_isaac)
            or sorted(self.hg_to_isaac) != list(range(29))
            or type(self.coefficients) is not tuple
            or len(self.coefficients) != 2
            or any(type(r) is not tuple for r in self.coefficients)
        ):
            raise ValueError(
                "immutable exact HG29-to-Isaac permutation and two reference lobes required"
            )
        coeff = np.array(self.coefficients)
        if (
            coeff.shape != (2, 29)
            or coeff.dtype.kind != "f"
            or not np.isfinite(coeff).all()
            or abs(coeff).max() > 0.06
        ):
            raise ValueError("finite bounded full-joint reference coefficients required")
        basis = np.zeros((29, 29))
        basis[np.arange(29), self.hg_to_isaac] = 1
        blocks = tuple(
            MotionSynergyPlan(
                self.start_frame,
                tuple(tuple(float(x) for x in r) for r in basis[i : i + 6]),
                (
                    tuple(float(x) for x in coeff[0, i : i + 6]),
                    tuple(float(x) for x in coeff[1, i : i + 6]),
                ),
            )
            for i in range(0, 29, 6)
        )
        for block in blocks:
            block.offsets()  # Validate every block BEFORE mutating any policy reference.
        return blocks


def condition_whole_joint_reference(policy: Any, plan: WholeJointReferencePlan) -> dict[str, Any]:
    if type(plan) is not WholeJointReferencePlan:
        raise ValueError("explicit data-only whole-joint reference experiment required")
    blocks = plan.blocks()
    # First validate/install on a shallow software workspace. An error may not
    # leave a partially changed reference on the actual owner.
    from copy import copy

    scratch = copy(policy)
    receipts = [condition_synergy_reference(scratch, b) for b in blocks]
    policy.motion_joint_pos = scratch.motion_joint_pos
    policy.motion_joint_vel = scratch.motion_joint_vel
    return dict(
        schema="soccer.whole_joint_reference.v1",
        applied=any(r["applied"] for r in receipts),
        blocks=receipts,
        coefficients=plan.coefficients,
        hg_to_isaac=plan.hg_to_isaac,
        maximum_reference_offset_rad=float(
            max(r.get("maximum_reference_offset_rad", 0) for r in receipts)
        ),
        unchanged_neural_weights=True,
        actual_joint_targets_overridden=False,
        physical_state_modified=False,
        hardware_authorized=False,
    )
