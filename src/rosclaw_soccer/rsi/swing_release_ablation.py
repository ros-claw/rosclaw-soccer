"""Single-mechanism SIM ablation, not an optimized recovery controller."""

import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def release_ablation_contract(world_hash: str) -> dict[str, Any]:
    from rosclaw_soccer.rsi.taskspace_swing_probe import CONTACT_RELEASE_FRAMES

    if CONTACT_RELEASE_FRAMES != 20:
        raise ValueError("ablation requires the unchanged original 20-frame release law")
    if type(world_hash) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", world_hash) is None:
        raise ValueError("explicit compiled SIM world required")
    value = dict(
        schema="soccer.rsi.suppress_postcontact_swing_release.v1",
        world_hash=world_hash,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        intervention=(
            "zero taskspace swing increment from the next control tick after first observed contact"
        ),
        contact_trigger="any of the original six force channels exceeds 1N",
        original_release_frames=20,
        macro_navigation_unchanged=True,
        recurrent_motor_enabled=True,
        precontact_numerical_parity_required=True,
        optimized_recovery_claimed=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(value, contract_hash=hash_json(value))


def require_zero_postcontact_swing(commitment: dict[str, Any], trace: dict[str, Any]) -> None:
    """Verify this ablation's actual increment, never infer improved task success."""
    if "swing_release_ablation" not in commitment:
        return
    if (
        commitment["swing_release_ablation"]
        != release_ablation_contract(commitment["compiled_model_hash"])
        or "bounded_control_profile" not in commitment
        or "foundation_observation_capture" not in commitment
        or commitment.get("execution_profile") != "taskspace_plus_motor"
        or "cerebellar_torque_contract" in commitment
        or commitment.get("swing_mode") != "bounded_contact_diagnostic"
    ):
        raise ValueError("explicit bounded no-torque-layer release ablation required")
    force = np.asarray(trace.get("force_n"))
    delta = np.asarray(trace.get("taskspace_joint_increment_rad"))
    if any(
        a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all()
        for a, shape in ((force, (300, 1, 6)), (delta, (300, 1, 29)))
    ):
        raise ValueError("complete finite release-ablation force and increment traces required")
    if np.any(force < 0):
        raise ValueError("contact magnitude cannot be negative")
    observed = np.flatnonzero(np.any(force[:, 0] > 1, axis=1))
    if len(observed) and np.any(delta[int(observed[0]) + 1 :] != 0):
        raise ValueError("postcontact taskspace increment must be exactly zero in this ablation")
