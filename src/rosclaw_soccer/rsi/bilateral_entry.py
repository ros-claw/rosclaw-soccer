"""Opt-in left-foot reference and entry prior, not mirrored physical state."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_sagittal_reference import mirror_sagittal_reference
from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True)
class ReflectedEntryAffordance(SkillEntryAffordance):
    original_affordance_hash: str

    @property
    def contract_hash(self) -> str:
        return hash_json(
            dict(
                schema="soccer.rsi.reflected_entry_affordance.v1",
                **self.__dict__,
                prior_kind="sagittal_reflection_of_right_historical_prior_not_executed_left_motion",
                contact_qualified=False,
                neural_training_updates=0,
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )


def contact_foot_context(
    source: np.ndarray, affordance: SkillEntryAffordance, foot: str
) -> tuple[np.ndarray, SkillEntryAffordance, dict[str, Any] | None]:
    if type(foot) is not str or foot not in ("right", "left"):
        raise ValueError("explicit right or left reference foot required")
    if foot == "right":
        return source, affordance, None
    if type(affordance) is not SkillEntryAffordance:
        raise ValueError("reflect the original right-foot prior exactly once")
    affordance.__post_init__()
    reflected = mirror_sagittal_reference(source)
    entry = ReflectedEntryAffordance(
        (affordance.ball_offset_xy_m[0], -affordance.ball_offset_xy_m[1]),
        affordance.historical_frame,
        affordance.historical_forward_speed_mps,
        affordance.historical_artifact_hash,
        affordance.contract_hash,
    )
    report = dict(
        schema="soccer.rsi.bilateral_entry_context.v1",
        intended_reference_foot=foot,
        source_pose_hash=hash_bytes(source.tobytes()),
        reflected_pose_hash=hash_bytes(reflected.tobytes()),
        original_affordance_hash=affordance.contract_hash,
        reflected_affordance_hash=entry.contract_hash,
        source_frequency_hz=120,
        source_window=[286, 767],
        entry_intervals=50,
        modifies_physical_state_or_control_history=False,
        mirrored_prior_is_executed_left_motion=False,
        dynamic_tracking_qualified=False,
        hardware_authorized=False,
    )
    return reflected, entry, report


def choose_bilateral_candidate(rows: list[dict[str, Any]]) -> int:
    if len(rows) != 9 or [r["law_index"] for r in rows] != [0, 0, 3, 3, 4, 4, 0, 3, 4]:
        raise ValueError("six original anchors and three fixed left options required")
    if [r["offset_root_m"] for r in rows[:6]] != [[0.0, 0.0, 0.0], [0.0, -0.025, -0.025]] * 3:
        raise ValueError("all original anchor offsets required")
    if any(r.get("contact_reference_foot", "right") != "right" for r in rows[:6]):
        raise ValueError("original anchors must remain right references")
    if any(
        r.get("contact_reference_foot") != "left"
        or r["offset_root_m"] != [0.0, 0.0, 0.0]
        or r.get("entry_reference_mode", "legacy") != "legacy"
        for r in rows[6:]
    ):
        raise ValueError("left options require unchanged legacy50 entry and zero C")
    for r in rows:
        score = r["score"]
        if type(score["qualified"]) is not bool or (
            score["qualified"]
            and (type(score["cost"]) not in (float, int) or not math.isfinite(score["cost"]))
        ):
            raise ValueError("typed original gate/cost required")
    eligible = [i for i, r in enumerate(rows) if r["score"]["qualified"]]
    return min(eligible, key=lambda i: (rows[i]["score"]["cost"], i)) if eligible else 0


def observed_contact_foot(
    model: Any, arrays: dict[str, np.ndarray], *, global_start_frame: int
) -> dict[str, Any]:
    forces = arrays["ball_contact_force_by_body_n"]
    if forces.ndim != 3 or forces.shape[1:] != (10, model.nbody) or not np.isfinite(forces).all():
        raise ValueError("finite complete native ball contact body array required")
    contacts = forces.reshape(-1, model.nbody)
    indices = {
        side: [model.body(side + suffix).id for suffix in ("_ankle_roll_link", "_ankle_pitch_link")]
        for side in ("left", "right")
    }
    first = {}
    for side, ids in indices.items():
        found = np.flatnonzero(np.any(contacts[:, ids] > 1, axis=1))
        first[side] = int(found[0]) if len(found) else None
    available = [value for value in first.values() if value is not None]
    earliest = min(available) if available else None
    return dict(
        first_contact_foot_sides=[
            side for side, k in first.items() if earliest is not None and k == earliest
        ],
        first_contact_global_frame=(global_start_frame + earliest // 10)
        if earliest is not None
        else None,
        first_contact_native_offset=earliest,
        first_native_contact_by_side=first,
        contact_threshold_n=1,
        intended_reference_is_not_observed_contact=True,
    )
