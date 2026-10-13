"""JSON-stable identity checks, not physics admission or execution authority."""

from typing import Any

from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_json


def require_native_training_identity(
    body: dict[str, Any],
    view: dict[str, Any],
    envelope: dict[str, Any],
    plan: PublicFirstTouchTrainingPlan,
    group: int,
    *,
    entry_source_hash: str,
    kernel_source_hash: str,
) -> None:
    """Compare serialized plans canonically; never weaken seed/source/safety checks.

    The caller must authenticate all inputs and independently replay physics.
    A successful identity comparison alone never admits a trajectory.
    """
    if type(group) is not int or not 0 <= group < len(plan.draws()):
        raise ValueError("declared group required")
    draw = plan.draws()[group]
    plan.require_sampling_view(draw, view)
    expected = {
        "seed": draw.seed,
        "lane": draw.lane,
        "course": list(draw.course),
        "partition": "DECLARED_PUBLIC_TRAINING",
        "entry_source_hash": entry_source_hash,
        "source_hash": kernel_source_hash,
        "step_model_hash": view["model_hash"],
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    # Canonical JSON distinguishes booleans from integers and preserves the
    # serialized tuple/list equivalence actually used by plan_hash.
    actual = {key: body.get(key) for key in expected}
    if (
        hash_json(actual) != hash_json(expected)
        or hash_json(body.get("public_training_declaration")) != hash_json(plan.contract())
        or envelope.get("schema") != "rosclaw.growth.shared_json_payload.v1"
        or envelope.get("location") != ["mean_model"]
        or envelope.get("envelope_hash")
        != hash_json({k: v for k, v in envelope.items() if k != "envelope_hash"})
        or envelope.get("stripped_document", {}).get("model_hash") != view["model_hash"]
    ):
        raise ValueError("exact declared identity and pinned native source required")
