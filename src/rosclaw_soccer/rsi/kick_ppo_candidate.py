"""Read a provisional research checkpoint; never authorize promotion or hardware."""

import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.velocity_policy import file_hash


def admit_pilot(path: Path, expected_hash: str) -> tuple[dict[str, str], dict[str, Any]]:
    import torch

    from rosclaw_soccer.rsi.kick_prior_actor import KICK_PRIOR_HASH

    if path.is_symlink() or file_hash(path) != expected_hash:
        raise ValueError("exact complete provisional head-PPO report required")
    report = json.loads(path.read_text())
    if report[
        "status"
    ] != "ONE_OWN_POLICY_KICK_PPO_PROPOSAL_AND_FOUR_VALUE_UPDATES_NOT_PHYSICALLY_QUALIFIED" or (
        report["actual_actor_optimizer_updates"] != 1
        or report["actor_update_attempts"] != 1
        or report["actual_value_optimizer_updates"] != 4
        or report["actor_decision"]["accepted"] is not True
        or report["original_motor_features_unchanged"] is not True
        or report["promotion_authorized"] is not False
        or report["hardware_authorized"] is not False
    ):
        raise ValueError("accepted actual one-step motor candidate, not a forecast model required")
    artifact = report["checkpoint"]
    cp = path.parent / artifact["path"]
    if (
        cp.is_symlink()
        or cp.parent.resolve() != path.parent.resolve()
        or file_hash(cp) != artifact["hash"]
    ):
        raise ValueError("exact confined candidate checkpoint required")
    checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
    required = dict(
        schema="soccer.g1.kick_head_ppo_pilot_checkpoint.v1",
        source_model_hash=KICK_PRIOR_HASH,
        actual_actor_optimizer_updates=1,
        actual_value_optimizer_updates=4,
        actor_update_attempts=1,
        simulation_only=True,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if any(checkpoint[k] != v for k, v in required.items()):
        raise ValueError("exact candidate lineage and simulation-only contract required")
    pins = dict(report["input_file_hashes"])
    pins.update({str(path): expected_hash, str(cp): artifact["hash"]})
    return pins, checkpoint


def candidate_factory(original_factory: Any, checkpoint: dict[str, Any]) -> Any:
    """Fresh private original policy type, with only the admitted readout changed."""
    import torch

    def create(*args: Any, **kwargs: Any) -> Any:
        policy = original_factory(*args, **kwargs)
        actor = policy.ort_session.actor
        original = actor.state_dict()
        changed = checkpoint["actor"]
        if set(changed) != set(original) or any(
            v.shape != original[k].shape
            or v.dtype != torch.float32
            or not torch.isfinite(v).all()
            or (not k.startswith("actor.6.") and not torch.equal(v, original[k]))
            for k, v in changed.items()
        ):
            raise ValueError("candidate changed frozen features or network contract")
        if not any(
            not torch.equal(changed[k], original[k]) for k in changed if k.startswith("actor.6.")
        ):
            raise ValueError("candidate has no learned motor readout change")
        actor.load_state_dict(changed, strict=True)
        return policy

    return create
