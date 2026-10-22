"""Whole-history contact consolidation weights; no privileged inference inputs."""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor

BANK_STATUS = "STROKE_CONTACT_CONSOLIDATION_BANK65_READY_NOT_TRAINED"


def context_witness_hash(data: dict[str, Any], checkpoint_kind: str) -> str:
    """Keep the original fixed exam fixtures separate from the new training bank."""
    if checkpoint_kind == "contact-consolidated":
        if data["status"] != BANK_STATUS:
            raise ValueError("contact checkpoint requires its actual bank65")
        return str(data["context_witness_hash"])
    if checkpoint_kind not in ("supervised", "ar1-actor-critic") or data["status"] == BANK_STATUS:
        raise ValueError("explicit checkpoint and data protocol required")
    return str(data["teacher_hash"])


def own_reference_sequence(
    parent: "FrozenFullEntryActor", observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute causal frozen reference history, never borrow teacher hidden state."""
    import torch

    from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor

    if (
        type(parent) is not FrozenFullEntryActor
        or parent.training
        or any(p.requires_grad or p.device.type != "cpu" for p in parent.parameters())
        or observations.ndim != 2
        or observations.shape[1] != 160
        or not len(observations)
        or observations.dtype != np.float32
        or not np.isfinite(observations).all()
    ):
        raise ValueError("frozen CPU reference and finite actual float32 history required")
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    nominals, deltas = [], []
    with torch.no_grad():
        for value in observations:
            tensor = torch.from_numpy(value).reshape(1, 1, 160)
            nominal, state, hidden = parent.base.forward_with_features(tensor, state)
            feature = torch.cat(
                ((tensor - parent.base.mean) / parent.base.denominator, hidden), dim=-1
            )
            prior = parent.residual(feature.reshape(-1, 288))[0]
            nominals.append(nominal[0, 0].numpy().copy())
            deltas.append(prior.numpy().copy())
    return np.stack(nominals), np.stack(deltas)


def contact_loss_weights(
    kick_rows: int, stand_rows: int, first_contact_substep: int | None
) -> np.ndarray:
    if (
        type(kick_rows) is not int
        or not 1 <= kick_rows <= 300
        or type(stand_rows) is not int
        or not 0 <= stand_rows <= 100
        or (
            first_contact_substep is not None
            and (
                type(first_contact_substep) is not int
                or not 0 <= first_contact_substep < 10 * (kick_rows + stand_rows)
            )
        )
    ):
        raise ValueError("actual bounded neural controls and measured contact label required")
    weights = np.ones(kick_rows + stand_rows, dtype=np.float32)
    weights[: min(80, kick_rows)] = 20
    weights[kick_rows:] = 20
    if first_contact_substep is not None:
        at = first_contact_substep // 10
        weights[max(0, at - 24) : min(kick_rows, at + 21)] = 100
    return weights
