"""Deterministic own-observation muscle-memory reference, not neural SGD or RL.

All motion is predicted from current own288 features. There is no case ID,
frame, trajectory cursor, teacher or kinematic simulator inside this model.
An inverse-square cardinal interpolator separates representation error from
whether the audited teacher trajectories can be followed closed-loop at all.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_compact_support import CompactTeacherSupport, check_features
from rosclaw_soccer.rsi.paid_interventional_student import data_support

CARDINAL_STATUS = "PAID_CARDINAL_MOTOR_MEMORY_FIT_NOT_PHYSICALLY_QUALIFIED"
CARDINAL_CONTROL_STATUS = "PAID_CARDINAL_MEMORY_ZERO_COEFFICIENT_CONTROL_NOT_PHYSICALLY_QUALIFIED"
GROWING_STATUS = "PAID_GROWING_CARDINAL_MEMORY_FIT_NOT_PHYSICALLY_QUALIFIED"
GROWING_CONTROL_STATUS = "PAID_GROWING_CARDINAL_MEMORY_ZERO_CONTROL_NOT_PHYSICALLY_QUALIFIED"


def cardinal_config(
    *, control: bool = False, canonical_behavior: bool = False, growing_memory: bool = False
) -> dict[str, Any]:
    if (
        type(control) is not bool
        or type(canonical_behavior) is not bool
        or type(growing_memory) is not bool
        or canonical_behavior
        and growing_memory
    ):
        raise ValueError("typed zero-coefficient control required")
    config = dict(
        protocol="paid.own288.cardinal_muscle_memory_reference.v1",
        algorithm="inverse_square_distance_cardinal_convex_interpolation",
        deterministic=True,
        control=control,
        optimization_seed=None,
        learned_representation="supervised_context_action_memory",
        neural_SGD=False,
        online_RL=False,
        features=288,
        actions=29,
        relative_correction_bound_rad=0.4,
        final_R2_bound_rad=0.2,
        radius_fraction=0.45,
        parent_frozen=True,
        base_frozen=True,
        runtime_case_lookup=False,
        runtime_frame_lookup=False,
        runtime_teacher=False,
        purpose="deterministic representation reference, NOT Champion promotion",
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
    )
    if canonical_behavior:
        config.update(
            protocol="paid.own288.cardinal_canonical_behavior_reference.v2",
            canonical_behavior=True,
            dataset_style="selected_successful_complete_behavior",
        )
    if growing_memory:
        config.update(
            protocol="paid.own288.growing_cardinal_memory_reference.v3",
            growing_memory=True,
            dataset_style="current_parent_and_executed_recovery",
            frozen_parent_type="canonical_cardinal_memory_v2",
        )
    return config


class CardinalMemoryHead(torch.nn.Module):
    bound_radians = 0.2
    memory_centres: torch.Tensor
    coefficients: torch.Tensor
    support: CompactTeacherSupport

    def __init__(
        self,
        support: CompactTeacherSupport,
        centres: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> None:
        super().__init__()
        check_features(centres)
        if (
            type(support) is not CompactTeacherSupport
            or coefficients.shape != (len(centres), 29)
            or coefficients.dtype != torch.float32
            or coefficients.device != centres.device
            or centres.device != support.centres.device
            or not torch.isfinite(coefficients).all()
            or (coefficients.abs() > 0.40002).any()
        ):
            raise ValueError("bounded own-context supervised memory required")
        self.support = support
        self.register_buffer("memory_centres", centres.detach().clone())
        self.register_buffer("coefficients", coefficients.detach().clone().clamp(-0.4, 0.4))

    def correction(self, features: torch.Tensor) -> torch.Tensor:
        check_features(features)
        if features.device != self.memory_centres.device:
            raise ValueError("same memory/query device required")
        rows = []
        # Bounded chunks and direct double subtraction avoid distance cancellation
        # next to a stored context, where cardinal precision is the whole point.
        for start in range(0, len(features), 16):
            x = features[start : start + 16].double()
            d2 = (x[:, None] - self.memory_centres.double()[None]).square().sum(-1)
            minimum = d2.min(dim=1, keepdim=True).values
            weights = minimum / d2.clamp_min(torch.finfo(torch.float64).tiny)
            weights = torch.where(minimum == 0, (d2 == 0).double(), weights)
            weights = weights / weights.sum(dim=1, keepdim=True)
            rows.append((weights @ self.coefficients.double()).float())
        return torch.cat(rows)

    def combine(
        self, features: torch.Tensor, prior: torch.Tensor, gain: torch.Tensor | None = None
    ) -> torch.Tensor:
        check_features(features)
        if (
            prior.shape != (len(features), 29)
            or prior.dtype != torch.float32
            or prior.device != features.device
            or not torch.isfinite(prior).all()
            or (prior.abs() > 0.2).any()
        ):
            raise ValueError("finite bounded frozen motor prior required")
        if gain is None:
            gain = self.support(features)
        if (
            gain.shape != (len(features), 1)
            or gain.dtype != torch.float32
            or gain.device != features.device
            or not torch.isfinite(gain).all()
            or (gain < 0).any()
            or (gain > 1).any()
        ):
            raise ValueError("finite same-context support required")
        if not (gain > 0).any():
            return prior
        predicted = (prior + gain * self.correction(features)).clamp(-0.2, 0.2)
        return torch.where(gain > 0, predicted, prior)


def fit_cardinal_head(
    z: dict[str, np.ndarray], *, control: bool = False
) -> tuple[CardinalMemoryHead, dict[str, int]]:
    cardinal_config(control=control)
    features, target, active = (z[k] for k in ("features", "target", "active"))
    if (
        active.dtype != np.bool_
        or active.shape != (len(features),)
        or target.dtype != np.float32
        or target.shape != (len(features), 29)
        or features.dtype != np.float32
        or not active.any()
    ):
        raise ValueError("complete typed active TRAIN memory required")
    unique: list[int] = []
    known: dict[bytes, int] = {}
    for index in np.flatnonzero(active):
        i = int(index)
        key = features[i].tobytes()
        if key in known:
            if not np.allclose(target[i], target[known[key]], rtol=0, atol=2e-6):
                raise ValueError("identical context has conflicting motor targets")
        else:
            known[key] = i
            unique.append(i)
    head = CardinalMemoryHead(
        data_support(z),
        torch.from_numpy(features[unique].copy()),
        torch.zeros(len(unique), 29) if control else torch.from_numpy(target[unique].copy()),
    ).eval()
    return head, dict(
        active_source_rows=int(active.sum()),
        memory_entries=len(unique),
        compatible_duplicates_removed=int(active.sum()) - len(unique),
    )


def cardinal_certificate(
    head: CardinalMemoryHead, z: dict[str, np.ndarray], *, control: bool = False
) -> dict[str, Any]:
    cardinal_config(control=control)
    maximum_error, squared_error, active, inactive = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for i, row in enumerate(z["features"]):
            x = torch.from_numpy(row.copy()).reshape(1, 288)
            prior = torch.from_numpy(z["parent_delta"][i].copy()).reshape(1, 29)
            gain = head.support(x)
            predicted = head.combine(x, prior, gain)
            if not torch.isfinite(predicted).all() or (predicted.abs() > 0.2).any():
                raise ValueError("original finite authority must hold")
            if control and not torch.equal(predicted, prior):
                raise ValueError(
                    "zero-coefficient control must exactly preserve every parent output"
                )
            if z["active"][i]:
                if float(gain) != 1:
                    raise ValueError("every active training context must open")
                error = predicted[0] - prior[0] - torch.from_numpy(z["target"][i].copy())
                maximum_error = max(maximum_error, float(error.abs().max()))
                squared_error += float(error.square().sum())
                active += 1
            else:
                if gain.count_nonzero() or not torch.equal(predicted, prior):
                    raise ValueError("old inactive anchors must preserve motor memory exactly")
                inactive += 1
    if not control and maximum_error > 2.1e-6:
        raise ValueError("cardinal TRAIN representation failed its own fixed precision contract")
    return dict(
        batch_shape=1,
        active_rows=active,
        inactive_rows=inactive,
        maximum_active_error_rad=maximum_error,
        active_MSE=squared_error / (29 * active),
        all_inactive_predictions_exact_parent=True,
        physical_qualification=False,
        neural_optimizer_updates=0,
        memory_entries_fit=0 if control else len(head.memory_centres),
        new_data_learning_effect=not control,
    )


class FrozenCardinalResidual(torch.nn.Module):
    bound_radians = 0.2

    def __init__(self, parent: torch.nn.Module, head: CardinalMemoryHead) -> None:
        from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter

        super().__init__()
        if type(parent) is not AncestralResidualRouter or type(head) is not CardinalMemoryHead:
            raise ValueError("exact frozen ancestor and learned cardinal memory required")
        self.parent = parent.requires_grad_(False).eval()
        self.head = head.requires_grad_(False).eval()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head.combine(features, self.parent(features))


class FrozenGrowingCardinalResidual(torch.nn.Module):
    """One explicitly typed growth layer; the old ancestral loader stays strict."""

    bound_radians = 0.2

    def __init__(self, parent: FrozenCardinalResidual, head: CardinalMemoryHead) -> None:
        super().__init__()
        if type(parent) is not FrozenCardinalResidual or type(head) is not CardinalMemoryHead:
            raise ValueError("exact current cardinal motor and supervised growth head required")
        self.parent = parent.requires_grad_(False).eval()
        self.head = head.requires_grad_(False).eval()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head.combine(features, self.parent(features))


def load_cardinal_student(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[Any, dict[str, Any]]:
    from rosclaw_soccer.rsi.paid_canonical_behavior import load_canonical_data
    from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor, digest
    from rosclaw_soccer.rsi.paid_growth_memory_data import load_growth_data
    from rosclaw_soccer.rsi.paid_replan_data import load_replanned_data

    if (
        checkpoint.is_symlink()
        or training_receipt.is_symlink()
        or checkpoint.parent != training_receipt.parent
        or checkpoint.stat().st_size > 16000000
        or digest(checkpoint) != checkpoint_hash
        or digest(training_receipt) != training_receipt_hash
        or (training_receipt.parent / "failed.json").exists()
    ):
        raise ValueError("exact bounded safe cardinal-memory artifacts required")
    r = json.loads(training_receipt.read_text())
    dp = training_receipt.parent / "declaration.json"
    s = torch.load(checkpoint, weights_only=True, map_location="cpu")
    control = s.get("config", {}).get("control")
    canonical = s.get("config", {}).get("canonical_behavior", False)
    growing = s.get("config", {}).get("growing_memory", False)
    config = cardinal_config(control=control, canonical_behavior=canonical, growing_memory=growing)
    schema = (
        "paid.cardinal_motor_memory.v3"
        if growing
        else "paid.cardinal_motor_memory.v2"
        if canonical
        else "paid.cardinal_motor_memory.v1"
    )
    expected_status = (
        (GROWING_CONTROL_STATUS if control else GROWING_STATUS)
        if growing
        else (CARDINAL_CONTROL_STATUS if control else CARDINAL_STATUS)
    )
    if (
        dp.is_symlink()
        or digest(dp) != r["declaration_hash"]
        or s.get("schema") != schema
        or s.get("config") != config
        or r.get("config") != config
        or r.get("status") != expected_status
        or r.get("checkpoint_hash") != checkpoint_hash
        or s.get("body_scene_hash") != scene_hash
        or r.get("optimizer_updates") != 0
        or r.get("own_native_steps") != 0
        or r.get("Fresh_fit_rows") != 0
        or r.get("DEV_fit_rows") != 0
        or r.get("parent_parameters_exact_unchanged") is not True
        or any(
            v.get(k) is not False
            for v in (s, r)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("typed research-only cardinal fitting contract required")
    d = json.loads(dp.read_text())
    for k in (
        "config",
        "source_head",
        "data_hash",
        "body_scene_hash",
        "parent_hash",
        "parent_checkpoint_hash",
        "declaration_hash",
    ):
        if s.get(k) != r.get(k) or (k != "declaration_hash" and d.get(k) != s.get(k)):
            raise ValueError("exact declared memory data/body/source identity required")
    loader = (
        load_growth_data if growing else load_canonical_data if canonical else load_replanned_data
    )
    parent, data, z = loader(Path(d["data_root"]), s["data_hash"], policy)
    if any(data[k] != s[k] for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")):
        raise ValueError("frozen motor lineage changed")
    head, counts = fit_cardinal_head(z, control=control)
    expected = head.state_dict()
    stored = s.get("memory", {})
    if set(stored) != set(expected) or any(
        not isinstance(stored[k], torch.Tensor)
        or stored[k].device.type != "cpu"
        or stored[k].dtype != v.dtype
        or stored[k].shape != v.shape
        or not torch.equal(stored[k], v)
        for k, v in expected.items()
    ):
        raise ValueError("stored memory differs from full causal TRAIN reconstruction")
    if s.get("counts") != counts or r.get("counts") != counts:
        raise ValueError("source rows, unique memories and removed duplicates differ")
    cert = cardinal_certificate(head, z, control=control)
    if s.get("learning_certificate") != cert or r.get("learning_certificate") != cert:
        raise ValueError("batch1 cardinal precision and ancestral retention certificate differs")
    residual = (
        FrozenGrowingCardinalResidual(parent.residual, head)
        if growing
        else FrozenCardinalResidual(parent.residual, head)
    ).eval()
    return FrozenFullEntryActor(parent.base, residual, parent.action_scale).eval(), r
