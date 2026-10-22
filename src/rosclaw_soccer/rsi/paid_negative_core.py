"""TRAIN-only protected cores with broad neural feedback, SIM research only.

Unlike positive compact cones, this gate is not identically zero everywhere
outside demonstrated teacher neighborhoods. That is a hypothesis, not an OOD
safety guarantee. No case IDs, clock, or trajectory cursor enter inference.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_canonical_behavior import digest
from rosclaw_soccer.rsi.paid_compact_support import check_features
from rosclaw_soccer.rsi.paid_replay_neural import ReplayMotorHead, replay_config, training_arrays

STATUS = "PAID_NEGATIVE_CORE_NEURAL_CPU_CERTIFIED_NOT_PHYSICALLY_QUALIFIED"


def negative_core_config() -> dict[str, Any]:
    return dict(
        protocol="paid.own288.negative_core_neural.v1",
        features=288,
        distance="exact_float64_euclidean_cKDTree_eps0_workers1",
        scipy_version="1.18.1",
        protected_ratio_core=0.1,
        ratio="distance_negative/(distance_negative+distance_positive)",
        gain="clip((ratio-0.1)/0.9,0,1)",
        frozen_neural_seed_index=0,
        frozen_neural_protocol="paid.own288.soft_replay_neural.v1",
        new_neural_optimizer_updates=0,
        labels_fit=0,
        final_R2_bound_rad=0.2,
        correction_bound_rad=0.4,
        runtime_case_lookup=False,
        runtime_frame_lookup=False,
        broader_unsupported_action_region=True,
        OOD_safety_claim=False,
        Fresh_rows_read=0,
        training32_already_consumed=True,
    )


class NegativeCoreGate(torch.nn.Module):
    def __init__(self, positive: torch.Tensor, negative: torch.Tensor) -> None:
        super().__init__()
        import scipy
        from scipy.spatial import cKDTree

        for x in (positive, negative):
            check_features(x)
            if x.device.type != "cpu":
                raise ValueError("exact CPU prototype lookup only")
        if scipy.__version__ != negative_core_config()["scipy_version"]:
            raise ValueError("pinned exact query implementation required")
        self._positive = cKDTree(positive.detach().numpy().astype(np.float64), copy_data=True)
        self._negative = cKDTree(negative.detach().numpy().astype(np.float64), copy_data=True)
        self._positive.data.flags.writeable = False
        self._negative.data.flags.writeable = False

    def distances(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        check_features(x)
        if x.device.type != "cpu":
            raise ValueError("CPU-only exact gate must not silently move query devices")
        query = x.detach().numpy().astype(np.float64)
        positive = self._positive.query(query, k=1, eps=0, workers=1)[0]
        negative = self._negative.query(query, k=1, eps=0, workers=1)[0]
        if (
            not np.isfinite(positive).all()
            or not np.isfinite(negative).all()
            or ((positive + negative) == 0).any()
        ):
            raise ValueError("finite disjoint positive/protected prototype geometry required")
        return np.asarray(positive), np.asarray(negative)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive, negative = self.distances(x)
        ratio = negative / (positive + negative)
        return torch.from_numpy(np.clip((ratio - 0.1) / 0.9, 0, 1).astype(np.float32)).reshape(
            -1, 1
        )


def gated_neural_delta(
    head: ReplayMotorHead, gate: NegativeCoreGate, x: torch.Tensor, prior: torch.Tensor
) -> torch.Tensor:
    gain = gate(x)
    if (
        prior.shape != (len(x), 29)
        or prior.dtype != torch.float32
        or prior.device.type != "cpu"
        or not torch.isfinite(prior).all()
        or (prior.abs() > 0.2).any()
    ):
        raise ValueError("finite bounded current-parent output required")
    if not (gain > 0).any():
        return prior
    candidate = (prior + gain * head(x)).clamp(-0.2, 0.2)
    return torch.where(gain > 0, candidate, prior)


def negative_core_certificate(
    head: ReplayMotorHead, gate: NegativeCoreGate, z: dict[str, np.ndarray]
) -> dict[str, Any]:
    x, y, prior, active = training_arrays(z)
    with torch.no_grad():
        gains = torch.cat([gate(x[start : start + 128]) for start in range(0, len(x), 128)])[:, 0]
        if not torch.equal(gains[active], torch.ones_like(gains[active])) or gains[~active].any():
            raise ValueError("all TRAIN teacher/protected anchor interpolation properties required")
        # Actual batch1 proposed bounded motion, including bitwise parent return.
        predictions = torch.cat(
            [
                gated_neural_delta(head, gate, row.reshape(1, 288), p.reshape(1, 29))
                for row, p in zip(x, prior, strict=True)
            ]
        )
    if not torch.equal(predictions[~active], prior[~active]):
        raise ValueError("every negative anchor must preserve current motor output exactly")
    if not torch.isfinite(predictions).all() or (predictions.abs() > 0.2).any():
        raise ValueError("finite original final motor authority required")
    # Independent brute-force checks for exact-tree distances; synthetic query
    # perturbations are geometry checks, NOT physics observations or labels.
    queries = x[np.linspace(0, len(x) - 1, 32, dtype=int)].clone()
    queries[:, 154] += 0.001
    dp, dn = gate.distances(queries)
    brute = []
    for collection, got in (
        (x[active].numpy().astype(np.float64), dp),
        (x[~active].numpy().astype(np.float64), dn),
    ):
        expected = np.array(
            [
                np.sqrt(np.square(collection - q).sum(1).min())
                for q in queries.numpy().astype(np.float64)
            ]
        )
        error = float(np.abs(expected - got).max())
        if error > 2e-12:
            raise ValueError("exact-nearest implementation differs from brute-force geometry")
        brute.append(error)
    return dict(
        TRAIN_rows=len(x),
        active_gain_one_rows=int(active.sum()),
        protected_gain_zero_rows=int((~active).sum()),
        CPU_batch1_motor_rows=len(x),
        maximum_protected_motor_change_rad=0.0,
        active_final_motor_MSE=float(
            (predictions[active] - (prior[active] + y[active]).clamp(-0.2, 0.2)).square().mean()
        ),
        maximum_final_R2_residual_rad=float(predictions.abs().max()),
        brute_force_queries=32,
        brute_force_maximum_distance_errors=brute,
        physical_qualification=False,
        unseen_generalization_claim=False,
    )


def reconstruct_negative_core(
    audit_root: Path, audit_hash: str, policy: Path
) -> tuple[Any, ReplayMotorHead, NegativeCoreGate, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from rosclaw_soccer.rsi.paid_replay_neural_student import AUDITED_STATUS, load_replay_fit

    rp = audit_root / "complete.json"
    if rp.is_symlink() or digest(rp) != audit_hash or (audit_root / "failed.json").exists():
        raise ValueError("exact closed independent neural audit required")
    audit = json.loads(rp.read_text())
    if (
        audit["status"] != AUDITED_STATUS
        or audit["config"] != replay_config(0)
        or audit["CPU_fit_reconstruction"] is not True
        or any(audit.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("fixed primary v1 neural reference, not best seed or tail-refit required")
    parent, head, fit = load_replay_fit(Path(audit["fit_root"]), audit["fit_hash"], policy)
    if any(
        audit[k] != fit[k]
        for k in (
            "checkpoint_hash",
            "checkpoint_path",
            "counts",
            "certificate",
            "body_scene_hash",
            "parent_hash",
        )
    ):
        raise ValueError("neural audit and fit differ")
    fd = Path(audit["fit_root"]) / "declaration.json"
    d = json.loads(fd.read_text())
    data_root = Path(d["data_root"])
    data = json.loads((data_root / "complete.json").read_text())
    if digest(data_root / "complete.json") != fit["data_hash"]:
        raise ValueError("same causally reconstructed TRAIN required")
    path = Path(data["feature_path"])
    if path.is_symlink() or path.parent != data_root or digest(path) != data["feature_hash"]:
        raise ValueError("exact verified features required")
    with np.load(path, allow_pickle=False) as a:
        z = {k: a[k].copy() for k in a.files}
    x, _, _, active = training_arrays(z)
    gate = NegativeCoreGate(x[active], x[~active])
    certificate = negative_core_certificate(head, gate, z)
    return parent, head, gate, fit, data, certificate


class FrozenNegativeCoreResidual(torch.nn.Module):
    bound_radians = 0.2

    def __init__(
        self, parent: torch.nn.Module, head: ReplayMotorHead, gate: NegativeCoreGate
    ) -> None:
        super().__init__()
        from rosclaw_soccer.rsi.paid_cardinal_memory import FrozenCardinalResidual

        if (
            type(parent) is not FrozenCardinalResidual
            or type(head) is not ReplayMotorHead
            or type(gate) is not NegativeCoreGate
        ):
            raise ValueError(
                "exact canonical parent/frozen neural head/reconstructed gate required"
            )
        self.parent = parent.eval().requires_grad_(False)
        self.head = head.eval().requires_grad_(False)
        self.gate = gate.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gated_neural_delta(self.head, self.gate, x, self.parent(x))


def load_negative_core_student(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[Any, dict[str, Any]]:
    from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor

    if (
        training_receipt.is_symlink()
        or digest(training_receipt) != training_receipt_hash
        or checkpoint.is_symlink()
        or checkpoint.parent != training_receipt.parent
        or checkpoint.stat().st_size > 1048576
        or digest(checkpoint) != checkpoint_hash
        or (training_receipt.parent / "failed.json").exists()
    ):
        raise ValueError("confined immutable metadata and source-bound learned weights required")
    r = json.loads(training_receipt.read_text())
    dp = training_receipt.parent / "declaration.json"
    if (
        r["status"] != STATUS
        or r["config"] != negative_core_config()
        or r["body_scene_hash"] != scene_hash
        or r["checkpoint_path"] != str(checkpoint)
        or r["checkpoint_hash"] != checkpoint_hash
        or r.get("optimizer_updates") != 0
        or r.get("own_native_steps") != 0
        or r.get("inherited_neural_optimizer_updates") != 6144
        or dp.is_symlink()
        or digest(dp) != r["declaration_hash"]
        or any(r.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("exact same-body SIM negative-core candidate required")
    d = json.loads(dp.read_text())
    if any(
        d[k] != r[k]
        for k in (
            "config",
            "audit_root",
            "audit_hash",
            "body_scene_hash",
            "data_hash",
            "parent_hash",
            "feature_hash",
            "source_head",
        )
    ):
        raise ValueError("negative-core declaration differs")
    parent, head, gate, fit, data, certificate = reconstruct_negative_core(
        Path(r["audit_root"]), r["audit_hash"], policy
    )
    expected = dict(
        schema="paid.negative_core_neural.references.v1",
        config=negative_core_config(),
        audit_hash=r["audit_hash"],
        data_hash=fit["data_hash"],
        feature_hash=data["feature_hash"],
        parent_hash=fit["parent_hash"],
        body_scene_hash=fit["body_scene_hash"],
        certificate=certificate,
        declaration_hash=r["declaration_hash"],
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state != expected
        or r["certificate"] != certificate
        or any(
            r[k] != expected[k]
            for k in ("parent_hash", "body_scene_hash", "data_hash", "feature_hash")
        )
    ):
        raise ValueError("recomputed gate/data/frozen neural model certificate differs")
    residual = FrozenNegativeCoreResidual(parent.residual, head, gate).eval()
    return FrozenFullEntryActor(parent.base, residual, parent.get_buffer("action_scale")).eval(), r
