"""Independent compact-fit verification. Offline artifacts remain unadmitted."""

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_compact_support import (
    CompactSupportedHead,
    CompactTeacherSupport,
    nearest_protected_radii,
)
from rosclaw_soccer.rsi.paid_full_entry_student import digest, load_full_entry_data

FIT_STATUS = "PAID_COMPACT_SUPPORT_OFFLINE_FIT_ONLY_NOT_ADMITTED"
AUDITED_STATUS = "PAID_COMPACT_SUPPORT_BATCH1_AUDITED_NOT_PHYSICALLY_QUALIFIED"


def compact_config(index: int) -> dict[str, Any]:
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError("one of three preregistered compact seeds required")
    return dict(
        protocol="full-entry-contact.compact-support.v3",
        seed_index=index,
        seed=2026102800 + index,
        head_steps=1024,
        gate_steps=0,
        lr=0.0003,
        batch_rows=512,
        active_weight=5.0,
        retention_weight=5.0,
        radius_fraction=0.45,
        radius_distance="direct_float64_euclidean_own288",
        head_minibatches="exact_v1_PCG_stream_after512_gate_iterations",
        final_R2_bound_rad=0.2,
        correction_bound_rad=0.4,
        parent_frozen=True,
        base_frozen=True,
        runtime_case_lookup=False,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
    )


def validate_compact_state(
    s: dict[str, Any], r: dict[str, Any], initial: dict[str, torch.Tensor]
) -> None:
    config = compact_config(s.get("config", {}).get("seed_index"))
    if (
        s.get("schema") != "paid.compact_support.fit_only.v1"
        or s.get("config") != config
        or r.get("config") != config
        or r.get("status") != FIT_STATUS
        or s.get("counts")
        != dict(optimizer_updates=1024, positive_forward_rows=524288, retention_forward_rows=524288)
        or r.get("counts") != s.get("counts")
        or r.get("retention_certificate") != s.get("retention_certificate")
        or r.get("parent_parameters_in_optimizer") != 0
        or r.get("base_parameters_in_optimizer") != 0
        or r.get("Fresh_fit_rows") != 0
        or r.get("DEV_fit_rows") != 0
        or r.get("own_native_steps") != 0
        or any(
            o.get(k) is not False
            for o in (s, r)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete offline-only compact training contract required")
    for k in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "feature_hash",
        "parent_hash",
        "parent_checkpoint_hash",
        "body_scene_hash",
    ):
        if not isinstance(s.get(k), str) or not s[k] or s[k] != r.get(k):
            raise ValueError("compact training identity differs")
    values, original = s.get("model", {}), s.get("initial_net", {})
    if set(values) != {"net." + k for k in initial} | {"support.centres", "support.radii"} or set(
        original
    ) != set(initial):
        raise ValueError("exact six head tensors and two support buffers required")
    for k, template in initial.items():
        for v in (values["net." + k], original[k]):
            if (
                not isinstance(v, torch.Tensor)
                or v.shape != template.shape
                or v.dtype != torch.float32
                or v.device.type != "cpu"
                or v.layout != torch.strided
                or not torch.isfinite(v).all()
            ):
                raise ValueError("exact finite CPU neural tensor required")
        if not torch.equal(original[k], template):
            raise ValueError("seeded initial head differs")
    changed = max(float((values["net." + k] - initial[k]).abs().max()) for k in initial)
    c = s["retention_certificate"]
    if (
        c.get("maximum_head_parameter_change") != changed
        or changed <= 0
        or c.get("nonzero_gradient_steps") != 1024
        or c.get("maximum_protected_gain_CPU") != 0
        or c.get("runtime_batch1_not_yet_certified") is not True
        or c.get("physical_qualification") is not False
    ):
        raise ValueError("effective parameter learning, not counters alone, required")
    opt = s.get("optimizer", {})
    groups, slots = opt.get("param_groups", []), opt.get("state", {})
    if (
        len(groups) != 1
        or groups[0].get("params") != list(range(6))
        or groups[0].get("lr") != 0.0003
        or groups[0].get("weight_decay") != 0
        or groups[0].get("betas") != (0.9, 0.999)
        or groups[0].get("eps") != 1e-8
        or set(slots) != set(range(6))
    ):
        raise ValueError("exact isolated six-tensor Adam required")
    for i, template in enumerate(initial.values()):
        v = slots[i].get("step")
        if (
            not isinstance(v, torch.Tensor)
            or v.numel() != 1
            or v.device.type != "cpu"
            or float(v) != 1024
        ):
            raise ValueError("actual 1024 Adam steps required")
        for k in ("exp_avg", "exp_avg_sq"):
            t = slots[i].get(k)
            if (
                not isinstance(t, torch.Tensor)
                or t.dtype != torch.float32
                or t.device.type != "cpu"
                or t.shape != template.shape
                or not torch.isfinite(t).all()
                or (k == "exp_avg_sq" and (t < 0).any())
            ):
                raise ValueError("finite CPU Adam moments required")


def load_compact_fit(
    root: Path, expected: str, policy: Path
) -> tuple[Any, CompactSupportedHead, dict[str, Any], dict[str, np.ndarray]]:
    """Safe research readback, deliberately not a generic runtime actor loader."""
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed compact fit required")
    r = json.loads(rp.read_text())
    cp = Path(r["checkpoint_path"])
    if (
        cp.parent != root
        or cp.is_symlink()
        or cp.stat().st_size > 16000000
        or digest(cp) != r["checkpoint_hash"]
        or digest(dp) != r["declaration_hash"]
    ):
        raise ValueError("confined bounded checkpoint/declaration commitment required")
    s = torch.load(cp, weights_only=True, map_location="cpu")
    config = compact_config(s.get("config", {}).get("seed_index"))
    decl = json.loads(dp.read_text())
    if any(
        decl.get(k) != s.get(k)
        for k in (
            "config",
            "source_head",
            "data_hash",
            "feature_hash",
            "parent_hash",
            "parent_checkpoint_hash",
            "body_scene_hash",
        )
    ):
        raise ValueError("compact declaration identity differs")
    parent, data, z = load_full_entry_data(Path(decl["data_root"]), s["data_hash"], policy)
    if any(
        data[k] != s.get(k)
        for k in ("feature_hash", "parent_hash", "parent_checkpoint_hash", "body_scene_hash")
    ):
        raise ValueError("exact frozen parent and full-entry TRAIN ancestry required")
    centres = torch.from_numpy(z["features"][z["teacher"]])
    radii = nearest_protected_radii(centres, torch.from_numpy(z["features"][~z["teacher"]]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        model = CompactSupportedHead(CompactTeacherSupport(centres, radii))
    validate_compact_state(s, r, model.net.state_dict())
    if (
        not torch.equal(s["model"]["support.centres"], centres)
        or s["model"]["support.centres"].dtype != torch.float32
        or s["model"]["support.centres"].device.type != "cpu"
        or s["model"]["support.radii"].dtype != torch.float64
        or s["model"]["support.radii"].device.type != "cpu"
        or s["model"]["support.radii"].shape != radii.shape
        or not torch.allclose(s["model"]["support.radii"], radii, rtol=1e-12, atol=1e-14)
    ):
        raise ValueError("CPU independently derived support geometry differs")
    model.load_state_dict(s["model"], strict=True)
    return parent, model.requires_grad_(False).eval(), r, z


def audit_compact_batch1(
    model: CompactSupportedHead, receipt: dict[str, Any], z: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Check actual inference shape, not training gains or CPU batch16 outputs."""
    protected = opened = active_rows = 0
    error = 0.0
    timings = []
    with torch.no_grad():
        for i, row in enumerate(z["features"]):
            x = torch.from_numpy(row.copy()).reshape(1, 288)
            prior = torch.from_numpy(z["parent_delta"][i].copy()).reshape(1, 29)
            t0 = time.perf_counter()
            gain = model.support(x)
            pred = model.combine(x, prior, gain)
            timings.append(time.perf_counter() - t0)
            if not torch.isfinite(pred).all() or (pred.abs() > 0.2).any():
                raise ValueError("finite original-authority runtime output required")
            if z["teacher"][i]:
                if float(gain) != 1.0:
                    raise ValueError("all teacher centres must open in runtime batch1")
                opened += 1
            else:
                if gain.count_nonzero() or not torch.equal(pred, prior):
                    raise ValueError("all old anchors must preserve exact parent in batch1")
                protected += 1
            if z["active"][i]:
                target = torch.from_numpy(z["target"][i].copy()).reshape(1, 29)
                error += float((pred - prior - target).square().sum())
                active_rows += 1
    mse = error / (active_rows * 29)
    c = receipt["retention_certificate"]
    if (
        protected != c["protected_rows_exact_parent"]
        or opened != c["teacher_centres_gain_exact_one"]
        or not math.isclose(mse, c["active_correction_MSE"], rel_tol=1e-5, abs_tol=1e-10)
        or not mse < c["no_op_active_correction_MSE"]
    ):
        raise ValueError("complete improved fit and CPU batch1/batch16 agreement required")
    return dict(
        batch_shape=1,
        CPU_support_query_rows=len(z["features"]),
        CPU_neural_head_forward_rows=len(z["features"]),
        protected_rows_exact_parent=protected,
        teacher_centres_gain_exact_one=opened,
        active_rows=active_rows,
        active_correction_MSE=mse,
        latency_quantiles_seconds=np.quantile(timings, [0.5, 0.95, 0.99, 1.0]).tolist(),
        latency_includes="compact_support_and_neural_head_only_not_base_or_physics",
        realtime_guaranteed=False,
        closed_loop_physics_qualified=False,
    )


class FrozenCompactResidual(torch.nn.Module):
    bound_radians = 0.2

    def __init__(self, parent: torch.nn.Module, head: CompactSupportedHead) -> None:
        from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter

        super().__init__()
        if type(parent) is not AncestralResidualRouter or type(head) is not CompactSupportedHead:
            raise ValueError("exact frozen parent and verified compact head required")
        self.parent = parent.requires_grad_(False).eval()
        self.head = head.requires_grad_(False).eval()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head.combine(features, self.parent(features))


def load_compact_student(
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
        or (training_receipt.parent / "failed.json").exists()
        or checkpoint.is_symlink()
        or digest(checkpoint) != checkpoint_hash
    ):
        raise ValueError("exact closed batch1 verification required")
    v = json.loads(training_receipt.read_text())
    dp = training_receipt.parent / "declaration.json"
    if (
        v.get("status") != AUDITED_STATUS
        or dp.is_symlink()
        or digest(dp) != v.get("declaration_hash")
        or v.get("checkpoint_path") != str(checkpoint)
        or v.get("checkpoint_hash") != checkpoint_hash
        or v.get("body_scene_hash") != scene_hash
        or v.get("hardware_authorized") is not False
        or v.get("promotion_authorized") is not False
    ):
        raise ValueError("simulation-only verified fit/body/checkpoint commitment required")
    d = json.loads(dp.read_text())
    if d.get("fit_root") != v.get("fit_root") or d.get("fit_hash") != v.get("fit_hash"):
        raise ValueError("declared exact offline fit required")
    parent, head, r, z = load_compact_fit(Path(v["fit_root"]), v["fit_hash"], policy)
    if any(
        v.get(k) != r.get(k)
        for k in ("checkpoint_hash", "checkpoint_path", "config", "body_scene_hash", "data_hash")
    ):
        raise ValueError("verified compact model identity changed")
    # Re-evaluate all safety/fit checks when loading. Wall-time measurements are
    # observational and cannot be expected to reproduce bitwise.
    own = audit_compact_batch1(head, r, z)
    saved = v["batch1_certificate"]
    for k in own:
        if k != "latency_quantiles_seconds" and own[k] != saved.get(k):
            raise ValueError("independent batch1 certificate differs")
    residual = FrozenCompactResidual(parent.residual, head).eval()
    return FrozenFullEntryActor(parent.base, residual, parent.get_buffer("action_scale")).eval(), v
