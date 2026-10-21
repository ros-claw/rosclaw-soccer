"""Support-only causal ablation over an unchanged, actually learned v3 motor head.

No new SGD, expanded authority, ID routing, or silent inheritance of physical
qualification. Inactive teacher prefixes join the exact rejection anchors.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_compact_student import (
    FIT_STATUS,
    FrozenCompactResidual,
    audit_compact_batch1,
    compact_config,
    load_compact_fit,
)
from rosclaw_soccer.rsi.paid_compact_support import (
    CompactSupportedHead,
    CompactTeacherSupport,
    nearest_protected_radii,
)
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor, digest

ACTIVE_STATUS = "PAID_ACTIVE_COMPACT_SUPPORT_AUDITED_NOT_PHYSICALLY_QUALIFIED"


def active_compact_config(seed: int) -> dict[str, Any]:
    source = compact_config(seed)
    return dict(
        protocol="full-entry-contact.active-support.v4",
        seed_index=seed,
        source_head_protocol=source["protocol"],
        support_positive="all201_active_TRAIN_teacher_states",
        support_negative="all21721_inactive_and_protected_TRAIN_states",
        radius_fraction=0.45,
        head_weights_exact_source=True,
        new_optimizer_updates=0,
        inherited_optimizer_updates=1024,
        final_R2_bound_rad=0.2,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
        runtime_case_lookup=False,
    )


def apply_active_support(head: CompactSupportedHead, z: dict[str, np.ndarray]) -> None:
    if int(z["active"].sum()) != 201 or int((~z["active"]).sum()) != 21721:
        raise ValueError("exact original active/inactive TRAIN partition required")
    centres = torch.from_numpy(z["features"][z["active"]])
    negative = torch.from_numpy(z["features"][~z["active"]])
    radii = nearest_protected_radii(centres, negative)
    head.support = CompactTeacherSupport(centres, radii)


def validate_active_payload(
    state: dict[str, Any], receipt: dict[str, Any], source: dict[str, Any]
) -> None:
    config = active_compact_config(source.get("config", {}).get("seed_index"))
    if (
        source.get("status") != FIT_STATUS
        or source.get("config") != compact_config(config["seed_index"])
        or state.get("schema") != "paid.active_compact_support.v1"
        or state.get("config") != config
        or receipt.get("config") != config
        or receipt.get("status") != ACTIVE_STATUS
        or receipt.get("optimizer_updates") != 0
        or receipt.get("inherited_optimizer_updates") != 1024
        or receipt.get("own_native_steps") != 0
        or receipt.get("Fresh_fit_rows") != 0
        or receipt.get("DEV_fit_rows") != 0
        or receipt.get("head_weights_exact_source") is not True
        or any(
            o.get(k) is not False
            for o in (state, receipt)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("support-only ablation over exact original trained head required")
    for k in ("data_hash", "body_scene_hash", "source_head", "source_fit_root", "source_fit_hash"):
        if not isinstance(state.get(k), str) or not state[k] or state[k] != receipt.get(k):
            raise ValueError("active support ancestry commitment differs")
    if any(state[k] != source[k] for k in ("data_hash", "body_scene_hash")):
        raise ValueError("unchanged TRAIN data and body required")
    support = state.get("support", {})
    if set(support) != {"centres", "radii"}:
        raise ValueError("only two support buffers may be requalified")
    for k, shape, dtype in (
        ("centres", (201, 288), torch.float32),
        ("radii", (201,), torch.float64),
    ):
        t = support[k]
        if (
            not isinstance(t, torch.Tensor)
            or t.shape != shape
            or t.dtype != dtype
            or t.device.type != "cpu"
            or not torch.isfinite(t).all()
            or k == "radii"
            and not (t > 0).all()
        ):
            raise ValueError("exact finite CPU active support buffers required")


def load_active_compact_student(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[Any, dict[str, Any]]:
    if (
        training_receipt.is_symlink()
        or checkpoint.is_symlink()
        or checkpoint.parent != training_receipt.parent
        or checkpoint.stat().st_size > 2000000
        or digest(checkpoint) != checkpoint_hash
        or digest(training_receipt) != training_receipt_hash
        or (training_receipt.parent / "failed.json").exists()
    ):
        raise ValueError("exact bounded active-support research artifacts required")
    r = json.loads(training_receipt.read_text())
    dp = training_receipt.parent / "declaration.json"
    if (
        dp.is_symlink()
        or digest(dp) != r["declaration_hash"]
        or r["checkpoint_path"] != str(checkpoint)
        or r["checkpoint_hash"] != checkpoint_hash
        or r["body_scene_hash"] != scene_hash
    ):
        raise ValueError("active support declaration/checkpoint/body differs")
    s = torch.load(checkpoint, weights_only=True, map_location="cpu")
    d = json.loads(dp.read_text())
    for k in (
        "config",
        "source_head",
        "source_fit_root",
        "source_fit_hash",
        "data_hash",
        "body_scene_hash",
    ):
        if d.get(k) != s.get(k) or d.get(k) != r.get(k):
            raise ValueError("active support declaration identity differs")
    source_root = Path(s["source_fit_root"])
    # Only v3 parents admitted: never an arbitrary recursive support chain.
    srp = source_root / "complete.json"
    if digest(srp) != s["source_fit_hash"]:
        raise ValueError("exact frozen learned-head source required")
    source = json.loads(srp.read_text())
    validate_active_payload(s, r, source)
    parent, head, source, z = load_compact_fit(source_root, s["source_fit_hash"], policy)
    before = {k: v.clone() for k, v in head.net.state_dict().items()}
    apply_active_support(head, z)
    if (
        not torch.equal(s["support"]["centres"], head.support.centres)
        or not torch.equal(s["support"]["radii"], head.support.radii)
        or any(not torch.equal(v, before[k]) for k, v in head.net.state_dict().items())
    ):
        raise ValueError("only CPU-rederived support may change; neural weights must not")
    own = audit_compact_batch1(head, source, z, active_only=True)
    if any(
        own[k] != r["batch1_certificate"].get(k) for k in own if k != "latency_quantiles_seconds"
    ):
        raise ValueError("active support own runtime-shape certificate differs")
    residual = FrozenCompactResidual(parent.residual, head).eval()
    return FrozenFullEntryActor(parent.base, residual, parent.action_scale).eval(), r
