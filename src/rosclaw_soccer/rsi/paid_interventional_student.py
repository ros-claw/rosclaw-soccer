"""Strict warm continual imitation over physically witnessed own-state interventions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_compact_student import (
    FIT_STATUS,
    FrozenCompactResidual,
    compact_config,
    validate_compact_state,
)
from rosclaw_soccer.rsi.paid_compact_support import (
    CompactSupportedHead,
    CompactTeacherSupport,
    nearest_protected_radii,
)
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor, digest
from rosclaw_soccer.rsi.paid_interventional_data import load_interventional_data

STUDENT_STATUS = "PAID_INTERVENTIONAL_DAGGER_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
CONTROL_STATUS = "PAID_INTERVENTIONAL_GEOMETRY_ONLY_CONTROL_NOT_PHYSICALLY_QUALIFIED"
REPLANNED_STATUS = "PAID_REPLANNED_INTERVENTIONAL_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
REPLANNED_CONTROL_STATUS = "PAID_REPLANNED_INTERVENTIONAL_GEOMETRY_CONTROL_NOT_PHYSICALLY_QUALIFIED"


def learner_config(index: int, *, control: bool = False, generation: int = 1) -> dict[str, Any]:
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError("three fixed integer optimization seeds required")
    if type(control) is not bool or control and index != 0:
        raise ValueError("one explicit fixed primary no-update causal control required")
    if type(generation) is not int or generation not in (1, 2):
        raise ValueError("explicit bounded learning generation required")
    result = dict(
        protocol="interventional_dagger.warm_contact.v1",
        seed_index=index,
        seed=2026103100 + index,
        new_steps=2048,
        inherited_steps=1024,
        Adam_total_steps=3072,
        lr=0.0003,
        canonical_teacher_batch=256,
        intervention_teacher_batch=256,
        retention_batch=512,
        active_weight=5.0,
        retention_weight=5.0,
        radius_fraction=0.45,
        active_rad=1e-6,
        support_rule="active_TRAIN_centres_vs_all_inactive_TRAIN_anchors",
        training_teacher_gains="one_raw_head_imitation_including_zero_prefix",
        common_warm_seed=0,
        independent_minibatch_RNG=True,
        random_reinitialization=False,
        warm_Adam_moments=True,
        features=288,
        final_R2_bound_rad=0.2,
        internal_difference_bound_rad=0.4,
        parent_frozen=True,
        base_frozen=True,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
        runtime_case_lookup=False,
    )
    if control:
        result.update(
            protocol="interventional_dagger.geometry_only_control.v1",
            control=True,
            new_steps=0,
            Adam_total_steps=1024,
        )
    if generation == 2:
        result.update(
            generation=2,
            protocol="interventional_dagger.replanned_geometry_control.v2"
            if control
            else "interventional_dagger.replanned_contact.v2",
            seed=2026110100 + index,
            inherited_steps=3072,
            Adam_total_steps=3072 if control else 5120,
        )
    return result


def load_warm_state(
    root: Path, expected: str, data: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact primary learned head source required")
    r = json.loads(rp.read_text())
    cp = Path(r["checkpoint_path"])
    if (
        r["status"] != FIT_STATUS
        or r["config"] != compact_config(0)
        or r["data_hash"] != data["original_hash"]
        or r["body_scene_hash"] != data["body_scene_hash"]
        or r["parent_hash"] != data["parent_hash"]
        or r["parent_checkpoint_hash"] != data["parent_checkpoint_hash"]
        or cp.is_symlink()
        or cp.parent != root
        or cp.stat().st_size > 16000000
        or digest(cp) != r["checkpoint_hash"]
        or digest(dp) != r["declaration_hash"]
    ):
        raise ValueError("bounded exact v3 primary warm head/Adam/body ancestry required")
    s = torch.load(cp, weights_only=True, map_location="cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(compact_config(0)["seed"])
        template = CompactSupportedHead(
            CompactTeacherSupport(torch.zeros(1, 288), torch.ones(1, dtype=torch.float64))
        )
    validate_compact_state(s, r, template.net.state_dict())
    if any(
        s[k] != r[k]
        for k in (
            "data_hash",
            "body_scene_hash",
            "parent_hash",
            "parent_checkpoint_hash",
            "declaration_hash",
        )
    ):
        raise ValueError("warm state identity differs")
    return s, r


def data_support(z: dict[str, np.ndarray]) -> CompactTeacherSupport:
    positive = torch.from_numpy(z["features"][z["active"]])
    negative = torch.from_numpy(z["features"][~z["active"]])
    return CompactTeacherSupport(positive, nearest_protected_radii(positive, negative))


def load_replanned_warm(
    root: Path, expected: str, data: dict[str, Any], policy: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse the exact current student that generated the new teacher prefixes."""
    rp = root / "complete.json"
    if rp.is_symlink() or digest(rp) != expected or expected != data["source_student_hash"]:
        raise ValueError("exact current-prefix source student must provide warm memory")
    r = json.loads(rp.read_text())
    if (
        r["status"] != STUDENT_STATUS
        or r["config"] != learner_config(0)
        or r["data_hash"] != data["previous_hash"]
        or any(
            r[k] != data[k] for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")
        )
    ):
        raise ValueError("previous generation primary 3072-step memory required")
    cp = Path(r["checkpoint_path"])
    # Fully validate its generation-one data, safe state, Adam, geometry and batch1
    # certificate before borrowing weights; physical qualification is not inherited.
    load_interventional_student(
        policy,
        checkpoint=cp,
        checkpoint_hash=r["checkpoint_hash"],
        training_receipt=rp,
        training_receipt_hash=expected,
        scene_hash=data["body_scene_hash"],
    )
    if digest(cp) != r["checkpoint_hash"]:
        raise ValueError("warm checkpoint changed during ancestry validation")
    state = torch.load(cp, weights_only=True, map_location="cpu")
    return dict(model=state["head"], optimizer=state["optimizer"]), r


def learning_certificate(
    head: CompactSupportedHead,
    z: dict[str, np.ndarray],
    original_rows: int,
    warm_net: dict[str, torch.Tensor],
    *,
    control: bool = False,
) -> dict[str, Any]:
    changed = max(float((v - warm_net[k]).abs().max()) for k, v in head.net.state_dict().items())
    if (
        type(control) is not bool
        or not np.isfinite(changed)
        or (changed != 0 if control else changed <= 0)
    ):
        raise ValueError("effective new neural parameter change required")
    counts = dict(active=0, inactive=0, protected=0, canonical_active=0, intervention_active=0)
    errors = dict(canonical=0.0, intervention=0.0)
    with torch.no_grad():
        for i, row in enumerate(z["features"]):
            x = torch.from_numpy(row.copy()).reshape(1, 288)
            p = torch.from_numpy(z["parent_delta"][i].copy()).reshape(1, 29)
            gain = head.support(x)
            pred = head.combine(x, p, gain)
            if not torch.isfinite(pred).all() or (pred.abs() > 0.2).any():
                raise ValueError("finite original-authority prediction required")
            if z["active"][i]:
                if float(gain) != 1:
                    raise ValueError("every active TRAIN centre must open")
                counts["active"] += 1
                key = "canonical" if i < original_rows else "intervention"
                counts[key + "_active"] += 1
                target = torch.from_numpy(z["target"][i].copy()).reshape(1, 29)
                errors[key] += float((pred - p - target).square().sum())
            else:
                if gain.count_nonzero() or not torch.equal(pred, p):
                    raise ValueError("all inactive/old anchors must preserve exact motor memory")
                counts["inactive"] += 1
            counts["protected"] += int(not z["teacher"][i])
    return dict(
        batch_shape=1,
        CPU_support_query_rows=len(z["features"]),
        CPU_neural_head_forward_rows=len(z["features"]),
        counts=counts,
        maximum_head_parameter_change=changed,
        active_correction_MSE=sum(errors.values()) / (29 * counts["active"]),
        canonical_active_MSE=errors["canonical"] / (29 * counts["canonical_active"]),
        intervention_active_MSE=errors["intervention"] / (29 * counts["intervention_active"]),
        all_inactive_predictions_exact_parent=True,
        physical_qualification=False,
        new_neural_weight_effect=not control,
    )


def validate_learner_state(s: dict[str, Any], r: dict[str, Any], warm: dict[str, Any]) -> None:
    control = s.get("config", {}).get("control", False)
    generation = s.get("config", {}).get("generation", 1)
    config = learner_config(
        s.get("config", {}).get("seed_index"), control=control, generation=generation
    )
    status = (
        (REPLANNED_CONTROL_STATUS if control else REPLANNED_STATUS)
        if generation == 2
        else (CONTROL_STATUS if control else STUDENT_STATUS)
    )
    steps = config["new_steps"]
    counts = dict(
        optimizer_updates=steps,
        inherited_optimizer_updates=config["inherited_steps"],
        canonical_forward_rows=256 * steps,
        intervention_forward_rows=256 * steps,
        retention_forward_rows=512 * steps,
        nonzero_gradient_steps=steps,
    )
    if (
        s.get("schema") != f"paid.interventional_dagger_student.v{generation}"
        or s.get("config") != config
        or r.get("config") != config
        or r.get("status") != status
        or s.get("counts") != counts
        or r.get("counts") != counts
        or s.get("learning_certificate") != r.get("learning_certificate")
        or r.get("parent_parameters_in_optimizer") != 0
        or r.get("base_parameters_in_optimizer") != 0
        or r.get("own_native_steps") != 0
        or r.get("Fresh_fit_rows") != 0
        or r.get("DEV_fit_rows") != 0
        or any(
            o.get(k) is not False
            for o in (s, r)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("full finite interventional learning contract required")
    for k in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "warm_hash",
        "body_scene_hash",
        "parent_hash",
        "parent_checkpoint_hash",
    ):
        if not isinstance(s.get(k), str) or not s[k] or s[k] != r.get(k):
            raise ValueError("new learner source/data/body commitment differs")
    initial = {k.removeprefix("net."): v for k, v in warm["model"].items() if k.startswith("net.")}
    values = s.get("head", {})
    saved = s.get("initial_net", {})
    if set(values) != {"net." + k for k in initial} | {"support.centres", "support.radii"} or set(
        saved
    ) != set(initial):
        raise ValueError("exact six neural tensors and support buffers required")
    for k, template in initial.items():
        for t in (values["net." + k], saved[k]):
            if (
                not isinstance(t, torch.Tensor)
                or t.shape != template.shape
                or t.dtype != torch.float32
                or t.device.type != "cpu"
                or t.layout != torch.strided
                or not torch.isfinite(t).all()
            ):
                raise ValueError("finite exact CPU head tensors required")
        if not torch.equal(saved[k], template):
            raise ValueError("warm inherited neural memory changed")
    change = max(float((values["net." + k] - v).abs().max()) for k, v in initial.items())
    cert = s["learning_certificate"]
    if (
        (change != 0 if control else change <= 0)
        or cert.get("maximum_head_parameter_change") != change
        or cert.get("all_inactive_predictions_exact_parent") is not True
        or cert.get("physical_qualification") is not False
        or cert.get("batch_shape") != 1
        or cert.get("new_neural_weight_effect") is not (not control)
    ):
        raise ValueError("actual new weight effect and explicit non-physical scope required")
    opt = s.get("optimizer", {})
    groups = opt.get("param_groups", [])
    slots = opt.get("state", {})
    if (
        len(groups) != 1
        or groups[0].get("params") != list(range(6))
        or groups[0].get("lr") != 0.0003
        or groups[0].get("betas") != (0.9, 0.999)
        or groups[0].get("eps") != 1e-8
        or groups[0].get("weight_decay") != 0
        or any(
            groups[0].get(k) is not False
            for k in ("maximize", "amsgrad", "capturable", "differentiable")
        )
        or set(slots) != set(range(6))
    ):
        raise ValueError("exact inherited six-tensor Adam contract required")
    for i, template in enumerate(initial.values()):
        step = slots[i].get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.numel() != 1
            or float(step) != config["Adam_total_steps"]
        ):
            raise ValueError("exact inherited and actual new Adam steps required")
        for k in ("exp_avg", "exp_avg_sq"):
            t = slots[i].get(k)
            if (
                not isinstance(t, torch.Tensor)
                or t.shape != template.shape
                or t.dtype != torch.float32
                or t.device.type != "cpu"
                or not torch.isfinite(t).all()
                or k == "exp_avg_sq"
                and (t < 0).any()
            ):
                raise ValueError("finite actual Adam moments required")
            if control and not torch.equal(t, warm["optimizer"]["state"][i][k]):
                raise ValueError("geometry-only control cannot modify inherited Adam moments")


def load_interventional_student(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[Any, dict[str, Any]]:
    if (
        checkpoint.is_symlink()
        or training_receipt.is_symlink()
        or checkpoint.parent != training_receipt.parent
        or checkpoint.stat().st_size > 16000000
        or digest(checkpoint) != checkpoint_hash
        or digest(training_receipt) != training_receipt_hash
        or (training_receipt.parent / "failed.json").exists()
    ):
        raise ValueError("exact bounded learned interventional artifacts required")
    r = json.loads(training_receipt.read_text())
    dp = training_receipt.parent / "declaration.json"
    s = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if (
        dp.is_symlink()
        or digest(dp) != s["declaration_hash"]
        or r["checkpoint_hash"] != checkpoint_hash
        or s["body_scene_hash"] != scene_hash
    ):
        raise ValueError("learner declaration/body differs")
    d = json.loads(dp.read_text())
    if any(
        d[k] != s[k]
        for k in (
            "config",
            "source_head",
            "data_hash",
            "warm_hash",
            "body_scene_hash",
            "parent_hash",
            "parent_checkpoint_hash",
        )
    ):
        raise ValueError("exact learner declaration differs")
    if s.get("config", {}).get("generation", 1) == 2:
        from rosclaw_soccer.rsi.paid_replan_data import load_replanned_data

        parent, data, z = load_replanned_data(Path(d["data_root"]), s["data_hash"], policy)
        warm, _ = load_replanned_warm(Path(d["warm_root"]), s["warm_hash"], data, policy)
    else:
        parent, data, z = load_interventional_data(Path(d["data_root"]), s["data_hash"], policy)
        warm, _ = load_warm_state(Path(d["warm_root"]), s["warm_hash"], data)
    validate_learner_state(s, r, warm)
    if any(s[k] != data[k] for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")):
        raise ValueError("interventional ancestry changed")
    with torch.random.fork_rng(devices=[]):
        head = CompactSupportedHead(data_support(z))
    geometry = head.support.state_dict()
    for k, v in geometry.items():
        t = s["head"]["support." + k]
        if (
            not isinstance(t, torch.Tensor)
            or t.dtype != v.dtype
            or t.shape != v.shape
            or t.device.type != "cpu"
            or not torch.allclose(t, v, rtol=1e-12, atol=1e-14)
        ):
            raise ValueError("fixed data-derived support geometry differs")
    head.load_state_dict(s["head"], strict=True)
    initial = {k.removeprefix("net."): v for k, v in warm["model"].items() if k.startswith("net.")}
    if (
        learning_certificate(
            head, z, data["original_rows"], initial, control=s["config"].get("control", False)
        )
        != s["learning_certificate"]
    ):
        raise ValueError("independent actual CPU batch1 learning certificate differs")
    residual = FrozenCompactResidual(parent.residual, head).eval()
    return FrozenFullEntryActor(parent.base, residual, parent.action_scale).eval(), r
