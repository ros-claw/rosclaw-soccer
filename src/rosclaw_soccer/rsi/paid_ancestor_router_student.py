"""Strict TRAIN data, functional retention and safe-load for competence routing."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter

ROUTER_STATUS = "PAID_ANCESTRAL_ROUTER_TRAINED_NOT_PHYSICALLY_QUALIFIED"


def router_config(seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("three predeclared router seeds required")
    return dict(
        protocol="ancestral_competence_router.v1",
        seed_index=seed,
        seed=2026102600 + seed,
        steps=512,
        lr=0.0003,
        batch_rows=512,
        hard_negative_rows=256,
        features=288,
        bound_radians=0.2,
        calibration_margin=0.1,
        positive_core_rad=0.005,
        parent_frozen=True,
        base_frozen=True,
        per_control_routing=True,
        runtime_case_lookup=False,
        Fresh_fit_rows=0,
        alternatives="exact_current_parent_or_exact_R2_zero",
    )


def validate_router_data(
    data: dict[str, Any], declaration: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    if (
        data.get("status") != "PAID_ANCESTRAL_ROUTER_TRAIN_FEATURES_CAUSALLY_RECOMPUTED"
        or declaration.get("schema") != "paid.ancestor_router.TRAIN_features.v1"
        or any(
            o.get("bank") != "train" or o.get("Fresh_fit_rows") != 0 for o in (data, declaration)
        )
        or data.get("source_head") != declaration.get("source_head")
        or data.get("parent_hash") != declaration.get("parent_hash")
        or data.get("parent_root") != declaration.get("parent_root")
        or data.get("own_native_steps") != 0
        or data.get("optimizer_updates") != 0
        or data.get("original_protected_by_kick") != [16, 14, 15]
        or declaration.get("original_protected_by_kick") != [16, 14, 15]
        or data.get("new_train_protected_by_kick") != [0, 19, 12]
        or declaration.get("teacher_cases") != [15, 16, 17, 18]
        or declaration.get("bound_radians") != 0.2
        or declaration.get("runtime_case_lookup") is not False
        or declaration.get("inference_features") != "own_normalized160+own_R2_hidden128"
        or declaration.get("alternatives") != "exact_current_parent_delta_or_exact_R2_zero_delta"
        or any(
            o.get(k) is not False
            for o in (data, declaration)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("exact current-parent TRAIN-only router curriculum required")
    n = data.get("rows")
    if type(n) is not int or not 1 <= n <= 50000:
        raise ValueError("bounded actual TRAIN rows required")
    shapes = dict(
        features=(n, 288),
        parent_delta=(n, 29),
        parent_difference=(n, 29),
        teacher=(n,),
        group=(n,),
        positive_core=(n,),
        protected_nonzero=(n,),
    )
    dtypes = dict(
        features=np.float32,
        parent_delta=np.float32,
        parent_difference=np.float32,
        teacher=np.bool_,
        group=np.int32,
        positive_core=np.bool_,
        protected_nonzero=np.bool_,
    )
    if set(arrays) != set(shapes) or any(
        arrays[k].shape != s or arrays[k].dtype != dtypes[k] or not np.isfinite(arrays[k]).all()
        for k, s in shapes.items()
    ):
        raise ValueError("exact finite numeric router archive required")
    prior, y, teacher = arrays["parent_delta"], arrays["parent_difference"], arrays["teacher"]
    if (
        np.abs(prior).max() > 0.2
        or np.abs(y).max() > 0.40002
        or y[~teacher].any()
        or not np.allclose(y[teacher], -prior[teacher], rtol=0, atol=2e-6)
        or not np.array_equal(
            arrays["positive_core"], teacher & (np.abs(prior).max(axis=1) > 0.005)
        )
        or not np.array_equal(arrays["protected_nonzero"], ~teacher & np.any(prior != 0, axis=1))
        or int(teacher.sum()) != data.get("teacher_rows")
        or int((~teacher).sum()) != data.get("protected_rows")
        or int(arrays["positive_core"].sum()) != data.get("positive_core_rows")
        or int(arrays["protected_nonzero"].sum()) != data.get("protected_nonzero_rows")
        or not arrays["positive_core"].any()
        or not arrays["protected_nonzero"].any()
        or data.get("own_base_feature_rows") != n
        or data.get("own_parent_head_rows") != n
    ):
        raise ValueError("causal R2 labels and functional retention masks required")
    trajectories = data.get("trajectories", [])
    cursor = 0
    seen = set()
    old = [0, 0, 0]
    new = [0, 0, 0]
    teachers = []
    if len(trajectories) != 80:
        raise ValueError("all80 actual trajectories required")
    for i, r in enumerate(trajectories):
        role, case, skill, length = r["role"], r["case"], r["skill"], r["rows"]
        if (
            type(case) is not int
            or type(skill) is not int
            or type(length) is not int
            or not 1 <= length <= 300
            or r["start"] != cursor
            or r["stop"] != cursor + length
            or (role, case, skill) in seen
            or skill not in range(3)
        ):
            raise ValueError("unique complete trajectory groups required")
        seen.add((role, case, skill))
        sl = slice(cursor, cursor + length)
        cursor += length
        if not np.all(arrays["group"][sl] == i) or not np.all(
            teacher[sl] == (role == "ancestor_teacher")
        ):
            raise ValueError("group or teacher role changed")
        if role == "ancestor_teacher":
            if skill != 2 or case not in (15, 16, 17, 18):
                raise ValueError("actual ancestor teachers required")
            teachers.append(case)
        elif role == "original_acquired_success" and case in range(16):
            old[skill] += 1
        elif role == "new_train_acquired_success" and case in range(20) and skill in (1, 2):
            new[skill] += 1
        else:
            raise ValueError("unknown training role")
    if (
        cursor != n
        or old != [16, 14, 15]
        or new != [0, 19, 12]
        or sorted(teachers) != [15, 16, 17, 18]
    ):
        raise ValueError("complete acquired frontier required")


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_router_data(
    root: Path, expected: str, policy: Path
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    from rosclaw_soccer.rsi.paid_progressive_student import TRAINED_STATUS, load_progressive_student

    rp, dp = root / "complete.json", root / "declaration.json"
    if rp.is_symlink() or digest(rp) != expected or (root / "failed.json").exists():
        raise ValueError("exact closed router curation required")
    d, decl = [json.loads(f.read_text()) for f in (rp, dp)]
    archive = Path(d["feature_path"])
    if (
        archive.is_symlink()
        or archive.parent != root
        or digest(archive) != d["feature_hash"]
        or digest(dp) != d["declaration_hash"]
    ):
        raise ValueError("confined committed numeric archive required")
    with np.load(archive, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    validate_router_data(d, decl, arrays)
    parent_root = Path(d["parent_root"])
    receipt_path = parent_root / "complete.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        digest(receipt_path) != d["parent_hash"]
        or receipt["status"] != TRAINED_STATUS
        or receipt["checkpoint_hash"] != d["parent_checkpoint_hash"]
        or receipt["config"]["seed_index"] != 0
        or receipt["body_scene_hash"] != d["body_scene_hash"]
    ):
        raise ValueError("exact declared compound parent only required")
    actor, _ = load_progressive_student(
        policy,
        checkpoint=Path(receipt["checkpoint_path"]),
        checkpoint_hash=receipt["checkpoint_hash"],
        training_receipt=receipt_path,
        training_receipt_hash=d["parent_hash"],
        scene_hash=d["body_scene_hash"],
    )
    actor.requires_grad_(False).eval()
    with torch.no_grad():
        for x, prior in zip(arrays["features"], arrays["parent_delta"], strict=True):
            if not np.array_equal(
                actor.residual(torch.from_numpy(x.copy()).reshape(1, 288))[0].numpy(), prior
            ):
                raise ValueError("own runtime-shape parent predictions differ")
    return actor, d, arrays


def calibrate_router_gate(gate: torch.nn.Sequential, negative: torch.Tensor) -> dict[str, Any]:
    import copy

    if (
        negative.ndim != 2
        or negative.shape[1] != 288
        or not len(negative)
        or not torch.isfinite(negative).all()
    ):
        raise ValueError("complete finite meaningful-negative support required")
    output = gate[-1]
    if not isinstance(output, torch.nn.Linear) or output.bias is None:
        raise ValueError("scalar output bias required")
    x = negative.detach().cpu()
    with torch.no_grad():
        cpu = copy.deepcopy(gate).cpu().eval()
        before = max(float(cpu(row.reshape(1, 288))[0, 0]) for row in x)
        shift = max(0.0, before + 0.1001)
        output.bias.sub_(shift)
        cpu = copy.deepcopy(gate).cpu().eval()
        after = max(float(cpu(row.reshape(1, 288))[0, 0]) for row in x)
        if not math.isfinite(after) or after > -0.1:
            raise ValueError("runtime-shape CPU threshold calibration failed")
    return dict(
        maximum_negative_before_CPU=before,
        maximum_negative_after_CPU=after,
        output_bias_shift=shift,
        margin=0.1,
        negative_rows=len(x),
        CPU_gate_rows=2 * len(x),
        batch_shape=1,
        only_output_bias_shifted=True,
        not_physical_qualification=True,
    )


def router_certificate(
    model: AncestralResidualRouter, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    protected = ~arrays["teacher"]
    negative = arrays["protected_nonzero"]
    core = arrays["positive_core"]
    maximum = -math.inf
    opened = positive_open = checks = 0
    with torch.no_grad():
        for i, row in enumerate(arrays["features"]):
            x = torch.from_numpy(row.copy()).reshape(1, 288)
            prior = model.parent(x)
            pred, logit = model.forward_with_gate(x)
            if not np.array_equal(prior[0].numpy(), arrays["parent_delta"][i]):
                raise ValueError("frozen parent changed")
            score = float(logit[0, 0])
            if not math.isfinite(score):
                raise ValueError("nonfinite router inference")
            if protected[i]:
                checks += 1
                opened += int(score >= 0)
                if not torch.equal(pred, prior):
                    raise ValueError("acquired-success functional output changed")
            if negative[i]:
                maximum = max(maximum, score)
            if core[i]:
                positive_open += int(score >= 0)
    if maximum > -0.1:
        raise ValueError("meaningful negative routing not rejected")
    return dict(
        maximum_meaningful_negative_logit_CPU=maximum,
        all_protected_predictions_exact_parent=True,
        protected_rows=checks,
        meaningful_negative_rows=int(negative.sum()),
        protected_gate_open_rows=opened,
        positive_core_rows=int(core.sum()),
        positive_core_open_rows=positive_open,
        CPU_parent_validation_rows=len(protected),
        CPU_parent_forward_rows=2 * len(protected),
        CPU_router_rows=len(protected),
        batch_shape=1,
    )


def validate_router_state(
    state: dict[str, Any], receipt: dict[str, Any], template: dict[str, torch.Tensor]
) -> None:
    config = router_config(state.get("config", {}).get("seed_index"))
    cert = state.get("retention_certificate", {})
    counts = state.get("counts", {})
    maximum = cert.get("maximum_meaningful_negative_logit_CPU")
    if (
        state.get("schema") != "paid.ancestral_router_student.v1"
        or state.get("config") != config
        or receipt.get("config") != config
        or receipt.get("status") != ROUTER_STATUS
        or receipt.get("counts") != counts
        or counts.get("optimizer_updates") != 512
        or counts.get("positive_rows") != 262144
        or counts.get("negative_rows") != 262144
        or cert != receipt.get("retention_certificate")
        or cert.get("all_protected_predictions_exact_parent") is not True
        or cert.get("batch_shape") != 1
        or type(maximum) not in (int, float)
        or not math.isfinite(maximum)
        or maximum > -0.1
        or state.get("calibration") != receipt.get("calibration")
        or state.get("calibration", {}).get("maximum_negative_after_CPU") != maximum
        or state.get("calibration", {}).get("batch_shape") != 1
        or state.get("calibration", {}).get("only_output_bias_shifted") is not True
        or receipt.get("parent_parameters_in_optimizer") != 0
        or receipt.get("base_parameters_in_optimizer") != 0
        or receipt.get("parent_parameters_exact_unchanged") is not True
        or receipt.get("Fresh_fit_rows") != 0
        or receipt.get("own_native_steps") != 0
        or any(
            o.get(k) is not False
            for o in (state, receipt)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete functional-retention router training contract required")
    for key in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "feature_hash",
        "parent_hash",
        "parent_checkpoint_hash",
        "body_scene_hash",
    ):
        if not isinstance(state.get(key), str) or not state[key] or state[key] != receipt.get(key):
            raise ValueError("router identity differs")
    for namespace in ("gate", "initial_gate"):
        values = state.get(namespace, {})
        if set(values) != set(template):
            raise ValueError("exact four gate tensors required")
        for key, v in values.items():
            if (
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.layout != torch.strided
                or v.shape != template[key].shape
                or not torch.isfinite(v).all()
            ):
                raise ValueError("finite CPU gate tensor required")
    if any(not torch.equal(state["initial_gate"][k], v) for k, v in template.items()):
        raise ValueError("seeded fresh gate required")
    optimizer = state.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    slots = optimizer.get("state", {})
    if (
        len(groups) != 1
        or groups[0].get("params") != list(range(4))
        or groups[0].get("lr") != 0.0003
        or set(slots) != set(range(4))
    ):
        raise ValueError("only four gate parameters may enter Adam")
    for i, v in enumerate(template.values()):
        step = slots[i].get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.numel() != 1
            or float(step) != 512
        ):
            raise ValueError("actual512 Adam updates required")
        for key in ("exp_avg", "exp_avg_sq"):
            a = slots[i].get(key)
            if (
                not isinstance(a, torch.Tensor)
                or a.device.type != "cpu"
                or a.dtype != torch.float32
                or a.shape != v.shape
                or not torch.isfinite(a).all()
                or (key == "exp_avg_sq" and (a < 0).any())
            ):
                raise ValueError("finite gate-only moments required")


class FrozenAncestralRouterActor(torch.nn.Module):
    def __init__(self, base: Any, router: AncestralResidualRouter, scale: torch.Tensor) -> None:
        super().__init__()
        if (
            scale.shape != (29,)
            or scale.dtype != torch.float32
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("original positive29 scale required")
        self.base, self.residual = base.requires_grad_(False).eval(), router
        self.register_buffer("action_scale", scale.detach().clone())

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        nominal, after, hidden = self.base.forward_with_features(obs, state)
        features = torch.cat([(obs - self.base.mean) / self.base.denominator, hidden], dim=-1)
        delta = self.residual(features.reshape(-1, 288)).reshape(*nominal.shape)
        return nominal + delta / self.action_scale, after


def load_ancestral_router_student(
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
        or checkpoint.stat().st_size > 16000000
        or digest(checkpoint) != checkpoint_hash
        or digest(training_receipt) != training_receipt_hash
        or (training_receipt.parent / "failed.json").exists()
    ):
        raise ValueError("bounded exact safe-load artifacts required")
    receipt = json.loads(training_receipt.read_text())
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    config = router_config(state.get("config", {}).get("seed_index"))
    dp = training_receipt.parent / "declaration.json"
    if (
        digest(dp) != state["declaration_hash"]
        or receipt["checkpoint_hash"] != checkpoint_hash
        or state["body_scene_hash"] != scene_hash
    ):
        raise ValueError("checkpoint/body/training commitment differs")
    declaration = json.loads(dp.read_text())
    if declaration["config"] != config or declaration["data_hash"] != state["data_hash"]:
        raise ValueError("training declaration differs")
    parent, data, arrays = load_router_data(
        Path(declaration["data_root"]), state["data_hash"], policy
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        model = AncestralResidualRouter(parent.residual)
    validate_router_state(state, receipt, model.gate.state_dict())
    if any(
        state[k] != data[k]
        for k in ("parent_hash", "parent_checkpoint_hash", "body_scene_hash", "feature_hash")
    ):
        raise ValueError("router data lineage changed")
    model.gate.load_state_dict(state["gate"], strict=True)
    model.eval()
    if router_certificate(model, arrays) != state["retention_certificate"]:
        raise ValueError("own CPU functional certificate recomputation differs")
    return FrozenAncestralRouterActor(parent.base, model, parent.action_scale).eval(), receipt
