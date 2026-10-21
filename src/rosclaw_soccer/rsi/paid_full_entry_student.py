"""Strict offline full-entry learner admission; no physical promotion or hardware."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
from rosclaw_soccer.rsi.paid_full_entry_contact import FullEntryContactResidual

FULL_ENTRY_STATUS = "PAID_FULL_ENTRY_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
IDENTITIES = (
    "source_head",
    "declaration_hash",
    "data_hash",
    "feature_hash",
    "parent_hash",
    "parent_checkpoint_hash",
    "body_scene_hash",
)
FIRST_R2_CASES = [1, 2, 3, 7, 9, 11, 13, 15, 16, 17, 18, 19]


def full_entry_config(seed: int, *, trajectory_support: bool = False) -> dict[str, Any]:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("three predeclared integer seeds required")
    if type(trajectory_support) is not bool:
        raise ValueError("explicit support ablation flag required")
    result = dict(
        protocol="full-entry-contact.v1",
        seed_index=seed,
        seed=2026102800 + seed,
        gate_steps=512,
        head_steps=1024,
        lr=0.0003,
        batch_rows=512,
        hard_negative_rows=256,
        positive_core_rad=0.005,
        active_rad=1e-6,
        active_weight=5.0,
        retention_weight=5.0,
        calibration_margin=0.1,
        features=288,
        final_bound_rad=0.2,
        internal_difference_bound_rad=0.4,
        parent_frozen=True,
        base_frozen=True,
        runtime_case_lookup=False,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
    )
    if trajectory_support:
        result.update(
            protocol="full-entry-contact.trajectory-support.v2",
            trajectory_support=True,
            gate_positive_support="whole_successful_teacher_trajectory",
            gate_negative_support="all_protected_acquired_states",
            head_minibatches="exact_v1_PCG_stream_after512_gate_iterations",
        )
    return result


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def head_rng_after_v1_gate(seed: int, core: np.ndarray, negative_rows: int) -> np.random.Generator:
    """Exact v1 minibatch state, independent of the ablation's new gate draws."""
    if (
        type(seed) is not int
        or seed not in (2026102800, 2026102801, 2026102802)
        or core.ndim != 1
        or not len(core)
        or core.dtype.kind not in "iu"
        or type(negative_rows) is not int
        or negative_rows < 256
    ):
        raise ValueError("fixed original seed and actual v1 support sizes required")
    rng = np.random.default_rng(seed)
    for _ in range(512):
        rng.choice(core, 512, replace=True)
        rng.integers(0, negative_rows, size=256)
    return rng


def validate_full_entry_data(
    d: dict[str, Any], declaration: dict[str, Any], z: dict[str, np.ndarray]
) -> None:
    if (
        d.get("status") != "PAID_FULL_ENTRY_TRAIN_FEATURES_CAUSALLY_RECOMPUTED"
        or declaration.get("schema") != "paid.full_entry.TRAIN_features.v1"
        or any(
            o.get("bank") != "train" or o.get("Fresh_fit_rows") != 0 or o.get("DEV_fit_rows") != 0
            for o in (d, declaration)
        )
        or any(
            d.get(k) != declaration.get(k)
            for k in (
                "source_head",
                "parent_root",
                "parent_hash",
                "old_protected_by_kick",
                "new_protected_by_kick",
            )
        )
        or d.get("old_protected_by_kick") != [16, 14, 15]
        or d.get("new_protected_by_kick") != [0, 19, 16]
        or d.get("optimizer_updates") != 0
        or d.get("own_native_steps") != 0
        or declaration.get("teacher_seed_index") != 0
        or declaration.get("novel_first_teacher_cases") != [4, 5]
        or declaration.get("canonical_R2_first_cases") != FIRST_R2_CASES
        or declaration.get("expected_trajectories") != 94
        or declaration.get("first_state_has_one_canonical_teacher") is not True
        or declaration.get("final_R2_relative_bound_rad") != 0.2
        or declaration.get("internal_difference_bound_rad") != 0.4
        or declaration.get("runtime_case_lookup") is not False
        or declaration.get("features") != "own_normalized160+own_R2_hidden128"
        or declaration.get("target") != "radian_difference_from_frozen_router_parent"
        or any(
            o.get(k) is not False
            for o in (d, declaration)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete TRAIN-only full-entry causal curriculum required")
    n = d.get("rows")
    if type(n) is not int or not 1 <= n <= 50000:
        raise ValueError("bounded actual feature rows required")
    shapes = dict(
        features=(n, 288),
        target=(n, 29),
        parent_delta=(n, 29),
        teacher=(n,),
        group=(n,),
        active=(n,),
    )
    types = dict(
        features=np.float32,
        target=np.float32,
        parent_delta=np.float32,
        teacher=np.bool_,
        group=np.int32,
        active=np.bool_,
    )
    if set(z) != set(shapes) or any(
        z[k].shape != v or z[k].dtype != types[k] or not np.isfinite(z[k]).all()
        for k, v in shapes.items()
    ):
        raise ValueError("exact finite numeric full-entry archive required")
    y, prior, teacher, active = (z[k] for k in ("target", "parent_delta", "teacher", "active"))
    if (
        np.abs(y).max() > 0.40002
        or np.abs(prior).max() > 0.2
        or np.abs(y + prior).max() > 0.20002
        or y[~teacher].any()
        or not np.array_equal(active, np.abs(y).max(axis=1) > 1e-6)
        or np.any(active & ~teacher)
        or not active.any()
        or not (~active).any()
        or int(teacher.sum()) != d.get("teacher_rows")
        or int((~teacher).sum()) != d.get("protected_rows")
        or int(active.sum()) != d.get("active_rows")
        or int((~active).sum()) != d.get("negative_support_rows")
        or int((np.abs(y).max(axis=1) > 0.005).sum()) != d.get("positive_core_rows")
        or d.get("positive_core_rows", 0) < 1
        or d.get("own_base_feature_rows") != n
        or d.get("own_parent_head_rows") != n
    ):
        raise ValueError("causal bounded differences and complete exact-zero protection required")
    trajectories = d.get("trajectories", [])
    if len(trajectories) != 94:
        raise ValueError("all94 actual trajectories required")
    cursor = 0
    old, new = [0, 0, 0], [0, 0, 0]
    novel, canonical = [], []
    seen = set()
    for i, r in enumerate(trajectories):
        role, case, skill, length = (r[k] for k in ("role", "case", "skill", "rows"))
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
            raise ValueError("unique complete ordered trajectory groups required")
        seen.add((role, case, skill))
        sl = slice(cursor, cursor + length)
        cursor += length
        is_teacher = role in ("novel_first_cartesian_teacher", "canonical_first_R2_teacher")
        if not np.all(z["group"][sl] == i) or not np.all(teacher[sl] == is_teacher):
            raise ValueError("own group/role changed")
        expected = "base" if is_teacher or skill == 0 else "parent"
        if r["recorded_policy"] != expected:
            raise ValueError("actual recorded motor identity required")
        if role == "novel_first_cartesian_teacher" and case in (4, 5) and skill == 0:
            novel.append(case)
        elif role == "canonical_first_R2_teacher" and case in FIRST_R2_CASES and skill == 0:
            canonical.append(case)
            if not np.allclose(y[sl], -prior[sl], rtol=0, atol=2e-6):
                raise ValueError("canonical first teacher must be actual R2 motor")
        elif role == "original_acquired_success" and case in range(16):
            old[skill] += 1
        elif role == "new_train_acquired_success" and case in range(20) and skill in (1, 2):
            new[skill] += 1
        else:
            raise ValueError("unknown or noncanonical training role")
    if (
        cursor != n
        or old != [16, 14, 15]
        or new != [0, 19, 16]
        or sorted(novel) != [4, 5]
        or sorted(canonical) != FIRST_R2_CASES
    ):
        raise ValueError("complete nonconflicting full-entry curriculum required")


def load_full_entry_data(
    root: Path, expected: str, policy: Path
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    from rosclaw_soccer.rsi.paid_ancestor_router_student import (
        ROUTER_STATUS,
        load_ancestral_router_student,
    )

    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed full-entry curation required")
    d, decl = [json.loads(f.read_text()) for f in (rp, dp)]
    f = Path(d["feature_path"])
    if (
        f.is_symlink()
        or f.parent != root
        or digest(f) != d["feature_hash"]
        or digest(dp) != d["declaration_hash"]
    ):
        raise ValueError("exact confined archive/declaration required")
    with np.load(f, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    validate_full_entry_data(d, decl, arrays)
    parent_root = Path(d["parent_root"])
    prp = parent_root / "complete.json"
    pr = json.loads(prp.read_text())
    if (
        digest(prp) != d["parent_hash"]
        or pr["status"] != ROUTER_STATUS
        or pr["config"]["seed_index"] != 0
        or pr["checkpoint_hash"] != d["parent_checkpoint_hash"]
        or pr["body_scene_hash"] != d["body_scene_hash"]
    ):
        raise ValueError("only exact frozen primary router ancestry admitted")
    actor, _ = load_ancestral_router_student(
        policy,
        checkpoint=Path(pr["checkpoint_path"]),
        checkpoint_hash=pr["checkpoint_hash"],
        training_receipt=prp,
        training_receipt_hash=d["parent_hash"],
        scene_hash=d["body_scene_hash"],
    )
    actor.requires_grad_(False).eval()
    with torch.no_grad():
        for x, p in zip(arrays["features"], arrays["parent_delta"], strict=True):
            if not np.array_equal(
                actor.residual(torch.from_numpy(x.copy()).reshape(1, 288))[0].numpy(), p
            ):
                raise ValueError("rowwise own routed-parent prediction differs")
    return actor, d, arrays


def full_entry_certificate(
    model: FullEntryContactResidual,
    arrays: dict[str, np.ndarray],
    *,
    trajectory_support: bool = False,
) -> dict[str, Any]:
    if type(trajectory_support) is not bool:
        raise ValueError("explicit gate support protocol required")
    negative = ~arrays["teacher"] if trajectory_support else ~arrays["active"]
    maximum = -math.inf
    protected = opened = core_open = teacher_open = 0
    mse = 0.0
    with torch.no_grad():
        for i, row in enumerate(arrays["features"]):
            x = torch.from_numpy(row.copy()).reshape(1, 288)
            prior = model.parent(x)
            pred = model(x)
            score = float(model.plastic.gate(x)[0, 0])
            if not np.array_equal(prior[0].numpy(), arrays["parent_delta"][i]) or not math.isfinite(
                score
            ):
                raise ValueError("frozen own parent or finite gate identity differs")
            if negative[i]:
                maximum = max(maximum, score)
            if arrays["teacher"][i]:
                teacher_open += int(score >= 0)
            if not arrays["teacher"][i]:
                protected += 1
                if not torch.equal(pred, prior):
                    raise ValueError("protected acquired motor prediction changed")
            if arrays["active"][i]:
                opened += int(score >= 0)
                core_open += int(score >= 0 and np.abs(arrays["target"][i]).max() > 0.005)
                mse += float(
                    (pred[0] - prior[0] - torch.from_numpy(arrays["target"][i])).square().mean()
                )
    if maximum > -0.1:
        raise ValueError("every declared negative-support gate must reject in runtime shape")
    return dict(
        maximum_negative_logit_CPU=maximum,
        all_protected_predictions_exact_parent=True,
        protected_rows=protected,
        negative_support_rows=int(negative.sum()),
        active_rows=int(arrays["active"].sum()),
        active_gate_open_rows=opened,
        positive_core_open_rows=core_open,
        active_correction_MSE=mse / int(arrays["active"].sum()),
        batch_shape=1,
        CPU_parent_forward_rows=2 * len(arrays["features"]),
        CPU_plastic_gate_forward_rows=2 * len(arrays["features"]),
        CPU_plastic_net_forward_rows=len(arrays["features"]),
        **(
            dict(teacher_gate_open_rows=teacher_open, gate_support="whole_teacher_trajectory")
            if trajectory_support
            else {}
        ),
    )


def validate_full_entry_state(
    state: dict[str, Any], receipt: dict[str, Any], initial: dict[str, torch.Tensor]
) -> None:
    config = full_entry_config(
        state.get("config", {}).get("seed_index"),
        trajectory_support=state.get("config", {}).get("trajectory_support", False),
    )
    cert = state.get("retention_certificate", {})
    counts = state.get("counts", {})
    maximum = cert.get("maximum_negative_logit_CPU")
    if (
        state.get("schema") != "paid.full_entry_student.v1"
        or state.get("config") != config
        or receipt.get("config") != config
        or receipt.get("status") != FULL_ENTRY_STATUS
        or receipt.get("retention_certificate") != cert
        or receipt.get("counts") != counts
        or counts.get("optimizer_updates") != 1536
        or counts.get("gate_optimizer_updates") != 512
        or counts.get("head_optimizer_updates") != 1024
        or counts.get("positive_forward_rows") != 524288
        or counts.get("retention_forward_rows") != 524288
        or cert.get("all_protected_predictions_exact_parent") is not True
        or cert.get("batch_shape") != 1
        or type(maximum) not in (float, int)
        or not math.isfinite(maximum)
        or maximum > -0.1
        or state.get("calibration") != receipt.get("calibration")
        or state.get("calibration", {}).get("maximum_negative_after_CPU") != maximum
        or state.get("calibration", {}).get("only_output_bias_shifted") is not True
        or state.get("calibration", {}).get("batch_shape") != 1
        or receipt.get("parent_parameters_in_optimizer") != 0
        or receipt.get("base_parameters_in_optimizer") != 0
        or receipt.get("parent_parameters_exact_unchanged") is not True
        or receipt.get("Fresh_fit_rows") != 0
        or receipt.get("DEV_fit_rows") != 0
        or receipt.get("own_native_steps") != 0
        or any(
            o.get(k) is not False
            for o in (state, receipt)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete staged full-entry training contract required")
    for k in IDENTITIES:
        if not isinstance(state.get(k), str) or not state[k] or state[k] != receipt.get(k):
            raise ValueError("full-entry ancestry/source commitment differs")
    for name in ("plastic", "initial_plastic"):
        values = state.get(name, {})
        if set(values) != set(initial):
            raise ValueError("exact10 plastic tensors required")
        for k, v in values.items():
            if (
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.layout != torch.strided
                or v.shape != initial[k].shape
                or not torch.isfinite(v).all()
            ):
                raise ValueError("finite CPU plastic tensors required")
    if any(not torch.equal(state["initial_plastic"][k], v) for k, v in initial.items()):
        raise ValueError("exact seeded plastic initialization required")
    for namespace, prefix, steps in (("gate_optimizer", "gate.", 512), ("optimizer", "net.", 1024)):
        values = [v for k, v in initial.items() if k.startswith(prefix)]
        opt = state.get(namespace, {})
        groups, slots = opt.get("param_groups", []), opt.get("state", {})
        if (
            len(groups) != 1
            or groups[0].get("params") != list(range(len(values)))
            or groups[0].get("lr") != 0.0003
            or set(slots) != set(range(len(values)))
        ):
            raise ValueError("exact separated gate/net Adam parameter sets required")
        for i, v in enumerate(values):
            step = slots[i].get("step")
            if (
                not isinstance(step, torch.Tensor)
                or step.device.type != "cpu"
                or step.numel() != 1
                or float(step) != steps
            ):
                raise ValueError("actual declared Adam update count required")
            for k in ("exp_avg", "exp_avg_sq"):
                t = slots[i].get(k)
                if (
                    not isinstance(t, torch.Tensor)
                    or t.device.type != "cpu"
                    or t.dtype != torch.float32
                    or t.shape != v.shape
                    or not torch.isfinite(t).all()
                    or (k == "exp_avg_sq" and (t < 0).any())
                ):
                    raise ValueError("finite separated Adam moments required")


class FrozenFullEntryActor(torch.nn.Module):
    def __init__(self, base: Any, residual: torch.nn.Module, scale: torch.Tensor) -> None:
        super().__init__()
        if (
            scale.shape != (29,)
            or scale.dtype != torch.float32
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("original positive29 action scales required")
        self.base, self.residual = base.requires_grad_(False).eval(), residual
        self.register_buffer("action_scale", scale.detach().clone())

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        nominal, after, hidden = self.base.forward_with_features(obs, state)
        features = torch.cat([(obs - self.base.mean) / self.base.denominator, hidden], dim=-1)
        delta = self.residual(features.reshape(-1, 288)).reshape(*nominal.shape)
        return nominal + delta / self.action_scale, after


def load_full_entry_student(
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
        raise ValueError("exact bounded safe-load artifacts required")
    r = json.loads(training_receipt.read_text())
    s = torch.load(checkpoint, weights_only=True, map_location="cpu")
    config = full_entry_config(
        s.get("config", {}).get("seed_index"),
        trajectory_support=s.get("config", {}).get("trajectory_support", False),
    )
    dp = training_receipt.parent / "declaration.json"
    if (
        digest(dp) != s["declaration_hash"]
        or r["checkpoint_hash"] != checkpoint_hash
        or s["body_scene_hash"] != scene_hash
    ):
        raise ValueError("training/checkpoint/body commitment differs")
    d = json.loads(dp.read_text())
    if d["config"] != config or d["data_hash"] != s["data_hash"]:
        raise ValueError("full-entry training declaration differs")
    parent, data, arrays = load_full_entry_data(Path(d["data_root"]), s["data_hash"], policy)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        plastic = DecoupledContactResidual()
    validate_full_entry_state(s, r, plastic.state_dict())
    if any(
        s[k] != data[k]
        for k in ("parent_hash", "parent_checkpoint_hash", "body_scene_hash", "feature_hash")
    ):
        raise ValueError("full-entry data ancestry changed")
    plastic.load_state_dict(s["plastic"], strict=True)
    model = FullEntryContactResidual(parent.residual, plastic).eval()
    if (
        full_entry_certificate(
            model, arrays, trajectory_support=config.get("trajectory_support", False)
        )
        != s["retention_certificate"]
    ):
        raise ValueError("own runtime-shape CPU retention recomputation differs")
    return FrozenFullEntryActor(parent.base, model, parent.get_buffer("action_scale")).eval(), r
