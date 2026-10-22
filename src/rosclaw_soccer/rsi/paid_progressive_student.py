"""Strict SIM research admission for acquired-parent progressive adaptation."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual

TRAINED_STATUS = "PAID_PROGRESSIVE_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
DATA_STATUS = "PAID_PROGRESSIVE_TRAIN_PARENT_FEATURES_CAUSALLY_RECOMPUTED"


def progressive_config(seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("one of three predeclared progressive seeds required")
    return dict(
        protocol="acquired-parent-progressive-contact-v1",
        seed_index=seed,
        seed=2026102400 + seed,
        lr=0.0003,
        gate_training_steps=512,
        head_training_steps=1024,
        batch_rows=512,
        features=288,
        original_observations=160,
        bound_radians=0.2,
        internal_correction_bound_radians=0.4,
        parent_frozen=True,
        R2_frozen=True,
        gate_frozen_during_head_training=True,
        gate_positive_core_rad=0.005,
        calibration_margin=0.1,
        parent_seed_index=0,
        teacher_cases=[8, 9],
        protected_successes_by_kick=[16, 14, 13],
        DEV_episodes_fit=0,
        runtime_case_lookup=False,
        memory_handoff="reset_for_kick_carry_own_base_for_stand",
    )


def validate_progressive_data(data: dict[str, Any], declaration: dict[str, Any]) -> None:
    if (
        data.get("status") != DATA_STATUS
        or declaration.get("schema") != "soccer.progressive_contact_curriculum.v1"
        or declaration.get("teacher_cases") != [8, 9]
        or declaration.get("final_R2_relative_bound_radians") != 0.2
        or declaration.get("internal_difference_bound_radians") != 0.4
        or declaration.get("features") != "own_normalized160+own_causal128"
        or declaration.get("runtime_case_lookup") is not False
        or data.get("own_native_steps") != 0
        or data.get("optimizer_updates") != 0
        or any(
            obj.get(k) is not False
            for obj in (data, declaration)
            for k in ("hardware_authorized", "promotion_authorized")
        )
        or any(
            obj.get("bank") != "train"
            or obj.get("DEV_episodes_fit") != 0
            or obj.get("protected_successes_by_kick") != [16, 14, 13]
            for obj in (data, declaration)
        )
        or data.get("parent_hash") != declaration.get("parent_hash")
        or data.get("source_head") != declaration.get("source_head")
    ):
        raise ValueError("exact TRAIN-only progressive acquired-success curriculum required")
    rows = data.get("trajectories", [])
    if len(rows) != 45:
        raise ValueError("all43 acquired successes and two teachers required")
    seen: set[tuple[str, int, int]] = set()
    cursor, teacher_rows, protected_rows = 0, 0, 0
    counts = [0, 0, 0]
    teachers = []
    for row in rows:
        role, case, skill = row.get("role"), row.get("case"), row.get("skill")
        n = row.get("rows")
        key = (role, case, skill)
        if (
            type(case) is not int
            or case not in range(16)
            or type(skill) is not int
            or skill not in range(3)
            or type(n) is not int
            or not 1 <= n <= 300
            or role not in ("teacher", "protected_acquired_success")
            or key in seen
            or row.get("start") != cursor
            or row.get("stop") != cursor + n
        ):
            raise ValueError("unique complete acquired-success trajectory ranges required")
        seen.add(key)
        cursor += n
        if role == "teacher":
            if skill != 2:
                raise ValueError("third-kick teachers only")
            teachers.append(case)
            teacher_rows += n
        else:
            counts[skill] += 1
            protected_rows += n
    active = data.get("active_rows")
    if (
        sorted(teachers) != [8, 9]
        or counts != [16, 14, 13]
        or cursor != data.get("rows")
        or cursor != data.get("CPU_base_feature_recompute_rows")
        or cursor != data.get("CPU_parent_head_recompute_rows")
        or teacher_rows != data.get("teacher_rows")
        or protected_rows != data.get("protected_acquired_rows")
        or type(active) is not int
        or not 0 < active <= teacher_rows
        or data.get("negative_support_rows") != cursor - active
    ):
        raise ValueError("actual own-parent data accounting differs")


def validate_progressive_arrays(data: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    n = data["rows"]
    shapes = dict(
        features=(n, 288),
        target=(n, 29),
        parent_delta=(n, 29),
        teacher=(n,),
        active=(n,),
        group=(n,),
    )
    dtypes = dict(
        features=np.float32,
        target=np.float32,
        parent_delta=np.float32,
        teacher=np.bool_,
        active=np.bool_,
        group=np.int32,
    )
    if set(arrays) != set(shapes) or any(
        arrays[k].shape != shape or arrays[k].dtype != dtypes[k] or not np.isfinite(arrays[k]).all()
        for k, shape in shapes.items()
    ):
        raise ValueError("exact finite progressive feature archive required")
    x, y, prior, teacher, active, group = (arrays[k] for k in shapes)
    if (
        np.abs(y).max() > 0.40002
        or np.abs(prior).max() > 0.2
        or np.abs(prior + y).max() > 0.20003
        or y[~teacher].any()
        or not np.array_equal(active, np.abs(y).max(axis=1) > 1e-6)
        or int(active.sum()) != data["active_rows"]
        or int(teacher.sum()) != data["teacher_rows"]
    ):
        raise ValueError("true acquired-zero targets and unchanged final authority required")
    for i, row in enumerate(data["trajectories"]):
        sl = slice(row["start"], row["stop"])
        if not np.all(group[sl] == i) or not np.all(teacher[sl] == (row["role"] == "teacher")):
            raise ValueError("feature groups or roles differ from committed trajectories")


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_progressive_data(
    root: Path, expected_hash: str, policy: Path
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    """Recompute parent's actual CPU predictions, not just trust saved labels."""
    from rosclaw_soccer.rsi.paid_recovery_proof import unique_proof_path
    from rosclaw_soccer.rsi.paid_successor_student_actor import load_successor_student

    rp, dp = root / "complete.json", root / "declaration.json"
    if rp.is_symlink() or _digest(rp) != expected_hash or (root / "failed.json").exists():
        raise ValueError("exact closed progressive curation required")
    data, declaration = json.loads(rp.read_text()), json.loads(dp.read_text())
    validate_progressive_data(data, declaration)
    if data["declaration_hash"] != _digest(dp):
        raise ValueError("curation declaration changed")
    archive = Path(data["feature_path"])
    if archive.is_symlink() or archive.parent != root or _digest(archive) != data["feature_hash"]:
        raise ValueError("exact confined progressive feature archive required")
    with np.load(archive, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    validate_progressive_arrays(data, arrays)
    pins = data["input_file_hashes"]
    parent_receipt = unique_proof_path(pins, data["parent_hash"])
    parent_training = json.loads(parent_receipt.read_text())
    if (
        _digest(parent_receipt) != data["parent_hash"]
        or parent_training["config"]["protocol"] != "successor-contact-student-v2"
        or parent_training["config"]["seed_index"] != 0
        or parent_training["checkpoint_hash"] != data["parent_checkpoint_hash"]
        or parent_training["base_wrapper_checkpoint_hash"] != data["base_wrapper_checkpoint_hash"]
    ):
        raise ValueError("exact predeclared learned parent required")
    actor, _ = load_successor_student(
        policy,
        checkpoint=unique_proof_path(pins, data["parent_checkpoint_hash"]),
        checkpoint_hash=data["parent_checkpoint_hash"],
        training_receipt=parent_receipt,
        training_receipt_hash=data["parent_hash"],
        scene_hash=data["body_scene_hash"],
    )
    actor.requires_grad_(False).eval()
    # Curator computes individual rows. Preserve that exact CPU execution shape;
    # a large batch can choose different floating point kernels.
    with torch.no_grad():
        for x, prior in zip(arrays["features"], arrays["parent_delta"], strict=True):
            actual = actor.residual(torch.from_numpy(x.copy()).reshape(1, 288))[0]
            if not np.array_equal(actual.numpy(), prior):
                raise ValueError("own CPU parent recomputation differs from saved parent delta")
    return actor, data, arrays


def validate_progressive_state(
    state: dict[str, Any], receipt: dict[str, Any], template: dict[str, torch.Tensor]
) -> None:
    config = progressive_config(state.get("config", {}).get("seed_index"))
    counts = state.get("counts", {})
    certificate = state.get("retention_certificate", {})
    maximum = certificate.get("maximum_negative_logit_CPU")
    if (
        state.get("schema") != "soccer.progressive_contact_student.v1"
        or state.get("config") != config
        or receipt.get("config") != config
        or receipt.get("status") != TRAINED_STATUS
        or receipt.get("counts") != counts
        or counts.get("optimizer_updates") != 1536
        or counts.get("gate_optimizer_updates") != 512
        or counts.get("head_optimizer_updates") != 1024
        or counts.get("positive_forward_rows") != 524288
        or counts.get("retention_forward_rows") != 524288
        or receipt.get("parent_parameters_in_optimizer") != 0
        or receipt.get("base_parameters_in_optimizer") != 0
        or receipt.get("parent_parameters_exact_unchanged") is not True
        or receipt.get("DEV_episodes_fit") != 0
        or receipt.get("own_native_steps") != 0
        or certificate != receipt.get("retention_certificate")
        or certificate.get("all_negative_predictions_exact_parent") is not True
        or certificate.get("all_negative_corrections_exact_zero") is not True
        or type(maximum) not in (int, float)
        or not math.isfinite(maximum)
        or maximum > -0.1
        or state.get("gate_calibration") != receipt.get("gate_calibration")
        or state.get("gate_calibration", {}).get("maximum_negative_after_CPU") != maximum
        or state.get("gate_calibration", {}).get("only_output_bias_shifted") is not True
        or state.get("gate_calibration", {}).get("all_negative_gate_rejected") is not True
        or any(
            obj.get(k) is not False
            for obj in (state, receipt)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("exact frozen learned-parent progressive training contract required")
    for key in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "features_hash",
        "parent_hash",
        "parent_checkpoint_hash",
        "body_scene_hash",
        "base_wrapper_checkpoint_hash",
        "base_wrapper_training_receipt_hash",
    ):
        if not isinstance(state.get(key), str) or not state[key] or state[key] != receipt.get(key):
            raise ValueError("progressive training identity differs: " + key)
    for namespace in ("plastic", "initial_plastic"):
        values = state.get(namespace, {})
        if set(values) != set(template):
            raise ValueError("only exact ten plastic tensor keys allowed")
        for key, expected in template.items():
            value = values[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.layout != torch.strided
                or value.shape != expected.shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError("finite CPU plastic tensor required")
    if any(not torch.equal(state["initial_plastic"][k], v) for k, v in template.items()):
        raise ValueError("exact seeded fresh plastic head required")
    for name, prefix, steps in (("optimizer", "net.", 1024), ("gate_optimizer", "gate.", 512)):
        tensors = [v for k, v in template.items() if k.startswith(prefix)]
        optimizer = state.get(name, {})
        groups, slots = optimizer.get("param_groups", []), optimizer.get("state", {})
        if (
            len(groups) != 1
            or groups[0].get("lr") != 0.0003
            or groups[0].get("params") != list(range(len(tensors)))
            or set(slots) != set(range(len(tensors)))
        ):
            raise ValueError("only separate declared plastic parameters may enter Adam")
        for i, expected in enumerate(tensors):
            step = slots[i].get("step")
            if (
                not isinstance(step, torch.Tensor)
                or step.device.type != "cpu"
                or step.numel() != 1
                or float(step) != steps
            ):
                raise ValueError("actual full plastic Adam update counts required")
            for key in ("exp_avg", "exp_avg_sq"):
                v = slots[i].get(key)
                if (
                    not isinstance(v, torch.Tensor)
                    or v.device.type != "cpu"
                    or v.dtype != torch.float32
                    or v.shape != expected.shape
                    or not torch.isfinite(v).all()
                    or (key == "exp_avg_sq" and (v < 0).any())
                ):
                    raise ValueError("finite plastic-only Adam moments required")


class FrozenProgressiveActor(torch.nn.Module):
    def __init__(
        self, base: Any, residual: ProgressiveContactResidual, scale: torch.Tensor
    ) -> None:
        super().__init__()
        if (
            scale.shape != (29,)
            or scale.dtype != torch.float32
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("original positive policy-order29 scale required")
        self.base, self.residual = base.requires_grad_(False).eval(), residual
        self.register_buffer("action_scale", scale.detach().clone())

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        nominal, after, hidden = self.base.forward_with_features(obs, state)
        features = torch.cat([(obs - self.base.mean) / self.base.denominator, hidden], dim=-1)
        delta = self.residual(features.reshape(-1, 288)).reshape(*nominal.shape)
        return nominal + delta / self.action_scale, after


def load_progressive_student(
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
        or _digest(checkpoint) != checkpoint_hash
        or _digest(training_receipt) != training_receipt_hash
        or (training_receipt.parent / "failed.json").exists()
    ):
        raise ValueError("bounded exact progressive safe-load artifacts required")
    receipt = json.loads(training_receipt.read_text())
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    config = progressive_config(state.get("config", {}).get("seed_index"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        plastic = DecoupledContactResidual()
    validate_progressive_state(state, receipt, plastic.state_dict())
    dp = training_receipt.parent / "declaration.json"
    if (
        _digest(dp) != state["declaration_hash"]
        or receipt["checkpoint_hash"] != checkpoint_hash
        or state["body_scene_hash"] != scene_hash
    ):
        raise ValueError("exact training/body/checkpoint commitment required")
    declaration = json.loads(dp.read_text())
    if declaration["config"] != config or declaration["data_hash"] != state["data_hash"]:
        raise ValueError("progressive training declaration differs")
    parent, data, arrays = load_progressive_data(
        Path(declaration["data_root"]), state["data_hash"], policy
    )
    if any(
        state[k] != data[k]
        for k in (
            "parent_hash",
            "parent_checkpoint_hash",
            "body_scene_hash",
            "base_wrapper_checkpoint_hash",
            "base_wrapper_training_receipt_hash",
        )
    ):
        raise ValueError("exact immutable learned-parent lineage required")
    if state["features_hash"] != data["feature_hash"]:
        raise ValueError("progressive numeric features changed")
    plastic.load_state_dict(state["plastic"], strict=True)
    model = ProgressiveContactResidual(parent.residual, plastic).eval()
    with torch.no_grad():
        x = torch.from_numpy(arrays["features"])
        negative = torch.from_numpy(~arrays["active"])
        logits = plastic.gate(x)
        prior = parent.residual(x)
        prediction = model(x)
        if float(logits[negative].max()) != state["retention_certificate"][
            "maximum_negative_logit_CPU"
        ] or not torch.equal(prediction[negative], prior[negative]):
            raise ValueError("own CPU exact learned-parent retention recomputation failed")
    actor = FrozenProgressiveActor(parent.base, model, parent.action_scale).eval()
    return actor, receipt


def load_paid_student(policy: Path, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Closed receipt dispatch; old student schemas keep their strict old loader."""
    from rosclaw_soccer.rsi.paid_successor_student_actor import load_successor_student

    path = kwargs["training_receipt"]
    if _digest(path) != kwargs["training_receipt_hash"]:
        raise ValueError("exact student receipt commitment required before dispatch")
    status = json.loads(path.read_text()).get("status")
    from rosclaw_soccer.rsi.paid_ancestor_router_student import (
        ROUTER_STATUS,
        load_ancestral_router_student,
    )

    if status == ROUTER_STATUS:
        return load_ancestral_router_student(policy, **kwargs)
    from rosclaw_soccer.rsi.paid_full_entry_student import (
        FULL_ENTRY_STATUS,
        load_full_entry_student,
    )

    if status == FULL_ENTRY_STATUS:
        return load_full_entry_student(policy, **kwargs)
    from rosclaw_soccer.rsi.paid_compact_student import AUDITED_STATUS, load_compact_student

    if status == AUDITED_STATUS:
        return load_compact_student(policy, **kwargs)
    from rosclaw_soccer.rsi.paid_active_compact_student import (
        ACTIVE_STATUS,
        load_active_compact_student,
    )

    if status == ACTIVE_STATUS:
        return load_active_compact_student(policy, **kwargs)
    from rosclaw_soccer.rsi.paid_interventional_student import (
        CONTROL_STATUS,
        PRECISION_CONTROL_STATUS,
        PRECISION_STATUS,
        REPLANNED_CONTROL_STATUS,
        REPLANNED_STATUS,
        STUDENT_STATUS,
        load_interventional_student,
    )

    if status in (
        STUDENT_STATUS,
        CONTROL_STATUS,
        REPLANNED_STATUS,
        REPLANNED_CONTROL_STATUS,
        PRECISION_STATUS,
        PRECISION_CONTROL_STATUS,
    ):
        return load_interventional_student(policy, **kwargs)
    from rosclaw_soccer.rsi.paid_cardinal_memory import (
        CARDINAL_CONTROL_STATUS,
        CARDINAL_STATUS,
        GROWING_CONTROL_STATUS,
        GROWING_STATUS,
        load_cardinal_student,
    )

    if status in (CARDINAL_STATUS, CARDINAL_CONTROL_STATUS, GROWING_STATUS, GROWING_CONTROL_STATUS):
        return load_cardinal_student(policy, **kwargs)
    from rosclaw_soccer.rsi.paid_replay_neural_student import (
        AUDITED_STATUS as REPLAY_AUDITED_STATUS,
    )
    from rosclaw_soccer.rsi.paid_replay_neural_student import load_replay_student

    if status == REPLAY_AUDITED_STATUS:
        return load_replay_student(policy, **kwargs)
    loader = load_progressive_student if status == TRAINED_STATUS else load_successor_student
    return loader(policy, **kwargs)
