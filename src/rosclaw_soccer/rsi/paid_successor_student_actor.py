"""Strict SIM-only admission for the independently trained successor student."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual, FrozenContactResidualActor
from rosclaw_soccer.rsi.paid_successor_features import (
    staged_successor_config,
    successor_student_config,
)


def _config(state: dict[str, Any]) -> dict[str, Any]:
    seed = state.get("config", {}).get("seed_index")
    return (
        staged_successor_config(seed)
        if state.get("schema") == "soccer.successor_contact_student.v2"
        else successor_student_config(seed)
    )


def validate_successor_state(
    state: dict[str, Any], receipt: dict[str, Any], template: dict[str, torch.Tensor]
) -> None:
    config = _config(state)
    staged = config["protocol"].endswith("v2")
    counts = state.get("counts", {})
    calls, evaluations = counts.get("projection_calls"), counts.get("projection_gate_evaluations")
    hist = state.get("projection_histogram", {})
    allowed = {str(x) for x in [0.0, 1.0] + [2.0 ** (-k) for k in range(1, 9)]}
    certificate = state.get("retention_certificate", {})
    maximum = certificate.get("maximum_negative_logit_CPU")
    if (
        state.get("schema")
        not in ("soccer.successor_contact_student.v1", "soccer.successor_contact_student.v2")
        or state.get("config") != config
        or receipt.get("config") != config
        or receipt.get("status") != "PAID_SUCCESSOR_NEURAL_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
        or receipt.get("counts") != counts
        or receipt.get("projection_histogram") != hist
        or receipt.get("retention_certificate") != certificate
        or counts.get("optimizer_updates") != (1536 if staged else 1024)
        or calls != (0 if staged else 1024)
        or counts.get("positive_forward_rows") != 524288
        or counts.get("retention_forward_rows") != 524288
        or type(evaluations) is not int
        or not (evaluations == 0 if staged else 1024 <= evaluations <= 10240)
        or not isinstance(hist, dict)
        or not set(hist) <= allowed
        or any(type(x) is not int or x < 0 for x in hist.values())
        or sum(hist.values()) != (0 if staged else 1024)
        or type(maximum) not in (int, float)
        or not math.isfinite(maximum)
        or maximum > -0.1 + 1e-5
        or certificate.get("all_negative_predictions_exact_zero") is not True
        or receipt.get("base_parameters_in_optimizer") != 0
        or receipt.get("DEV_episodes_fit") != 0
        or receipt.get("own_native_steps") != 0
        or any(
            obj.get(key) is not False
            for obj in (state, receipt)
            for key in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError(
            "exact complete frozen-base successor training/retention contract required"
        )
    for name in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "features_hash",
        "base_wrapper_checkpoint_hash",
        "base_wrapper_training_receipt_hash",
        "body_scene_hash",
    ):
        if state.get(name) != receipt.get(name):
            raise ValueError("successor training identity differs: " + name)
    for namespace in ("residual", "initial_residual"):
        tensors = state.get(namespace, {})
        if not isinstance(tensors, dict) or set(tensors) != set(template):
            raise ValueError("exact ten plastic tensor keys required")
        for key, expected in template.items():
            value = tensors[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.layout != torch.strided
                or value.shape != expected.shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError("finite CPU plastic tensor required: " + key)
    if any(
        not torch.equal(state["initial_residual"][key], value) for key, value in template.items()
    ):
        raise ValueError("exact declared-seed zero-head initialization required")
    plans = [(state.get("optimizer", {}), list(template.values()), 1024)]
    if staged:
        calibration = state.get("gate_calibration", {})
        if (
            calibration != receipt.get("gate_calibration")
            or calibration.get("all_negative_gate_rejected") is not True
            or calibration.get("only_output_bias_shifted") is not True
            or calibration.get("maximum_negative_after_CPU") != maximum
            or calibration.get("margin") != 0.1
            or counts.get("head_optimizer_updates") != 1024
            or counts.get("gate_optimizer_updates") != 512
        ):
            raise ValueError(
                "explicit staged gate calibration and independent optimizer work required"
            )
        plans = [
            (
                state.get("optimizer", {}),
                [v for k, v in template.items() if k.startswith("net.")],
                1024,
            ),
            (
                state.get("gate_optimizer", {}),
                [v for k, v in template.items() if k.startswith("gate.")],
                512,
            ),
        ]
    for optimizer, tensors, steps in plans:
        groups = optimizer.get("param_groups", [])
        if (
            len(groups) != 1
            or groups[0].get("params") != list(range(len(tensors)))
            or groups[0].get("lr") != config["lr"]
            or set(optimizer.get("state", {})) != set(range(len(tensors)))
        ):
            raise ValueError("optimizer may contain only declared plastic tensors")
        for i, expected in enumerate(tensors):
            slot = optimizer["state"][i]
            step = slot.get("step")
            if (
                not isinstance(step, torch.Tensor)
                or step.device.type != "cpu"
                or step.numel() != 1
                or float(step) != steps
            ):
                raise ValueError("actual complete Adam step count per plastic parameter required")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = slot.get(key)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32
                    or tensor.shape != expected.shape
                    or not torch.isfinite(tensor).all()
                    or (key == "exp_avg_sq" and (tensor < 0).any())
                ):
                    raise ValueError("finite bounded plastic-only Adam moments required")


def load_successor_student(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
    from rosclaw_soccer.rsi.paid_recovery_proof import unique_proof_path

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    if (
        checkpoint.is_symlink()
        or training_receipt.is_symlink()
        or checkpoint.stat().st_size > 16000000
        or digest(checkpoint) != checkpoint_hash
        or digest(training_receipt) != training_receipt_hash
        or (training_receipt.parent / "failed.json").exists()
    ):
        raise ValueError("bounded exact safe-load successor artifacts required")
    receipt = json.loads(training_receipt.read_text())
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    config = _config(state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        residual = ContactResidual()
    validate_successor_state(state, receipt, residual.state_dict())
    if receipt["checkpoint_hash"] != checkpoint_hash or state["body_scene_hash"] != scene_hash:
        raise ValueError("exact trained body/checkpoint binding required")
    dp = training_receipt.parent / "declaration.json"
    if digest(dp) != state["declaration_hash"]:
        raise ValueError("committed training declaration required")
    declaration = json.loads(dp.read_text())
    pins = receipt["input_file_hashes"]
    data_path = Path(declaration["data_root"]) / "complete.json"
    if pins.get(str(data_path)) != state["data_hash"] or digest(data_path) != state["data_hash"]:
        raise ValueError("exact causal TRAIN feature curation required")
    data = json.loads(data_path.read_text())
    from rosclaw_soccer.rsi.paid_successor_curriculum import validate_train_curriculum

    validate_train_curriculum(data, json.loads((data_path.parent / "declaration.json").read_text()))
    archive = Path(data["feature_path"])
    if (
        data["DEV_episodes_fit"] != 0
        or data["protected_successes_by_kick"] != [16, 14, 6]
        or data["feature_hash"] != state["features_hash"]
        or digest(archive) != state["features_hash"]
        or data["base_wrapper_checkpoint_hash"] != state["base_wrapper_checkpoint_hash"]
        or declaration["config"] != config
        or declaration["negative_support_rows"] != data["negative_support_rows"]
        or state["counts"]["projection_forward_rows"]
        != data["negative_support_rows"] * state["counts"]["projection_gate_evaluations"]
    ):
        raise ValueError("actual bounded TRAIN-only feature/projection identity required")
    residual.load_state_dict(state["residual"], strict=True)
    residual.eval()
    with np.load(archive, allow_pickle=False) as z, torch.no_grad():
        features = torch.from_numpy(z["features"].copy())
        active = torch.from_numpy(z["active"].copy())
        logits = residual.gate(features)
        maximum = float(logits[~active].max())
        if (
            maximum != state["retention_certificate"]["maximum_negative_logit_CPU"]
            or residual(features)[~active].count_nonzero()
        ):
            raise ValueError("own CPU gate/zero-retention recomputation differs")
    base_cp = unique_proof_path(pins, state["base_wrapper_checkpoint_hash"])
    base_rp = unique_proof_path(pins, state["base_wrapper_training_receipt_hash"])
    wrapper, _ = load_contact_residual_actor(
        policy,
        checkpoint=base_cp,
        checkpoint_hash=state["base_wrapper_checkpoint_hash"],
        training_receipt=base_rp,
        training_receipt_hash=state["base_wrapper_training_receipt_hash"],
        scene_hash=scene_hash,
    )
    scale = wrapper.action_scale
    if not isinstance(scale, torch.Tensor):
        raise ValueError("original actor29 policy-order tensor scale required")
    actor = FrozenContactResidualActor(wrapper.base, residual, scale).eval()
    return actor, receipt
