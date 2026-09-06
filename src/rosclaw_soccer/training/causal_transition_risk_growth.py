"""Four-GPU risk-sensitive training for causal skill hand-offs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.causal_skill_transition_risk import (
    RISK_CANDIDATE_POLICY_FRAMES,
    CausalTransitionProbeSample,
    G1CausalSkillTransitionMemoryActor,
    G1CausalSkillTransitionRiskActor,
    save_causal_skill_transition_memory_actor,
    save_causal_skill_transition_risk_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_START_TO_TRIGGER = {
    1.88: 84,
    1.92: 86,
    1.96: 88,
    2.00: 90,
    2.04: 93,
}


def load_causal_transition_probe_samples(
    discovery_reports: tuple[Path, ...],
) -> tuple[tuple[CausalTransitionProbeSample, ...], dict[str, Any]]:
    """Load candidate-level labels with report and trajectory integrity checks."""

    if len(discovery_reports) < 2:
        raise ValueError("causal transition risk training needs multiple discovery generations")
    samples: list[CausalTransitionProbeSample] = []
    source_rows: list[dict[str, str]] = []
    context_ids: set[str] = set()
    for raw_path in discovery_reports:
        path = raw_path.expanduser().resolve()
        payload = _load_bound_json(path, "report_hash")
        if (
            payload.get("schema_version")
            not in {
                "rosclaw.growth.causal_transition_discovery.v2",
                "rosclaw.growth.causal_transition_discovery.v3",
            }
            or payload.get("status") != "PASS_CAUSAL_TRANSITION_DISCOVERY"
        ):
            raise ValueError("causal transition risk source discovery is not passing")
        cases = payload.get("cases")
        if not isinstance(cases, dict) or not cases:
            raise ValueError("causal transition risk source cases are missing")
        source_rows.append(
            {
                "file": str(path),
                "file_hash": hash_bytes(path.read_bytes()),
                "report_hash": str(payload["report_hash"]),
            }
        )
        for context_id, case in cases.items():
            if context_id in context_ids or not isinstance(case, dict):
                raise ValueError("causal transition risk contexts overlap or are malformed")
            context_ids.add(context_id)
            features = case.get("selected_features")
            probes = case.get("probes")
            trajectory = path.parent / str(case.get("trajectory_file", ""))
            if (
                not isinstance(features, list)
                or not isinstance(probes, list)
                or not trajectory.is_file()
                or hash_bytes(trajectory.read_bytes()) != case.get("trajectory_file_hash")
            ):
                raise ValueError("causal transition risk source trajectory binding changed")
            for probe in probes:
                if not isinstance(probe, dict):
                    raise ValueError("causal transition risk probe is malformed")
                raw_start = probe.get("receiver_start_sec")
                safe = probe.get("safe")
                chain_passed = probe.get("chain_passed")
                if (
                    not isinstance(raw_start, int | float)
                    or isinstance(raw_start, bool)
                    or type(safe) is not bool
                    or type(chain_passed) is not bool
                ):
                    raise ValueError("causal transition risk probe labels are malformed")
                start = float(raw_start)
                trigger = next(
                    (value for key, value in _START_TO_TRIGGER.items() if abs(start - key) < 1e-9),
                    None,
                )
                if trigger is None:
                    raise ValueError("causal transition risk probe uses an unknown phase")
                probe_hash = hash_json(
                    {
                        "source_report_hash": payload["report_hash"],
                        "context_hash": case.get("context_hash"),
                        "probe": probe,
                    }
                )
                samples.append(
                    CausalTransitionProbeSample(
                        sample_id=f"{context_id}@{trigger}",
                        context_id=context_id,
                        features=tuple(float(value) for value in features),
                        trigger_policy_frame=trigger,
                        safe=safe,
                        chain_passed=chain_passed,
                        source_report_hash=str(payload["report_hash"]),
                        source_probe_hash=str(probe_hash),
                    )
                )
    if (
        len(context_ids) < 16
        or len(samples) != len(context_ids) * len(RISK_CANDIDATE_POLICY_FRAMES)
        or len({sample.sample_id for sample in samples}) != len(samples)
    ):
        raise ValueError("causal transition risk probe coverage is incomplete")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_transition_probe_manifest.v1",
        "source_reports": source_rows,
        "context_count": len(context_ids),
        "sample_count": len(samples),
        "safe_sample_count": sum(sample.safe for sample in samples),
        "chain_sample_count": sum(sample.chain_passed for sample in samples),
        "samples_hash": hash_json([asdict(sample) for sample in samples]),
        "physics_authority": "CPU_MUJOCO",
    }
    manifest["manifest_hash"] = hash_json(manifest)
    return tuple(samples), manifest


def train_causal_transition_risk_population(
    *,
    discovery_reports: tuple[Path, ...],
    output_dir: Path,
    devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    seeds: tuple[int, ...] = (4240, 4241, 4242, 4243),
    epochs: int = 1800,
    hidden_size: int = 16,
) -> tuple[G1CausalSkillTransitionRiskActor, dict[str, Any]]:
    """Train four risk heads and calibrate selection on held-out contexts."""

    if (
        len(devices) != 4
        or len(seeds) != 4
        or len(set(seeds)) != 4
        or not all(device.startswith("cuda:") for device in devices)
        or not 200 <= epochs <= 10_000
        or not 4 <= hidden_size <= 64
    ):
        raise ValueError("causal transition risk population mapping is invalid")
    samples, manifest = load_causal_transition_probe_samples(discovery_reports)
    contexts = sorted({sample.context_id for sample in samples})
    validation_folds = tuple(tuple(contexts[index::4]) for index in range(4))
    training_folds = tuple(
        tuple(context for context in contexts if context not in validation)
        for validation in validation_folds
    )
    if any(
        len(training) < 12 or len(validation) < 4
        for training, validation in zip(training_folds, validation_folds, strict=True)
    ):
        raise ValueError("causal transition risk context split is too small")
    observations = np.asarray([sample.features for sample in samples], dtype=np.float64)
    center = np.mean(observations, axis=0)
    scale = np.std(observations, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    minimum = np.min(observations, axis=0)
    maximum = np.max(observations, axis=0)
    head_values: list[dict[str, Any]] = []
    head_reports: list[dict[str, Any]] = []
    for device, seed, training_contexts, validation_contexts in zip(
        devices, seeds, training_folds, validation_folds, strict=True
    ):
        training = tuple(sample for sample in samples if sample.context_id in training_contexts)
        values, metrics = _fit_risk_head(
            training,
            center=center,
            scale=scale,
            device=device,
            seed=seed,
            epochs=epochs,
            hidden_size=hidden_size,
        )
        head_values.append(values)
        head_reports.append(
            {
                "device": device,
                "seed": seed,
                "training_context_ids": training_contexts,
                "validation_context_ids": validation_contexts,
                **metrics,
            }
        )
    source_snapshot_hash = str(manifest["manifest_hash"])
    implementation_hash = _implementation_hash()
    training_snapshot_hash = str(
        hash_json(
            {
                "source_snapshot_hash": source_snapshot_hash,
                "implementation_hash": implementation_hash,
                "head_training_context_ids": training_folds,
                "head_validation_context_ids": validation_folds,
                "devices": devices,
                "seeds": seeds,
                "epochs": epochs,
                "hidden_size": hidden_size,
            }
        )
    )

    def actor_for(
        safety: float, chain: float, advantage: float
    ) -> G1CausalSkillTransitionRiskActor:
        return G1CausalSkillTransitionRiskActor(
            source_snapshot_hash=source_snapshot_hash,
            training_snapshot_hash=training_snapshot_hash,
            implementation_hash=implementation_hash,
            feature_center=tuple(float(value) for value in center),
            feature_scale=tuple(float(value) for value in scale),
            feature_minimum=tuple(float(value) for value in minimum),
            feature_maximum=tuple(float(value) for value in maximum),
            hidden_weights=tuple(values["hidden_weights"] for values in head_values),
            hidden_bias=tuple(values["hidden_bias"] for values in head_values),
            output_weights=tuple(values["output_weights"] for values in head_values),
            output_bias=tuple(values["output_bias"] for values in head_values),
            head_training_context_ids=training_folds,
            head_validation_context_ids=validation_folds,
            safety_probability_threshold=safety,
            chain_probability_threshold=chain,
            minimum_chain_advantage=advantage,
        )

    calibrated: list[tuple[G1CausalSkillTransitionRiskActor, dict[str, Any]]] = []
    for safety in (0.50, 0.60, 0.70, 0.80, 0.90):
        for chain in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
            for advantage in (0.00, 0.05, 0.10, 0.15):
                actor = actor_for(safety, chain, advantage)
                calibrated.append((actor, _selection_metrics(actor, samples, out_of_fold=True)))
    eligible = [
        item
        for item in calibrated
        if item[1]["unsafe_selection_count"] == 0
        and item[1]["actor_chain_success_count"] >= item[1]["parent_chain_success_count"] + 1
        and item[1]["nonzero_selection_count"] >= 2
    ]
    pool = eligible or calibrated
    pool.sort(
        key=lambda item: (
            item[1]["unsafe_selection_count"],
            -item[1]["actor_chain_success_count"],
            -item[1]["chain_success_gain"],
            -item[1]["nonzero_selection_count"],
            -item[0].safety_probability_threshold,
            -item[0].chain_probability_threshold,
            -item[0].minimum_chain_advantage,
        )
    )
    actor, validation_metrics = pool[0]
    calibration_passed = bool(eligible)
    output = _new_external_output(output_dir)
    actor_path = output / "causal-transition-risk-actor.json"
    save_causal_skill_transition_risk_actor(actor, actor_path)
    _write_json(output / "probe-manifest.json", manifest)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_transition_risk_population.v1",
        "status": (
            "PASS_CAUSAL_TRANSITION_RISK_CALIBRATION"
            if calibration_passed
            else "REJECTED_CAUSAL_TRANSITION_RISK_CALIBRATION"
        ),
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "source_snapshot_hash": source_snapshot_hash,
        "training_snapshot_hash": training_snapshot_hash,
        "context_count": len(contexts),
        "training_context_count_per_head": [len(values) for values in training_folds],
        "validation_context_count_per_head": [len(values) for values in validation_folds],
        "probe_count": len(samples),
        "head_reports": head_reports,
        "calibration_thresholds": {
            "safety_probability": actor.safety_probability_threshold,
            "chain_probability": actor.chain_probability_threshold,
            "minimum_chain_advantage": actor.minimum_chain_advantage,
        },
        "validation_selection": validation_metrics,
        "calibration_candidate_count": len(calibrated),
        "eligible_calibration_count": len(eligible),
        "minimum_calibration_nonzero_selection_count": 2,
        "calibration_prediction_mode": "GROUPED_FOUR_FOLD_OUT_OF_FOLD",
        "four_gpu_training_only": True,
        "final_runtime": "NUMPY_JSON_ONLY",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "risk-population-report.json", report)
    if not calibration_passed:
        raise RuntimeError("causal transition risk calibration did not pass")
    return actor, report


def build_causal_transition_memory_actor(
    *,
    discovery_reports: tuple[Path, ...],
    source_neural_rejection_paths: tuple[Path, ...],
    output_dir: Path,
) -> tuple[G1CausalSkillTransitionMemoryActor, dict[str, Any]]:
    """Compile a chance-constrained local memory after neural calibration failures."""

    samples, manifest = load_causal_transition_probe_samples(discovery_reports)
    if len(source_neural_rejection_paths) < 2:
        raise ValueError("causal transition memory needs multiple neural rejection reports")
    neural_rejections: list[dict[str, str]] = []
    for raw_path in source_neural_rejection_paths:
        path = raw_path.expanduser().resolve()
        payload = _load_bound_json(path, "report_hash")
        if payload.get("status") != "REJECTED_CAUSAL_TRANSITION_RISK_CALIBRATION":
            raise ValueError("causal transition memory source must be a neural rejection")
        neural_rejections.append(
            {
                "file": str(path),
                "file_hash": hash_bytes(path.read_bytes()),
                "report_hash": str(payload["report_hash"]),
            }
        )
    grouped: dict[str, dict[int, CausalTransitionProbeSample]] = defaultdict(dict)
    for sample in samples:
        grouped[sample.context_id][sample.trigger_policy_frame] = sample
    context_ids = tuple(sorted(grouped))
    features = np.asarray(
        [grouped[context][88].features for context in context_ids], dtype=np.float64
    )
    safe = np.asarray(
        [
            [grouped[context][frame].safe for frame in RISK_CANDIDATE_POLICY_FRAMES]
            for context in context_ids
        ],
        dtype=np.bool_,
    )
    chain = np.asarray(
        [
            [grouped[context][frame].chain_passed for frame in RISK_CANDIDATE_POLICY_FRAMES]
            for context in context_ids
        ],
        dtype=np.bool_,
    )
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    minimum = np.min(features, axis=0)
    maximum = np.max(features, axis=0)
    normalized = (features - center) / scale
    pairwise = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    implementation_hash = _implementation_hash()
    source_snapshot_hash = str(
        hash_json(
            {
                "probe_manifest_hash": manifest["manifest_hash"],
                "neural_rejections": neural_rejections,
                "implementation_hash": implementation_hash,
            }
        )
    )

    def actor_for(
        neighbors: int, chain_fraction: float, advantage: float
    ) -> G1CausalSkillTransitionMemoryActor:
        kth = np.sort(pairwise, axis=1)[:, neighbors - 1]
        maximum_neighbor_distance = float(np.clip(np.max(kth) * 1.25, 0.5, 20.0))
        return G1CausalSkillTransitionMemoryActor(
            source_snapshot_hash=source_snapshot_hash,
            implementation_hash=implementation_hash,
            feature_center=tuple(float(value) for value in center),
            feature_scale=tuple(float(value) for value in scale),
            feature_minimum=tuple(float(value) for value in minimum),
            feature_maximum=tuple(float(value) for value in maximum),
            prototype_context_ids=context_ids,
            prototype_features=tuple(tuple(float(value) for value in row) for row in features),
            safe_labels=tuple(tuple(bool(value) for value in row) for row in safe),
            chain_labels=tuple(tuple(bool(value) for value in row) for row in chain),
            neighbor_count=neighbors,
            minimum_neighbor_chain_fraction=chain_fraction,
            minimum_chain_advantage=advantage,
            maximum_neighbor_distance=maximum_neighbor_distance,
        )

    calibrated: list[tuple[G1CausalSkillTransitionMemoryActor, dict[str, Any]]] = []
    for neighbors in range(3, min(10, len(context_ids) - 1) + 1):
        for chain_fraction in (0.50, 0.60, 0.67, 0.80, 1.00):
            for advantage in (0.01, 0.10, 0.20):
                actor = actor_for(neighbors, chain_fraction, advantage)
                calibrated.append((actor, _memory_selection_metrics(actor, grouped)))
    eligible = [
        item
        for item in calibrated
        if item[1]["unsafe_selection_count"] == 0
        and item[1]["actor_chain_success_count"] >= item[1]["parent_chain_success_count"] + 1
        and item[1]["nonzero_selection_count"] >= 2
    ]
    pool = eligible or calibrated
    pool.sort(
        key=lambda item: (
            item[1]["unsafe_selection_count"],
            -item[1]["actor_chain_success_count"],
            -item[1]["chain_success_gain"],
            -item[1]["nonzero_selection_count"],
            -item[0].neighbor_count,
            -item[0].minimum_neighbor_chain_fraction,
            -item[0].minimum_chain_advantage,
        )
    )
    actor, calibration = pool[0]
    passed = bool(eligible)
    output = _new_external_output(output_dir)
    actor_path = output / "causal-transition-memory-actor.json"
    save_causal_skill_transition_memory_actor(actor, actor_path)
    _write_json(output / "probe-manifest.json", manifest)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_transition_memory_population.v1",
        "status": (
            "PASS_CAUSAL_TRANSITION_MEMORY_CALIBRATION"
            if passed
            else "REJECTED_CAUSAL_TRANSITION_MEMORY_CALIBRATION"
        ),
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "source_snapshot_hash": source_snapshot_hash,
        "source_probe_manifest_hash": manifest["manifest_hash"],
        "source_neural_rejections": neural_rejections,
        "context_count": len(context_ids),
        "probe_count": len(samples),
        "calibration_candidate_count": len(calibrated),
        "eligible_calibration_count": len(eligible),
        "selected_hyperparameters": {
            "neighbor_count": actor.neighbor_count,
            "minimum_neighbor_chain_fraction": actor.minimum_neighbor_chain_fraction,
            "minimum_chain_advantage": actor.minimum_chain_advantage,
            "maximum_neighbor_distance": actor.maximum_neighbor_distance,
        },
        "leave_one_context_out_calibration": calibration,
        "training_backend": "NUMPY_EPISODIC_MEMORY",
        "final_runtime": "NUMPY_JSON_ONLY",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "memory-population-report.json", report)
    if not passed:
        raise RuntimeError("causal transition memory calibration did not pass")
    return actor, report


def _fit_risk_head(
    samples: tuple[CausalTransitionProbeSample, ...],
    *,
    center: np.ndarray[Any, Any],
    scale: np.ndarray[Any, Any],
    device: str,
    seed: int,
    epochs: int,
    hidden_size: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch

    x_context = (
        np.asarray([sample.features for sample in samples], dtype=np.float64) - center
    ) / scale
    phase = np.asarray(
        [(sample.trigger_policy_frame - 88) / 5.0 for sample in samples], dtype=np.float64
    )[:, None]
    x_array = np.concatenate((x_context, phase), axis=1)
    y_array = np.asarray(
        [(float(sample.safe), float(sample.chain_passed)) for sample in samples],
        dtype=np.float64,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    target_device = torch.device(device)
    x = torch.as_tensor(x_array, dtype=torch.float32, device=target_device)
    y = torch.as_tensor(y_array, dtype=torch.float32, device=target_device)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, 2),
    ).to(target_device)
    positive = torch.sum(y, dim=0)
    negative = y.shape[0] - positive
    positive_weight = torch.clamp(negative / torch.clamp(positive, min=1.0), 1.0, 6.0)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=2.0e-3)
    for _ in range(epochs):
        logits = model(x)
        loss = loss_function(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probability = torch.sigmoid(model(x)).cpu().numpy()
    first = cast(Any, model[0])
    second = cast(Any, model[2])
    predictions = probability >= 0.5
    return (
        {
            "hidden_weights": tuple(
                tuple(float(value) for value in row)
                for row in first.weight.detach().cpu().numpy().astype(np.float64)
            ),
            "hidden_bias": tuple(
                float(value) for value in first.bias.detach().cpu().numpy().astype(np.float64)
            ),
            "output_weights": tuple(
                tuple(float(value) for value in row)
                for row in second.weight.detach().cpu().numpy().astype(np.float64)
            ),
            "output_bias": tuple(
                float(value) for value in second.bias.detach().cpu().numpy().astype(np.float64)
            ),
        },
        {
            "training_loss": float(loss.detach().cpu()),
            "safe_accuracy": float(np.mean(predictions[:, 0] == y_array[:, 0])),
            "chain_accuracy": float(np.mean(predictions[:, 1] == y_array[:, 1])),
        },
    )


def _selection_metrics(
    actor: G1CausalSkillTransitionRiskActor,
    samples: tuple[CausalTransitionProbeSample, ...],
    *,
    out_of_fold: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[CausalTransitionProbeSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.context_id].append(sample)
    actor_success = 0
    parent_success = 0
    unsafe = 0
    nonzero = 0
    rows: list[dict[str, Any]] = []
    for context_id in sorted(grouped):
        candidates = {sample.trigger_policy_frame: sample for sample in grouped[context_id]}
        observation = np.asarray(grouped[context_id][0].features, dtype=np.float64)
        if out_of_fold:
            matching_heads = [
                index
                for index, validation in enumerate(actor.head_validation_context_ids)
                if context_id in validation
            ]
            if len(matching_heads) != 1:
                raise ValueError("causal transition risk OOF context mapping is invalid")
            probabilities = actor._candidate_probabilities(observation)[
                matching_heads[0] : matching_heads[0] + 1
            ]
            minimum = np.asarray(actor.feature_minimum, dtype=np.float64)
            maximum = np.asarray(actor.feature_maximum, dtype=np.float64)
            span = np.maximum(maximum - minimum, 0.05)
            support_distance = float(
                np.linalg.norm(
                    (
                        np.maximum(minimum - observation, 0.0)
                        + np.maximum(observation - maximum, 0.0)
                    )
                    / span
                )
            )
            decision = actor._decision_from_probabilities(
                observation, probabilities, support_distance
            )
        else:
            decision = actor.decide(observation)
        selected = candidates[decision.trigger_policy_frame]
        parent = candidates[actor.parent_trigger_policy_frame]
        actor_success += selected.chain_passed
        parent_success += parent.chain_passed
        unsafe += not selected.safe
        nonzero += decision.residual_frames != 0
        rows.append(
            {
                "context_id": context_id,
                "selected_trigger_policy_frame": decision.trigger_policy_frame,
                "selected_safe": selected.safe,
                "selected_chain_passed": selected.chain_passed,
                "parent_chain_passed": parent.chain_passed,
                "predicted_safe_probability": decision.predicted_safe_probability,
                "predicted_chain_probability": decision.predicted_chain_probability,
                "ensemble_probability_spread": decision.ensemble_probability_spread,
                "used_parent_fallback": decision.used_parent_fallback,
            }
        )
    return {
        "context_count": len(grouped),
        "actor_chain_success_count": actor_success,
        "parent_chain_success_count": parent_success,
        "chain_success_gain": actor_success - parent_success,
        "unsafe_selection_count": unsafe,
        "nonzero_selection_count": nonzero,
        "rows": rows,
    }


def _memory_selection_metrics(
    actor: G1CausalSkillTransitionMemoryActor,
    grouped: dict[str, dict[int, CausalTransitionProbeSample]],
) -> dict[str, Any]:
    actor_success = 0
    parent_success = 0
    unsafe = 0
    nonzero = 0
    rows: list[dict[str, Any]] = []
    for context_id in sorted(grouped):
        candidates = grouped[context_id]
        observation = np.asarray(candidates[88].features, dtype=np.float64)
        decision = actor._decide(observation, excluded_context_id=context_id)
        selected = candidates[decision.trigger_policy_frame]
        parent = candidates[actor.parent_trigger_policy_frame]
        actor_success += selected.chain_passed
        parent_success += parent.chain_passed
        unsafe += not selected.safe
        nonzero += decision.residual_frames != 0
        rows.append(
            {
                "context_id": context_id,
                "selected_trigger_policy_frame": decision.trigger_policy_frame,
                "selected_safe": selected.safe,
                "selected_chain_passed": selected.chain_passed,
                "parent_chain_passed": parent.chain_passed,
                "neighbor_safe_fraction": decision.predicted_safe_probability,
                "neighbor_chain_fraction": decision.predicted_chain_probability,
                "used_parent_fallback": decision.used_parent_fallback,
            }
        )
    return {
        "context_count": len(grouped),
        "actor_chain_success_count": actor_success,
        "parent_chain_success_count": parent_success,
        "chain_success_gain": actor_success - parent_success,
        "unsafe_selection_count": unsafe,
        "nonzero_selection_count": nonzero,
        "rows": rows,
    }


def _load_bound_json(path: Path, hash_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal transition risk bound JSON must be an object")
    claimed = payload.pop(hash_key, None)
    try:
        if claimed != hash_json(payload):
            raise ValueError("causal transition risk source integrity changed")
    finally:
        if claimed is not None:
            payload[hash_key] = claimed
    return payload


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "causal_skill_transition_risk.py",
        Path(__file__).parents[1] / "growth" / "causal_skill_transition.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("causal transition risk output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "build_causal_transition_memory_actor",
    "load_causal_transition_probe_samples",
    "train_causal_transition_risk_population",
]
