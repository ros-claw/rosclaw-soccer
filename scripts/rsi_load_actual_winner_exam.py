"""Bind an actual-path zero-anchor candidate to the unchanged own-body exam."""

import json
from pathlib import Path

import numpy as np
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.pre_parent_reference_features import FEATURE_NAMES
from rosclaw_soccer.rsi.reference_student_execution import own_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_fit_actual_winner_batch import load_batch
from scripts.rsi_stage_release_contact_replay import require_terminal


def require_actual_fit(stage, fit, *, configuration, prepared, batch):
    """Validate real numeric lineage, not a synthetic 'training passed' bit."""
    ignored = [
        i
        for i, n in enumerate(FEATURE_NAMES)
        if n.startswith("previous_applied_reference_increment:")
    ]
    initial = initial_parameters(135, 12, hidden_dimension=64, seed=2026101902)
    weight = np.asarray(initial["weight_ih"])
    weight[:, ignored] = 0
    initial["weight_ih"] = weight.tolist()
    model = own_model(fit["numeric_model"])
    lr = fit["learning_receipt"]
    if (
        stage.get("schema") != "soccer.rsi.actual_winner_fit_complete.v1"
        or stage.get("status") != "NUMERIC_CANDIDATE_REQUIRES_OWN_BODY_EXAM"
        or stage.get("configuration") != configuration
        or configuration.get("profile") != "one-actual-winner-zero-anchor-fit-v1"
        or configuration.get("fit_seed") != 2026101902
        or configuration.get("fit_steps") != 128
        or configuration.get("maximum_numeric_candidates") != 1
        or stage.get("numeric_fit_hash") != fit["report_hash"]
        or stage.get("prepared_complete_hash") != prepared["report_hash"]
        or stage.get("selection_hash") != prepared["selection"]["report_hash"]
        or type(stage.get("optimizer_updates")) is not int
        or stage["optimizer_updates"] != 128
        or any(
            stage.get(k) is not False
            for k in (
                "own_body_student_evaluated",
                "validation_checkpoint_selection",
                "dataset_admission_authorized",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or model["parent_model_hash"] != configuration["parent_model_hash"]
        or model["world_hash"] != configuration["world_hash"]
        or not np.array_equal(model["mean"], batch["mean"])
        or not np.array_equal(model["scale"], batch["scale"])
        or fit.get("normalization_hash")
        != hash_json(dict(mean=batch["mean"].tolist(), scale=batch["scale"].tolist()))
        or fit.get("whole_observations_hash")
        != hash_bytes(batch["original_observations"].tobytes())
        or fit.get("whole_reference_labels_hash") != hash_bytes(batch["targets"].tobytes())
        or fit.get("initial_parameters_hash") != hash_json(initial)
        or fit.get("past_reference_input_ablation") is not True
        or fit.get("ignored_input_weights_remain_exactly_zero") is not True
        or fit.get("previous_reference_inputs_are_teacher_forced") is not False
        or fit.get("ignored_input_feature_names") != [FEATURE_NAMES[i] for i in ignored]
        or np.any(np.asarray(model["parameters"]["weight_ih"])[:, ignored] != 0)
        or lr.get("parameters") != model["parameters"]
        or lr.get("fitted_parameters_hash") != hash_json(model["parameters"])
        or type(lr.get("completed_optimizer_steps")) is not int
        or lr["completed_optimizer_steps"] != 128
        or lr.get("config")
        != dict(
            steps=128,
            batch_episodes=4,
            learning_rate=0.0004,
            residual_cap=0.2,
            seed=2026101902,
            compute_device="cpu",
        )
        or fit.get("input_episode_count") != 16
        or fit.get("optimized_episode_count") != 3
        or fit.get("train_source_seeds") != [20261077, 20261093]
        or fit.get("validation_source_seeds") != [20261096]
        or fit.get("validation_excluded_from_gradient_and_normalization") is not True
    ):
        raise ValueError(
            "exact finite zero-anchor fit and reconstructed train-only lineage required"
        )
    return model


def load_actual_winner(cfg, *, sealed, digest, pins):
    spec = cfg["actual_winner_fit"]
    if type(spec) is not dict or set(spec) != {
        "fit_configuration",
        "fit_configuration_file_hash",
        "fit_unit",
    }:
        raise ValueError("explicit actual-winner fit source/configuration required")
    require_terminal(spec["fit_unit"])
    cp = Path(spec["fit_configuration"])
    if digest(cp) != spec["fit_configuration_file_hash"]:
        raise ValueError("unchanged actual-winner fit configuration required")
    pins[str(cp)] = digest(cp)
    fc = json.loads(cp.read_text())
    root = Path(fc["output_root"])
    stage, fit = (sealed(root / name) for name in ("complete.json", "numeric-fit.json"))
    for name, h in stage["input_file_hashes"].items():
        if digest(name) != h:
            raise ValueError("actual-winner fit original source or evidence changed")
        pins[name] = h
    prepared, batch = load_batch(fc, sealed=sealed, pins=pins)
    model = require_actual_fit(stage, fit, configuration=fc, prepared=prepared, batch=batch)
    if (
        model["parent_model_hash"] != cfg["behavior_model_hash"]
        or model["world_hash"] != cfg["mesh_task_contact_world_hash"]
    ):
        raise ValueError("unchanged exam world and frozen parent required")
    return model, stage, fit
