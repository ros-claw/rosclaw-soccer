"""Three-seed offline consequence prediction after whole-bank physical proof.

Not teacher imitation, online RL, an activated torque actor, or promotion.
Only12 local development contexts may fit weights/normalization. The six
original consumed retention contexts remain evaluation-only, never Fresh.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.postcontact_outcome_dataset import masked_brier, measured_dataset
from rosclaw_soccer.rsi.postcontact_outcome_model import EPOCHS, SEEDS, fit, predict
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_replay_postcontact_full_bank_physics import require_full_physical_review
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--readback-file-hash", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        args.output.exists()
        or (args.root / "failure.json").exists()
        or digest(args.readback) != args.readback_file_hash
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused diagnostic output and clean exact frozen learning source required")
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def sealed(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and h != expected)
            or (str(file) in pins and pins[str(file)] != h)
        ):
            raise ValueError("same original ordinary physics and learning input bytes required")
        pins[str(file)] = h
        report = json.loads(file.read_text())
        if report["report_hash"] != hash_json(
            {k: v for k, v in report.items() if k != "report_hash"}
        ):
            raise ValueError("complete sealed learning inputs required")
        return report

    rb = sealed(args.readback)
    proof, pd = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    if (
        proof["schema"] != "soccer.rsi.recorded_postcontact_full_bank_complete.v1"
        or proof["declaration_hash"] != pd["report_hash"]
        or pd["raw_readback_hash"] != rb["report_hash"]
        or pd["producer_complete_hash"] != rb["complete_hash"]
        or len(proof["rows"]) != 252
        or any(
            type(proof.get(k)) is not int or proof[k] != n
            for k, n in dict(
                comparison_bodies_verified=252,
                new_physical_bodies=238,
                new_physical_steps=714000,
                reused_physical_bodies=14,
                reused_physical_steps=42000,
                new_neural_inferences=0,
                new_optimizer_updates=0,
            ).items()
        )
        or any(
            proof.get(k) is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("whole actual714000-step bank verification, not a teacher, required")
    for name, h in pd["input_file_hashes"].items():
        file = Path(name)
        if file.is_symlink() or digest(file) != h or (name in pins and pins[name] != h):
            raise ValueError("all original physical source pins required before fitting")
        pins[name] = h
    for i, row in enumerate(proof["rows"]):
        ci, ai = divmod(i, 14)
        if (
            row != sealed(args.root / f"context-{ci}-action-{ai}-complete.json")
            or row["context_index"] != ci
            or row["action_index"] != ai
            or row["physical_execution_reused"] is not (ci == 1)
        ):
            raise ValueError("all252 ordered physical comparison proofs required")
        require_full_physical_review(row["physical_review"])
    dataset = measured_dataset(rb)
    folds = dataset.group_folds()
    if args.preflight:
        print("DIAGNOSTIC_OUTCOME_FIT_READY_NO_ACTOR_OR_OPTIMIZER_YET", flush=True)
        return
    args.output.mkdir()
    d = dict(
        schema="soccer.rsi.postcontact_outcome_fit_declaration.v1",
        source_head=args.source_head,
        raw_readback_hash=rb["report_hash"],
        physical_complete_hash=proof["report_hash"],
        input_file_hashes=pins,
        seeds=list(SEEDS),
        epochs_per_model=EPOCHS,
        model_budget=len(SEEDS) * (len(folds) + 1),
        optimizer_update_budget=len(SEEDS) * (len(folds) + 1) * EPOCHS,
        training_contexts=[dataset.context_indices[i] for i in dataset.local_indices()],
        retention_contexts=[dataset.context_indices[i] for i in dataset.retention_indices()],
        contexts_without_eligible_observation=list(dataset.excluded_context_indices),
        all18_original_candidate_rows=rb["candidate_supervision_not_admitted"],
        fit_purpose="diagnostic measured consequence prediction, not failed-teacher imitation",
        validation="leave-original-source-cluster-out; all derivative headings kept together",
        maximum_runtime_seconds=3600,
        mode="SIM_ONLY",
        new_physical_step_budget=0,
        motor_policy_activation_authorized=False,
        teacher_qualified=False,
        online_rl_claimed=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    d["report_hash"] = hash_json(d)
    write_once(args.output / "declaration.json", d)
    results = []
    completed_fits = completed_updates = 0
    fitting = False
    current = None
    deadline = time.monotonic() + 3600
    try:
        for seed in SEEDS:
            for fold in range(len(folds) + 1):
                if time.monotonic() >= deadline:
                    raise RuntimeError("bounded diagnostic learning deadline exhausted")
                if fold < len(folds):
                    train, test = folds[fold]
                    role = "source_cluster_cross_validation"
                else:
                    train, test = dataset.local_indices(), dataset.retention_indices()
                    role = "fit_all_local_evaluate_consumed_retention_only"
                current = dict(seed=seed, fold=fold, role=role)
                fitting = True
                model = fit(
                    dataset.features[train],
                    dataset.labels[train],
                    dataset.mask[train],
                    seed=seed,
                    dataset_hash=d["report_hash"],
                    training_context_indices=[dataset.context_indices[i] for i in train],
                )
                fitting = False
                completed_fits += 1
                completed_updates += model["optimizer_updates"]
                write_once(args.output / f"model-{seed}-{fold}.json", model)
                metrics = None
                if len(test):
                    p = predict(model, dataset.features[test])
                    m, y = dataset.mask[train], dataset.labels[train]
                    prior = (np.sum(y & m, axis=0) + 1.0) / (np.sum(m, axis=0) + 2.0)
                    constant = np.broadcast_to(prior, p.shape)
                    metrics = dict(
                        evaluation_context_indices=[dataset.context_indices[i] for i in test],
                        measured_head_counts=dataset.mask[test].sum(axis=(0, 1)).tolist(),
                        brier_pass_safe=masked_brier(p, dataset.labels[test], dataset.mask[test]),
                        training_only_constant_brier=masked_brier(
                            constant, dataset.labels[test], dataset.mask[test]
                        ),
                        probabilities_pass_safe=p.tolist(),
                    )
                # Timings include source-independent JSON seal/weight validation.
                durations = []
                for _ in range(100):
                    start = time.perf_counter()
                    predict(model, dataset.features[train[:1]])
                    durations.append(time.perf_counter() - start)
                row = dict(
                    current,
                    model_hash=model["report_hash"],
                    evaluation=metrics,
                    optimizer_updates=model["optimizer_updates"],
                    sealed_single_observation_prediction_seconds=dict(
                        median=float(np.median(durations)), maximum=max(durations)
                    ),
                )
                row["report_hash"] = hash_json(row)
                write_once(args.output / f"result-{seed}-{fold}.json", row)
                results.append(row)
                print("DIAGNOSTIC_OUTCOME_MODEL_FIT_COMPLETE", current, metrics, flush=True)
        if any(digest(n) != h for n, h in pins.items()):
            raise ValueError("learning inputs or frozen source changed during fitting")
        result = dict(
            schema="soccer.rsi.postcontact_outcome_fit_complete.v1",
            declaration_hash=d["report_hash"],
            results=results,
            models_fitted=len(results),
            optimizer_updates=sum(r["optimizer_updates"] for r in results),
            physical_steps=0,
            source_contexts=18,
            observed_local_training_states=len(dataset.local_indices()),
            original_clusters_with_training_observations=len(
                set(dataset.source_clusters[i] for i in dataset.local_indices())
            ),
            fitted_model_is_motor_actor=False,
            teacher_qualified=False,
            motor_policy_activation_authorized=False,
            online_rl_claimed=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(args.output / "complete.json", result)
    except BaseException as error:
        write_once(
            args.output / "failure.json",
            dict(
                current=current,
                completed_models=completed_fits,
                completed_model_evaluations=len(results),
                completed_model_optimizer_updates=completed_updates,
                current_fit_optimizer_updates_unreported=fitting,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
