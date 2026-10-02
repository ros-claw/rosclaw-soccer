"""Predeclare one current-anchor-protected update on all sealed 104 rollouts.

No new sampling, reward shaping, course routing, physical qualification or
activation. The new zero-addition behavior is globally the same frozen NN.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import rosclaw.growth.correlated_residual_gradient as gradient_module

from rosclaw_soccer.rsi.consolidated_smooth_motor import make_model as make_baseline
from rosclaw_soccer.rsi.current_memory_learning import fit_update
from rosclaw_soccer.rsi.current_memory_motor import initial_model
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.smooth_memory_motor import validate_model as validate_behavior
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_compressed_bank_storage import capacity_check
from scripts.rsi_fit_smooth_memory_motor import checked_curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "learning-root",
        "exploration-root",
        "behavior-model",
        "parent-bank-root",
        "consolidated-memory",
        "consolidation-manifest",
        "audit-source-root",
        "core-root",
        "output-root",
        "system-reserve-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--raw-residual-cap", type=float, choices=(0.05, 0.2), default=0.2)
    parser.add_argument("--learning-rate", type=float, choices=(1e-4, 4e-4), default=4e-4)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        Path(gradient_module.__file__).resolve()
        != (args.core_root / "src/rosclaw/growth/correlated_residual_gradient.py").resolve()
    ):
        raise ValueError("imported optimizer must belong to the declared immutable Core checkout")
    for root in (source, args.core_root):
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
            raise ValueError("immutable clean learner and Core source required")
    behavior = load_json_artifact(args.behavior_model)
    validate_behavior(behavior)
    if behavior["generation"] != 0:
        raise ValueError("globally exact initial NN behavior required")
    manifest = _sealed(args.learning_root / "rollout_manifest.json")
    exploration = _sealed(args.exploration_root / "training_summary.json")
    bank, review = (
        _sealed(args.parent_bank_root / n)
        for n in ("validation_summary.json", "independent_review.json")
    )
    checked_curriculum(
        behavior,
        exploration,
        bank,
        review,
        load_json_artifact(args.exploration_root / "commitment.json"),
    )
    npz = args.learning_root / "rollouts.npz"
    if (
        manifest["schema"] != "soccer.rsi.smooth_memory_on_policy_bank.v1"
        or manifest["parent_model_hash"] != behavior["model_hash"]
        or manifest["source_summary_hash"] != exploration["report_hash"]
        or manifest["physical_rollout_count"] != 104
        or manifest["frame_sample_count"] != 28080
        or manifest["independent_contexts"] != 13
        or manifest["sampling_rho"] != 0.9
        or manifest["candidate_previous_mean_required"] is not True
        or manifest["data_hash"] != hash_bytes(npz.read_bytes())
        or manifest["source_hash"]
        != hash_bytes(
            (args.audit_source_root / "scripts/rsi_fit_parallel_memory_motor.py").read_bytes()
        )
        or manifest["audit_helper_hash"]
        != hash_bytes(
            (args.audit_source_root / "scripts/rsi_audit_memory_learning_rollouts.py").read_bytes()
        )
        or len(manifest["records"]) != 104
        or [r["group"] for r in manifest["records"]] != list(range(104))
        or any(
            obj.get(k) is not False
            for obj in (manifest, exploration, bank, review)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("all immutable independently reconstructed 104 physical rollouts required")
    with np.load(npz, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if not np.array_equal(arrays["trajectory_index"], np.repeat(np.arange(104), 270)):
        raise ValueError("no omitted or reordered training trajectories")
    for record in manifest["records"]:
        g = record["group"]
        row = exploration["rows"][g // 8]
        sample = row["samples"][g % 8]
        if (
            (record["seed"], record["lane"]) != (row["seed"], row["lane"])
            or record["sample"] != g % 8
            or record["report_hash"] != sample["report_hash"]
            or any(record["outcome"][k] != v for k, v in sample.items() if k in record["outcome"])
            or not np.all(
                arrays["terminal_return"][arrays["trajectory_index"] == g]
                == terminal_return(record["outcome"])
            )
        ):
            raise ValueError("all original measured success/failure labels must stay unchanged")
    memory = load_json_artifact(args.consolidated_memory)
    consolidation = _sealed(args.consolidation_manifest)
    if (
        consolidation["bank_summary_hash"] != bank["report_hash"]
        or consolidation["bank_review_hash"] != review["report_hash"]
        or consolidation["parent_model_hash"] != behavior["parent_model_hash"]
        or len(consolidation["records"]) != 39
    ):
        raise ValueError("all 39 successes of the same qualified current NN parent required")
    model = initial_model(
        make_baseline(behavior, memory, consolidation),
        cap=args.raw_residual_cap,
        learning_rate=args.learning_rate,
    )
    paths = [
        args.behavior_model,
        npz,
        args.consolidated_memory,
        args.consolidation_manifest,
        args.learning_root / "rollout_manifest.json",
        args.exploration_root / "training_summary.json",
        args.exploration_root / "commitment.json",
        args.parent_bank_root / "validation_summary.json",
        args.parent_bank_root / "independent_review.json",
        source / "scripts/rsi_fit_current_memory_gradient.py",
        source / "src/rosclaw_soccer/rsi/current_memory_motor.py",
        source / "src/rosclaw_soccer/rsi/current_memory_learning.py",
        args.core_root / "src/rosclaw/growth/correlated_residual_gradient.py",
    ]
    bound = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in paths}
    budget = capacity_check(args.output_root.parent, args.system_reserve_path, 256 * 1024**2)
    declaration = dict(
        schema="soccer.rsi.current_memory_gradient_commitment.v1",
        inputs=bound,
        source_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        core_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.core_root, text=True
        ).strip(),
        initial_model_hash=model["model_hash"],
        behavior_model_hash=behavior["model_hash"],
        qualified_parent_model_hash=behavior["parent_model_hash"],
        physical_batch_hash=manifest["report_hash"],
        physical_rollout_count=104,
        frame_sample_count=28080,
        new_physical_executions=0,
        rewards="UNCHANGED_ORIGINAL_MEASURED_TERMINAL_RETURN",
        protection="ALL_39_CURRENT_PARENT_SUCCESSES_DURING_TRAINING_AND_EXECUTION",
        raw_residual_cap=args.raw_residual_cap,
        learning_rate=args.learning_rate,
        rho=0.9,
        optimizer_steps_limit=160,
        conditional_and_marginal_kl_limit=0.005,
        partition="TRAIN_CONSUMED",
        fresh_holdout_opened=False,
        promotion_authorized=False,
        hardware_authorized=False,
        storage_preflight=budget,
    )
    declaration["report_hash"] = hash_json(declaration)
    args.output_root.mkdir(exist_ok=False)
    write_once(args.output_root / "commitment.json", declaration)
    write_once(args.output_root / "initial_model.json.gz", model)
    print("CURRENT_MEMORY_GRADIENT_DECLARED=" + declaration["report_hash"], flush=True)
    learned = fit_update(model, arrays, batch_hash=manifest["report_hash"])
    if any(hash_bytes(Path(p).read_bytes()) != digest for p, digest in bound.items()):
        raise ValueError("learner inputs or source changed during fitting")
    for root, expected in (
        (source, declaration["source_head"]),
        (args.core_root, declaration["core_head"]),
    ):
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            != expected
            or subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        ):
            raise ValueError("immutable learner or Core source changed during fitting")
    write_once(args.output_root / "model.json.gz", learned)
    receipt = dict(
        schema="soccer.rsi.current_memory_gradient_learning.v1",
        commitment_hash=declaration["report_hash"],
        model_hash=learned["model_hash"],
        initial_model_hash=model["model_hash"],
        learning_receipt=learned["learning_receipt"],
        qualification="UNQUALIFIED_OFFLINE_UPDATE_NOT_PHYSICAL_GROWTH",
        new_physical_executions=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    receipt["report_hash"] = hash_json(receipt)
    write_once(args.output_root / "learning.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
