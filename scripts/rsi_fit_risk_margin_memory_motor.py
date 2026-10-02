"""One declared risk-margin PPO ablation on ALL 104 already audited rollouts.

Existing physical observations/actions/likelihoods stay unchanged. New reward
labels are explicit offline measurements, not new episodes or runtime routing.
Original actor-critic optimizer and limits stay unchanged; current successes
are consolidated AFTER training, explicitly not a constrained-optimizer claim.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.consolidated_smooth_motor import make_model
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.smooth_memory_learning import fit_update
from rosclaw_soccer.rsi.smooth_memory_motor import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_fit_smooth_memory_motor import checked_curriculum
from scripts.rsi_train_protected_online_motor_v308 import _head


def risk_margin_return(
    outcome: dict[str, Any], *, boundary_weight: float = 8.0
) -> tuple[float, dict[str, float]]:
    """Finite measured margins; no course, role, frame or future runtime selector."""
    if (
        type(boundary_weight) not in (float, int)
        or not np.isfinite(boundary_weight)
        or not 8 <= boundary_weight <= 32
    ):
        raise ValueError("finite preregistered boundary weight between 8 and 32 required")
    baseline = terminal_return(outcome)
    forward, lateral, excursion = [
        outcome[k] for k in ("forward_60_m", "lateral_60_m", "maximum_lateral_excursion_m")
    ]
    missing_direction = forward is None and lateral is None
    if (forward is None) != (lateral is None) or (missing_direction and outcome["high_quality"]):
        raise ValueError("inconsistent missing post-contact direction")
    finite_values = (excursion,) if missing_direction else (forward, lateral, excursion)
    if not all(type(v) in (float, int) and np.isfinite(v) for v in finite_values):
        raise ValueError("finite independently measured displacement margins required")
    if excursion < 0:
        raise ValueError("nonnegative measured ball excursion required")
    # Acceptance still uses original ratio <= .3, excursion <= 4, pelvis >= .65.
    # This objective encourages room BELOW those unchanged acceptance limits.
    boundary = boundary_weight * float(np.clip((excursion - 3.0) / 1.0, 0.0, 1.0)) ** 2
    if missing_direction:
        # Preserve genuine misses/late contacts; no invented displacement and
        # no dropped trajectory.
        direction = 5.0
    else:
        ratio = abs(lateral) / max(forward, 0.01)
        direction = 5.0 * float(np.clip((ratio - 0.2) / 0.1, 0.0, 1.0)) ** 2
    return baseline - boundary - direction, dict(
        original_terminal_return=baseline,
        boundary_margin_penalty=boundary,
        direction_margin_penalty=direction,
    )


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
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--boundary-weight", type=float, choices=(8.0, 16.0, 32.0), default=8.0)
    parser.add_argument("--system-reserve-path", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    model = load_json_artifact(args.behavior_model)
    validate_model(model)
    old_candidate = load_json_artifact(args.learning_root / "model.json")
    validate_model(old_candidate)
    manifest = _sealed(args.learning_root / "rollout_manifest.json")
    exploration = _sealed(args.exploration_root / "training_summary.json")
    bank, review = (
        _sealed(args.parent_bank_root / n)
        for n in ("validation_summary.json", "independent_review.json")
    )
    checked_curriculum(
        model,
        exploration,
        bank,
        review,
        load_json_artifact(args.exploration_root / "commitment.json"),
    )
    path = args.learning_root / "rollouts.npz"
    if (
        manifest["schema"] != "soccer.rsi.smooth_memory_on_policy_bank.v1"
        or manifest["parent_model_hash"] != model["model_hash"]
        or manifest["source_summary_hash"] != exploration["report_hash"]
        or manifest["physical_rollout_count"] != 104
        or manifest["frame_sample_count"] != 28080
        or manifest["independent_contexts"] != 13
        or manifest["sampling_rho"] != 0.9
        or manifest["candidate_previous_mean_required"] is not True
        or manifest["data_hash"] != hash_bytes(path.read_bytes())
        or old_candidate["learning_receipt"]["physical_batch_hash"] != manifest["report_hash"]
        or old_candidate["learning_receipt"]["learner_parent_hash"] != model["model_hash"]
        or manifest["source_hash"]
        != hash_bytes(
            (args.audit_source_root / "scripts/rsi_fit_parallel_memory_motor.py").read_bytes()
        )
        or manifest["audit_helper_hash"]
        != hash_bytes(
            (args.audit_source_root / "scripts/rsi_audit_memory_learning_rollouts.py").read_bytes()
        )
        or any(
            obj.get(k) is not False
            for obj in (manifest, model, old_candidate)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("entire immutable independently reconstructed 104-rollout bank required")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if (
        len(manifest["records"]) != 104
        or [r["group"] for r in manifest["records"]] != list(range(104))
        or not np.array_equal(arrays["trajectory_index"], np.repeat(np.arange(104), 270))
        or len(arrays["observation"]) != 28080
    ):
        raise ValueError("all ordered successes and failures required; no sample truncation")
    rewards = []
    for record in manifest["records"]:
        group = record["group"]
        declared_row = exploration["rows"][group // 8]
        declared_sample = declared_row["samples"][group % 8]
        if (
            (record["seed"], record["lane"]) != (declared_row["seed"], declared_row["lane"])
            or record["sample"] != group % 8
            or record["report_hash"] != declared_sample["report_hash"]
            or any(
                record["outcome"][key] != value
                for key, value in declared_sample.items()
                if key in record["outcome"]
            )
        ):
            raise ValueError("audited sample identity/outcome diverged from actual exploration")
        ids = arrays["trajectory_index"] == group
        if not np.all(arrays["terminal_return"][ids] == terminal_return(record["outcome"])):
            raise ValueError("historical measured labels changed")
        shaped, margins = risk_margin_return(
            record["outcome"], boundary_weight=args.boundary_weight
        )
        arrays["terminal_return"][ids] = shaped
        rewards.append(
            dict(
                group=group,
                report_hash=record["report_hash"],
                shaped_terminal_return=shaped,
                **margins,
            )
        )
    memory = load_json_artifact(args.consolidated_memory)
    consolidation = _sealed(args.consolidation_manifest)
    if consolidation["parent_model_hash"] != model["frozen_parent"]["model_hash"]:
        raise ValueError("consolidation must bind the exact unchanged qualified NN parent")
    inputs = [
        args.behavior_model,
        path,
        args.consolidated_memory,
        args.consolidation_manifest,
        args.learning_root / "rollout_manifest.json",
        args.learning_root / "model.json",
        args.exploration_root / "training_summary.json",
        args.parent_bank_root / "validation_summary.json",
        args.parent_bank_root / "independent_review.json",
    ]
    pins = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in inputs}
    commitment = dict(
        schema="soccer.rsi.risk_margin_reward_learning.v1",
        partition="TRAIN_CONSUMED",
        source_commit=_head(source),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        input_hashes=pins,
        source_manifest_hash=manifest["report_hash"],
        behavior_model_hash=model["model_hash"],
        comparison_model_hash=old_candidate["model_hash"],
        existing_physical_rollouts_reused=104,
        new_physical_executions=0,
        all_records_used=True,
        altered_fields=["offline_terminal_return"],
        boundary_margin_m=3.0,
        boundary_scale_m=1.0,
        boundary_penalty=args.boundary_weight,
        direction_margin_ratio=0.2,
        direction_scale_ratio=0.1,
        direction_penalty=5.0,
        acceptance_thresholds_unchanged=True,
        optimizer_mathematics_unchanged=True,
        current_parent_memory_consolidation="AFTER_OPTIMIZER_NOT_CONSTRAINED_PPO",
        rewards=rewards,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    commitment["report_hash"] = hash_json(commitment)
    if args.system_reserve_path is not None:
        from scripts.rsi_compressed_bank_storage import capacity_check

        capacity_check(args.output_root.parent, args.system_reserve_path, 256 * 1024**2)
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_once(args.output_root / "learning_commitment.json", commitment)
    learned = fit_update(model, arrays, batch_hash=commitment["report_hash"])
    consolidated = make_model(learned, memory, consolidation)
    if (
        any(hash_bytes(p.read_bytes()) != pins[str(p.resolve())] for p in inputs)
        or _head(source) != commitment["source_commit"]
        or hash_bytes(Path(__file__).read_bytes()) != commitment["source_hash"]
    ):
        raise ValueError("risk-margin learning inputs changed during optimization")
    write_once(args.output_root / "base_model.json.gz", learned)
    write_once(args.output_root / "model.json.gz", consolidated)
    print(
        dict(
            model_hash=consolidated["model_hash"],
            base_model_hash=learned["model_hash"],
            receipt=learned["learning_receipt"],
            new_physical_executions=0,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
