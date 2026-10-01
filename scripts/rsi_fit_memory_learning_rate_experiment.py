"""One predeclared optimizer-rate ablation on a complete previously audited bank.

No new observations, relabeling, fabricated executions or policy promotion.
The input must have completed serial/parallel full-data equivalence review.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.output_memory_accelerated_learning import fit_update
from rosclaw_soccer.rsi.output_memory_step_motor import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "learning-root",
        "behavior-model",
        "audit-source-root",
        "equivalence-report",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=4e-4, choices=(4e-4,))
    args = parser.parse_args()
    model = json.loads(args.behavior_model.read_text())
    prior = json.loads((args.learning_root / "model.json").read_text())
    validate_model(model)
    validate_model(prior)
    manifest = _sealed(args.learning_root / "rollout_manifest.json")
    proof = _sealed(args.equivalence_report)
    path = args.learning_root / "rollouts.npz"
    if (
        model["generation"] != 0
        or prior["generation"] != 1
        or manifest["schema"] != "soccer.rsi.output_memory_on_policy_bank.v1"
        or manifest["partition"] != "TRAIN_CONSUMED"
        or manifest["parent_model_hash"] != model["model_hash"]
        or prior["learning_receipt"]["learner_parent_hash"] != model["model_hash"]
        or prior["learning_receipt"]["physical_batch_hash"] != manifest["report_hash"]
        or proof["schema"] != "soccer.rsi.parallel_learning_equivalence.v1"
        or proof["arrays_bitwise_equal"] is not True
        or proof["policies_equal_except_manifest_binding"] is not True
        or prior["model_hash"] not in proof["model_hashes"]
        or manifest["report_hash"] not in proof["manifest_hashes"]
        or manifest["data_hash"] not in proof["data_hashes"]
        or proof["parent_model_hash"] != model["model_hash"]
        or proof["source_summary_hash"] != manifest["source_summary_hash"]
        or manifest["data_hash"] != hash_bytes(path.read_bytes())
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
            for obj in (manifest, model, prior, proof)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete independently reviewed unchanged physical bank required")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    count = len(manifest["records"])
    if (
        count == 0
        or manifest["physical_rollout_count"] != count
        or manifest["frame_sample_count"] != count * 270
        or proof["physical_rollout_count"] != count
        or proof["frame_sample_count"] != count * 270
        or [r["group"] for r in manifest["records"]] != list(range(count))
        or not np.array_equal(arrays["trajectory_index"], np.repeat(np.arange(count), 270))
        or len(arrays["observation"]) != count * 270
    ):
        raise ValueError("complete ordered nonselected physical trajectories required")
    for r in manifest["records"]:
        values = arrays["terminal_return"][arrays["trajectory_index"] == r["group"]]
        if not np.all(values == terminal_return(r["outcome"])):
            raise ValueError("actual terminal outcomes changed")
    args.output_root.mkdir(parents=True, exist_ok=False)
    commitment = dict(
        schema="soccer.rsi.memory_learning_rate_experiment.v1",
        partition="TRAIN_CONSUMED",
        manifest_hash=manifest["report_hash"],
        data_hash=manifest["data_hash"],
        equivalence_review_hash=proof["report_hash"],
        behavior_model_hash=model["model_hash"],
        comparison_candidate_hash=prior["model_hash"],
        learning_rate=args.learning_rate,
        optimizer_steps_ceiling=160,
        raw_residual_cap=0.05,
        cumulative_kl_ceiling=0.005,
        objective_and_critic_unchanged=True,
        all_successes_and_failures_used=True,
        physical_executions_added=0,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    commitment["report_hash"] = hash_json(commitment)
    write_once(args.output_root / "commitment.json", commitment)
    learned = fit_update(
        model, arrays, batch_hash=manifest["report_hash"], learning_rate=args.learning_rate
    )
    if (
        model != json.loads(args.behavior_model.read_text())
        or manifest != _sealed(args.learning_root / "rollout_manifest.json")
        or manifest["data_hash"] != hash_bytes(path.read_bytes())
        or proof != _sealed(args.equivalence_report)
    ):
        raise ValueError("immutable audited learning inputs changed during optimization")
    write_once(args.output_root / "model.json", learned)
    print(
        json.dumps(dict(model_hash=learned["model_hash"], receipt=learned["learning_receipt"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
