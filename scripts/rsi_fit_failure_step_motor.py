"""Audit expanded failure physics and fit the existing protected actor/critic.

No new motor math, outcome routing or fresh data. Every added feature and latent
Gaussian action is reconstructed from actual closed-loop traces before fitting.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.kernel_guarded_step_network import fit_update, validate_model
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_failed_step_courses import failure_rows
from scripts.rsi_fit_protected_phase_step_motor import cpu_features, gpu_features
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "exploration-root",
        "bank-physics-root",
        "prior-rollouts",
        "prior-gpu-exploration",
        "zero-model",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--optimizer", choices=("minibatch", "full-batch-backtracking"), default="minibatch"
    )
    args = parser.parse_args()
    summary = _sealed(args.exploration_root / "training_summary.json")
    commitment = summary["commitment"]
    parent = json.loads(args.zero_model.read_text())
    validate_model(parent)
    bank = _sealed(args.bank_physics_root / "validation_summary.json")
    reviewed = _sealed(args.bank_physics_root / "independent_review.json")
    expected = failure_rows(bank, reviewed)
    courses = [[r["seed"], r["lane"]] for r in expected]
    base = parent["encoder"]["base_model"]
    count = len(courses) * commitment["samples_per_course"]
    if (
        summary["schema"] != "soccer.rsi.failed_step_course_exploration.v1"
        or commitment != json.loads((args.exploration_root / "commitment.json").read_text())
        or commitment["partition"] != "TRAIN_CONSUMED"
        or commitment["courses"] != courses
        or commitment["bank_hash"] != bank["report_hash"]
        or commitment["bank_review_hash"] != reviewed["report_hash"]
        or commitment["base_model_hash"] != base["model_hash"]
        or parent["generation"] != 0
        or summary["exploration_executions"] != count
        or summary["physical_executions"] != count + 2 * len(courses)
        or summary["independent_contexts"] != len(courses)
        or len(summary["rows"]) != len(courses)
        or any(summary.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
        or len(commitment["sampling_view_hashes"]) != count
    ):
        raise ValueError("complete unchanged failure curriculum and actual warm behavior required")
    prior = _sealed(args.prior_rollouts / "rollout_manifest.json")
    prior_path = args.prior_rollouts / "rollouts.npz"
    if (
        prior["partition"] != "TRAIN_CONSUMED"
        or prior["physical_rollout_count"] != 64
        or prior["frame_sample_count"] != 17280
        or prior["data_hash"] != hash_bytes(prior_path.read_bytes())
        or any(prior.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
        or len(prior["records"]) != 64
    ):
        raise ValueError("complete original dual-backend on-policy bank required")
    with np.load(prior_path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    records = []
    for i, record in enumerate(prior["records"]):
        if record["backend"] == "MuJoCo":
            x, phase, raw = cpu_features(Path(record["folder"]), record)
            policy = raw["executed_motor_policy"]
            model = policy["step_motor_proof"]["model"]["base_model"]
            decoder = CompiledStepMotor(policy)
        else:
            folder = args.prior_gpu_exploration / (
                f"seed{record['seed']}-lane{record['lane']}-sample-{record['sample']}-actor"
            )
            x, phase, raw = gpu_features(folder)
            policy = raw["contact_motor_policy"]
            model = policy["step_motor_proof"]["model"]
            decoder = CompiledStepMotor.from_legacy_preview(policy)
        draws = [decoder.latent_sample(obs, frame) for frame, obs in enumerate(x, start=30)]
        ids = slice(i * 270, (i + 1) * 270)
        if (
            raw["report_hash"] != record["report_hash"]
            or model["training_sampling"]["base_model_hash"] != base["model_hash"]
            or not np.array_equal(x, arrays["observation"][ids])
            or not np.array_equal(phase, arrays["phase_index"][ids])
            or not np.array_equal(np.stack([v[0] for v in draws]), arrays["latent_action"][ids])
            or not np.array_equal(
                np.asarray([v[1] for v in draws]), arrays["old_log_probability"][ids]
            )
            or not np.all(arrays["terminal_return"][ids] == terminal_return(record["outcome"]))
            or not np.all(arrays["std_raw"][ids] == model["training_sampling"]["std_raw"])
            or not np.all(arrays["trajectory_index"][ids] == i)
        ):
            raise ValueError("prior on-policy physical observations or lineage changed")
        records.append(record)
    chunks: dict[str, list[Any]] = {k: [v] for k, v in arrays.items()}
    for i, (course, row) in enumerate(zip(courses, summary["rows"], strict=True)):
        seed, lane = course
        if (
            row["index"] != i
            or [row["seed"], row["lane"]] != course
            or len(row["samples"]) != commitment["samples_per_course"]
        ):
            raise ValueError("course or sample identity changed")
        parent_report = _sealed(
            args.exploration_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
        )
        if parent_report["report_hash"] != row["parent_report_hash"]:
            raise ValueError("measured course parent changed")
        for s, sampled in enumerate(row["samples"]):
            j = i * commitment["samples_per_course"] + s
            folder = args.exploration_root / f"seed{seed}-lane{lane}-sample-{s}-actor"
            x, phase, raw = gpu_features(folder)
            measured = _outcome(folder, raw["contact_motor_policy_hash"], commitment)["outcome"]
            policy = raw["contact_motor_policy"]
            model = policy["step_motor_proof"]["model"]
            if (
                sampled["sample"] != s
                or sampled["report_hash"] != raw["report_hash"]
                or sampled["view_hash"] != model["model_hash"]
                or model["model_hash"] != commitment["sampling_view_hashes"][j]
                or model["training_sampling"]["base_model_hash"] != base["model_hash"]
                or raw["parent_report_hash"] != parent_report["report_hash"]
                or raw["environments"][0]["course"] != parent_report["environments"][0]["course"]
                or any(sampled[k] != v for k, v in measured.items())
            ):
                raise ValueError(
                    "expanded rollout differs from independently reconstructed physics"
                )
            decoder = CompiledStepMotor.from_legacy_preview(policy)
            draws = [decoder.latent_sample(obs, frame) for frame, obs in enumerate(x, start=30)]
            group = len(records)
            chunk = dict(
                observation=x,
                phase_index=phase,
                latent_action=np.stack([v[0] for v in draws]),
                old_log_probability=np.asarray([v[1] for v in draws]),
                terminal_return=np.full(270, terminal_return(measured)),
                std_raw=np.full(270, model["training_sampling"]["std_raw"]),
                trajectory_index=np.full(270, group, dtype=np.int64),
            )
            for key in chunks:
                chunks[key].append(chunk[key])
            records.append(
                dict(
                    backend="IsaacLab",
                    seed=seed,
                    lane=lane,
                    sample=s,
                    folder=str(folder),
                    report_hash=raw["report_hash"],
                    outcome=measured,
                    group=group,
                )
            )
            print(f"AUDITED_FAILURE_ROLLOUT course={i} sample={s} group={group}", flush=True)
    combined = {k: np.concatenate(v) for k, v in chunks.items()}
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "rollouts.npz"
    np.savez_compressed(path, **combined)
    manifest = dict(
        schema="soccer.rsi.expanded_failure_motor_rollout_bank.v1",
        partition="TRAIN_CONSUMED",
        prior_bank_hash=prior["report_hash"],
        exploration_summary_hash=summary["report_hash"],
        parent_model_hash=parent["model_hash"],
        records=records,
        physical_rollout_count=len(records),
        frame_sample_count=len(combined["observation"]),
        independent_contexts=len({(r["seed"], r["lane"]) for r in records}),
        backend_count=2,
        data_hash=hash_bytes(path.read_bytes()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    write_once(args.output_root / "rollout_manifest.json", manifest)
    if args.optimizer == "full-batch-backtracking":
        from rosclaw_soccer.rsi.kernel_full_batch_learning import fit_update as full_batch_update

        candidate = full_batch_update(parent, combined, batch_hash=manifest["report_hash"])
    else:
        candidate = fit_update(parent, combined, batch_hash=manifest["report_hash"])
    write_once(args.output_root / "model.json", candidate)
    print(
        json.dumps(dict(model_hash=candidate["model_hash"], receipt=candidate["learning_receipt"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
