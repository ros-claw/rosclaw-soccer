"""Reconstruct actual current-parent exploration, then train the residual MLP."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.output_memory_motor_learning import fit_update
from rosclaw_soccer.rsi.output_memory_step_motor import SAMPLING_SCHEMA, validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_failed_step_courses import failure_rows, sampling_seed
from scripts.rsi_fit_protected_phase_step_motor import gpu_observations
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def checked_curriculum(
    model: dict[str, Any],
    summary: dict[str, Any],
    bank: dict[str, Any],
    review: dict[str, Any],
    declared: dict[str, Any],
) -> list[list[int]]:
    failures = failure_rows(bank, review, arm="candidate")
    commitment = summary["commitment"]
    courses = [[r["seed"], r["lane"]] for r in failures]
    samples = commitment["samples_per_course"]
    if type(samples) is not int or not 4 <= samples <= 16:
        raise ValueError("bounded declared sample count required")
    count = len(courses) * samples
    reference_hash = (
        model["model_hash"] if model.get("generation", 0) > 0 else model["parent_model_hash"]
    )
    if (
        summary["schema"] != "soccer.rsi.output_memory_failure_exploration.v1"
        or commitment != declared
        or commitment["schema"] != "soccer.rsi.output_memory_exploration_commitment.v1"
        or commitment["behavior_kind"] != "OUTPUT_MEMORY_CURRENT_PARENT"
        or commitment["partition"] != "TRAIN_CONSUMED"
        or commitment["courses"] != courses
        or len(summary["rows"]) != len(courses)
        or commitment["bank_hash"] != bank["report_hash"]
        or commitment["bank_review_hash"] != review["report_hash"]
        or commitment["base_model_hash"] != model["model_hash"]
        or commitment["warm_model_hash"] != model["model_hash"]
        or commitment["frozen_parent_model_hash"] != model["parent_model_hash"]
        or bank["commitment"]["model_hash"] != reference_hash
        or (
            model.get("generation", 0) > 0
            and (
                commitment.get("failure_reference_model_hash") != reference_hash
                or type(commitment.get("sampling_generation")) is not int
                or commitment.get("sampling_generation") != model["generation"]
                or commitment.get("sampling_seed_namespace") != "GENERATION_STRIDE_100000"
            )
        )
        or commitment["std_raw"] != 0.1
        or len(commitment["sampling_view_hashes"]) != count
        or summary["exploration_executions"] != count
        or summary["physical_executions"] != count + 2 * len(courses)
        or summary["independent_contexts"] != len(courses)
        or any(
            obj.get(k) is not False
            for obj in (summary, commitment)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete current-parent on-policy failure curriculum required")
    return courses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exploration-root", "parent-bank-root", "model", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    validate_model(model)
    summary = _sealed(args.exploration_root / "training_summary.json")
    bank = _sealed(args.parent_bank_root / "validation_summary.json")
    review = _sealed(args.parent_bank_root / "independent_review.json")
    commitment = summary["commitment"]
    courses = checked_curriculum(
        model,
        summary,
        bank,
        review,
        json.loads((args.exploration_root / "commitment.json").read_text()),
    )
    records: list[dict[str, Any]] = []
    chunks: dict[str, list[Any]] = {
        k: []
        for k in (
            "observation",
            "phase_index",
            "latent_action",
            "old_log_probability",
            "terminal_return",
            "std_raw",
            "trajectory_index",
        )
    }
    for i, (course, row) in enumerate(zip(courses, summary["rows"], strict=True)):
        seed, lane = course
        if (
            row["index"] != i
            or [row["seed"], row["lane"]] != course
            or len(row["samples"]) != commitment["samples_per_course"]
        ):
            raise ValueError("declared course/sample identity changed")
        parent_folder = args.exploration_root / f"seed{seed}-lane{lane}-reproduction-parent"
        parent = _sealed(parent_folder / "report.json")
        audit_lateral_approach(parent_folder)
        if (
            parent["report_hash"] != row["parent_report_hash"]
            or parent["training_course_seed"] != seed
            or parent["single_course_lane"] != lane
            or parent["source_hash"] != commitment["runner_hash"]
            or parent["asset_hash"] != commitment["asset_hash"]
        ):
            raise ValueError("physical parent changed")
        for s, sample in enumerate(row["samples"]):
            folder = args.exploration_root / f"seed{seed}-lane{lane}-sample-{s}-actor"
            decoded: list[Any] = []
            raw = _sealed(folder / "report.json")
            checked = _outcome(
                folder, raw["contact_motor_policy_hash"], commitment, decoder_sink=decoded
            )
            if checked["report"] != raw:
                raise ValueError("physical report changed during reconstruction")
            x, phase, raw = gpu_observations(folder, raw)
            view = raw["contact_motor_policy"]["step_motor_proof"]["model"]
            if (
                len(decoded) != 1
                or view["schema"] != SAMPLING_SCHEMA
                or view["mean_model"] != model
                or view["seed"] != sampling_seed(i, s, generation=model.get("generation", 0))
                or view["model_hash"] != sample["view_hash"]
                or view["model_hash"]
                != commitment["sampling_view_hashes"][i * commitment["samples_per_course"] + s]
                or sample["sample"] != s
                or raw["report_hash"] != sample["report_hash"]
                or raw["training_course_seed"] != seed
                or raw["single_course_lane"] != lane
                or raw["parent_report_hash"] != parent["report_hash"]
                or raw["environments"][0]["course"] != parent["environments"][0]["course"]
                or raw["source_hash"] != commitment["runner_hash"]
                or raw["asset_hash"] != commitment["asset_hash"]
            ):
                raise ValueError("sample is not the current model's actual physical rollout")
            # The complete motor, physics, swing and approach audits ran above;
            # no rollout reward label substitutes for these measured outcomes.
            outcome = checked["outcome"]
            if any(sample[k] != v for k, v in outcome.items()):
                raise ValueError("sample outcome differs from independently measured physics")
            draws = [
                decoded[0].latent_sample(v, frame, int(p))
                for frame, (v, p) in enumerate(zip(x, phase, strict=True), start=30)
            ]
            group = len(records)
            values = dict(
                observation=x,
                phase_index=phase,
                latent_action=np.stack([v[0] for v in draws]),
                old_log_probability=np.asarray([v[1] for v in draws]),
                terminal_return=np.full(270, terminal_return(outcome)),
                std_raw=np.full(270, view["std_raw"]),
                trajectory_index=np.full(270, group, dtype=np.int64),
            )
            for key in chunks:
                chunks[key].append(values[key])
            records.append(
                dict(
                    group=group,
                    seed=seed,
                    lane=lane,
                    sample=s,
                    backend="IsaacLab",
                    folder=str(folder),
                    report_hash=raw["report_hash"],
                    outcome=outcome,
                )
            )
            print(f"AUDITED_CURRENT_PARENT_ROLLOUT course={i} sample={s} group={group}", flush=True)
    combined = {k: np.concatenate(v) for k, v in chunks.items()}
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "rollouts.npz"
    np.savez_compressed(path, **combined)
    manifest = dict(
        schema="soccer.rsi.output_memory_on_policy_bank.v1",
        partition="TRAIN_CONSUMED",
        source_summary_hash=summary["report_hash"],
        parent_model_hash=model["model_hash"],
        records=records,
        physical_rollout_count=len(records),
        frame_sample_count=len(combined["observation"]),
        independent_contexts=len(courses),
        data_hash=hash_bytes(path.read_bytes()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    write_once(args.output_root / "rollout_manifest.json", manifest)
    learned = fit_update(model, combined, batch_hash=manifest["report_hash"])
    write_once(args.output_root / "model.json", learned)
    print(
        json.dumps(dict(model_hash=learned["model_hash"], receipt=learned["learning_receipt"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
