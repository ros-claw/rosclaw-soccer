"""Audit actual AR(1) physical exploration and train its conditional actor."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.smooth_memory_learning import fit_update
from rosclaw_soccer.rsi.smooth_memory_motor import SAMPLING_SCHEMA, validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_failed_step_courses import failure_rows
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
    selection = commitment["course_selection"]
    if selection == "FIRST_FOUR_CURRENT_PARENT_FAILURES_IN_SEALED_ORDER":
        if len(failures) < 4:
            raise ValueError("four preregistered failure courses required")
        chosen = failures[:4]
    elif selection == "ALL_CURRENT_PARENT_FAILURES_IN_SEALED_ORDER":
        chosen = failures
    else:
        raise ValueError("declared immutable failure selection required")
    courses = [[r["seed"], r["lane"]] for r in chosen]
    samples = commitment["samples_per_course"]
    if type(samples) is not int or not 4 <= samples <= 16:
        raise ValueError("bounded declared sample count required")
    count = len(courses) * samples
    if model.get("generation", 0) > 0:
        reference_hash = model["model_hash"]
    else:
        parent = model["frozen_parent"]
        reference_hash = (
            parent["model_hash"]
            if parent.get("generation", 0) > 0
            else parent["frozen_parent"]["model_hash"]
        )
    if (
        summary["schema"] != "soccer.rsi.smooth_memory_failure_exploration.v1"
        or commitment != declared
        or commitment["schema"] != "soccer.rsi.smooth_memory_exploration_commitment.v1"
        or commitment["behavior_kind"] != "OUTPUT_MEMORY_CURRENT_PARENT_AR1"
        or commitment["partition"] != "TRAIN_CONSUMED"
        or commitment["courses"] != courses
        or len(summary["rows"]) != len(courses)
        or commitment["bank_hash"] != bank["report_hash"]
        or commitment["bank_review_hash"] != review["report_hash"]
        or commitment["base_model_hash"] != model["model_hash"]
        or commitment["warm_model_hash"] != model["model_hash"]
        or commitment["frozen_parent_model_hash"] != model["parent_model_hash"]
        or bank["commitment"]["model_hash"] != reference_hash
        or commitment["failure_reference_model_hash"] != reference_hash
        or commitment["failure_reference_total_courses"] != len(failures)
        or commitment["std_raw"] != 0.1
        or commitment["sampling_rho"] != 0.9
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
        raise ValueError("complete current-parent conditional exploration bank required")
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
            raise ValueError("course/sample identity changed")
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
            raw = _sealed(folder / "report.json")
            decoded: list[Any] = []
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
                or view["rho"] != 0.9
                or view["std_raw"] != 0.1
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
                raise ValueError("sample is not this declared physical AR policy")
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
            print(f"AUDITED_AR_PHYSICAL_ROLLOUT course={i} sample={s} group={group}", flush=True)
    combined = {k: np.concatenate(v) for k, v in chunks.items()}
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "rollouts.npz"
    np.savez_compressed(path, **combined)
    manifest = dict(
        schema="soccer.rsi.smooth_memory_on_policy_bank.v1",
        partition="TRAIN_CONSUMED",
        source_summary_hash=summary["report_hash"],
        parent_model_hash=model["model_hash"],
        sampling_rho=0.9,
        candidate_previous_mean_required=True,
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
    learned = fit_update(model, combined, batch_hash=manifest["report_hash"], rho=0.9)
    write_once(args.output_root / "model.json", learned)
    print(json.dumps(dict(model_hash=learned["model_hash"], receipt=learned["learning_receipt"])))


if __name__ == "__main__":
    main()
