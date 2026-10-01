"""Audit complete physical exploration, reconstruct latent actions, update actor/critic."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.contact_motor_evidence import audit_motor_execution
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.step_motor_ppo import fit_update
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exploration-root", "teacher-root", "pilot-summary", "output-root", "step-model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    summary = _sealed(args.exploration_root / "training_summary.json")
    pilot = _sealed(args.pilot_summary)
    commitment = summary["commitment"]
    warm = json.loads(args.step_model.read_text())
    if (
        commitment["base_model_hash"] != warm["model_hash"]
        or commitment["pilot_hash"] != pilot["report_hash"]
        or commitment["partition"] != "TRAIN_CONSUMED"
        or summary["exploration_executions"] != sum(len(r["samples"]) for r in summary["rows"])
        or summary["promotion_authorized"] is not False
    ):
        parser.error("complete consumed physical rollout summary required")
    # v316 asset identity is checked against the already sealed pilot and parents.
    contract = {**commitment, "asset_hash": pilot["commitment"]["asset_hash"]}
    observations, actions, logp, returns, scales, group_ids = [], [], [], [], [], []
    records: list[dict[str, Any]] = []
    for row in summary["rows"]:
        seed, lane = row["seed"], row["lane"]
        parent = _sealed(
            args.exploration_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
        )
        if (
            parent["report_hash"] != row["parent_report_hash"]
            or parent["asset_hash"] != contract["asset_hash"]
        ):
            raise ValueError("exploration parent asset or identity changed")
        for sampled in row["samples"]:
            sample = sampled["sample"]
            folder = args.exploration_root / f"seed{seed}-lane{lane}-sample-{sample}-actor"
            raw = _sealed(folder / "report.json")
            measured = _outcome(folder, raw["contact_motor_policy_hash"], contract)
            outcome = measured["outcome"]
            if (
                raw["report_hash"] != sampled["report_hash"]
                or raw["parent_report_hash"] != parent["report_hash"]
                or raw["training_course_seed"] != seed
                or raw["single_course_lane"] != lane
                or raw["environments"][0]["course"] != parent["environments"][0]["course"]
                or any(outcome[k] != sampled[k] for k in outcome)
            ):
                raise ValueError("exploration physics outcome differs from sealed learning record")
            audit_motor_execution(folder, raw)
            view = raw["contact_motor_policy"]["step_motor_proof"]["model"]
            decoder = CompiledStepMotor.from_legacy_preview(raw["contact_motor_policy"])
            if (
                view["model_hash"] != sampled["view_hash"]
                or view["training_sampling"]["base_model_hash"] != warm["model_hash"]
            ):
                raise ValueError("exploration used a different mean actor")
            group = len(records)
            order = raw["taskspace_joint_order"]
            ids = [order.index(n) for n in G1_DDS_JOINT_NAMES]
            with (
                np.load(folder / "body_trace.npz", allow_pickle=False) as body,
                np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as swing,
                np.load(folder / "contact_motor_trace.npz", allow_pickle=False) as motor,
                np.load(folder / "trace.npz", allow_pickle=False) as physics,
            ):
                for frame in range(30, 300):
                    x = features_at_frame(
                        body,
                        frame=frame,
                        nominal_target=swing["executed_taskspace_joint_target_rad"][frame, 0, ids],
                        previous=motor["applied_joint_delta_rad"][frame - 1, 0],
                        previous_contact_forces=physics["ball_body_contact_force_peak_n"][
                            frame - 1, 0
                        ],
                    )
                    z, density = decoder.latent_sample(x, frame)
                    observations.append(x)
                    actions.append(z)
                    logp.append(density)
                    returns.append(terminal_return(outcome))
                    scales.append(view["training_sampling"]["std_raw"])
                    group_ids.append(group)
            records.append(
                dict(
                    seed=seed,
                    lane=lane,
                    sample=sample,
                    report_hash=raw["report_hash"],
                    outcome=outcome,
                )
            )
    teacher_manifest = _sealed(args.teacher_root / "manifest.json")
    teacher_path = args.teacher_root / "step_motor_teachers.npz"
    if (
        teacher_manifest["data_hash"] != hash_bytes(teacher_path.read_bytes())
        or teacher_manifest["report_hash"] != warm["training_bank_hash"]
    ):
        raise ValueError("warm-start teacher bank changed")
    arrays: dict[str, Any] = dict(
        observation=np.asarray(observations),
        latent_action=np.asarray(actions),
        old_log_probability=np.asarray(logp),
        terminal_return=np.asarray(returns),
        std_raw=np.asarray(scales),
        trajectory_index=np.asarray(group_ids),
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "rollouts.npz"
    np.savez_compressed(path, **arrays)
    batch = dict(
        schema="soccer.rsi.audited_step_motor_rollout_bank.v1",
        partition="TRAIN_CONSUMED",
        exploration_summary_hash=summary["report_hash"],
        teacher_manifest_hash=teacher_manifest["report_hash"],
        records=records,
        physical_rollout_count=len(records),
        frame_sample_count=len(observations),
        independent_contexts=len(summary["rows"]),
        data_hash=hash_bytes(path.read_bytes()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    batch["report_hash"] = hash_json(batch)
    write_once(args.output_root / "rollout_manifest.json", batch)
    with np.load(teacher_path, allow_pickle=False) as teachers:
        positive = teachers["actor_supervision_mask"]
        # All consumed successful teachers regularize learning; none are claimed fresh.
        updated = fit_update(
            warm,
            arrays,
            batch_hash=batch["report_hash"],
            teacher_features=teachers["observation"][positive],
        )
    write_once(args.output_root / "model.json", updated)
    print(
        json.dumps(dict(model_hash=updated["model_hash"], receipt=updated["learning_receipt"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
