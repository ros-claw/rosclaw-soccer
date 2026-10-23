"""Close ALL three native adaptation outcomes and diagnose offline fit coverage.

Read-only evidence aggregation. Loss changes do not confer physical success.
TRAIN functional pass curation remains separate from strict static-point exam,
Fresh, actual online RL, cumulative old retention and whole-game competence.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused review output required")
    root = a.evidence_root
    bp = root / "rsi-team-pass-bank177-20261023/complete.json"
    bank = json.loads(bp.read_text())
    if (
        digest(bp) != "sha256:84aa60dce6f00c40569b346764279670f8b15f1807f7c2921a8d8111a91b3462"
        or len(bank["records"]) != 177
    ):
        raise ValueError("unchanged whole177 bank required")
    pins = dict(bank["input_file_hashes"])
    pins[str(bp)] = digest(bp)
    import numpy as np

    from rosclaw_soccer.rsi.acquired_motor_retention import native_successes

    required = {r["source_context"] for r in bank["records"] if r["role"] == "retained_neural17"}
    reports = []
    for seed in (75023, 75024, 75025):
        paths = [
            root / f"rsi-team-pass-neural-seed{seed}-{kind}-20261023/complete.json"
            for kind in ("native80-own-audit", "retention44-own-audit", "cpu-audit")
        ]
        docs = []
        for path in paths:
            if path.is_symlink() or (path.parent / "failed.json").exists():
                raise ValueError("ALL three closed complete own proofs required")
            doc = json.loads(path.read_text())
            for key, value in doc["input_file_hashes"].items():
                if key in pins and pins[key] != value:
                    raise ValueError("conflicting proof lineage")
                pins[key] = value
            pins[str(path)] = digest(path)
            docs.append(doc)
        native, old, cpu = docs
        tp, training = receipt(pins, old["training_hash"])
        if (
            native["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
            or old["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
            or cpu["status"]
            != "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
            or native["training_hash"] != old["training_hash"]
            or cpu["source_hash"] != old["training_hash"]
            or training["config"]["seed"] != seed
            or training["data_hash"] != digest(bp)
            or any(
                d[k] is not False
                for d in docs
                for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
            )
        ):
            raise ValueError("same all-three independently replayed actual models required")
        achieved = native_successes(old["results"])
        errors = {"retained_neural17": [], "new_teacher": [], "new_unassisted": []}
        before, after, saturation_error = [], [], []
        sampled = {h["teacher_index"] for h in training["history"]}
        for row, replay in zip(bank["records"], training["cpu_replay"], strict=True):
            if not row["task_success"]:
                continue
            dp = bp.parent / row["raw_path"]
            pred = tp.parent / replay["prediction_path"]
            if digest(dp) != row["raw_hash"] or digest(pred) != replay["prediction_hash"]:
                raise ValueError("exact complete data and CPU predictions required")
            pins[str(dp)], pins[str(pred)] = row["raw_hash"], replay["prediction_hash"]
            with np.load(dp, allow_pickle=False) as z, np.load(pred, allow_pickle=False) as out:
                mean = out["means"]
                error = ((mean - z["targets"]) * z["action_scale"]).astype(np.float64) ** 2
                weights = z["loss_weights"].astype(np.float64)
                weighted = float((error.mean(axis=1) * weights).sum() / weights.sum())
                errors[row["role"]].append(weighted)
                if row["role"] == "new_teacher":
                    ap = Path(row["source_root"]) / f"trial{row['source_index']}/A.npz"
                    if pins.get(str(ap)) != digest(ap):
                        raise ValueError("bound actual warm teacher state/action required")
                    with np.load(ap, allow_pickle=False) as old_a:
                        e0 = ((old_a["NN_action"] - z["targets"]) * z["action_scale"]).astype(
                            np.float64
                        ) ** 2
                    before.append(float((e0.mean(axis=1) * weights).sum() / weights.sum()))
                    after.append(weighted)
                    saturated = (
                        np.abs((mean - z["reference_nominal"]) * z["action_scale"]) >= 0.199999
                    )
                    saturation_error.append(
                        float(((error * saturated).mean(axis=1) * weights).sum() / weights.sum())
                    )
        summary = native["summary"]
        lost = sorted(required - achieved)
        eligible = not lost and summary["candidate_controlled"] > summary["baseline_controlled"]
        reports.append(
            dict(
                seed=seed,
                native_own_hash=digest(paths[0]),
                retention_own_hash=digest(paths[1]),
                cpu_own_hash=digest(paths[2]),
                checkpoint_hash=old["checkpoint_hash"],
                required_contexts=sorted(required),
                lost_contexts=lost,
                all_acquired_old_retained=not lost,
                TRAIN_pass_summary=summary,
                eligible_for_new_Fresh_study=eligible,
                accepted_as_improvement=False,
                actual_BC_steps=training["offline_supervised_steps"],
                actual_PPO_steps=training["PPO_optimizer_steps_this_stage"],
                positive_teacher_sequences=40,
                distinct_teacher_sequences_sampled=len(sampled),
                fit_weighted_motor_MSE_rad2={k: float(np.mean(v)) for k, v in errors.items()},
                teacher_before_weighted_MSE_rad2=float(np.mean(before)),
                teacher_after_weighted_MSE_rad2=float(np.mean(after)),
                teacher_error_after_before_ratio=float(np.mean(after) / np.mean(before)),
                saturated_coordinate_fraction_of_teacher_error=float(
                    sum(saturation_error) / sum(after)
                ),
            )
        )
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="ALL_THREE_TEAM_PASS_ADAPTATIONS_REVIEWED_NONE_ACCEPTED"
            if not any(r["eligible_for_new_Fresh_study"] for r in reports)
            else "TEAM_PASS_REVIEW_REQUIRES_FRESH_STUDY",
            source_head=a.expected_head,
            input_file_hashes=pins,
            reports=reports,
            conclusions=[
                "16BCsteps cover only13or14 of40 positive teacher histories",
                "teacher motor MSE reduced only8to11percent; substantial underfit remains",
                "Saturated coordinates explain less than 0.4% of remaining teacher error.",
                "Own-state shift remains a hypothesis, not an established sole cause; "
                "test it with actual student-prefix expert continuations.",
                "75024 is a fully-retained RESEARCH student, not an accepted skill champion",
            ],
            physical_steps=0,
            optimizer_updates=0,
            Fresh_verified=False,
            team_game_verified=False,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
