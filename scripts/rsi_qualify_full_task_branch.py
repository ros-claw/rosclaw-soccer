"""Prospective whole-remainder zero/repeat parity on all eighteen gold bodies.

Gold arrays are comparator-only. Forecasts run from OWN completed foot contact
and are persisted before the main body advances. No new action/policy gain.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_full_task_branch_inputs import prepare
from scripts.rsi_stage_release_contact_replay import digest


def require_same_arrays(a, b):
    if a.keys() != b.keys() or any(
        a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or a[k].tobytes() != b[k].tobytes()
        for k in a
    ):
        raise ValueError("whole body/control/NN arrays must match exactly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if (
        Path(__file__).resolve() != p.source / "scripts/rsi_qualify_full_task_branch.py"
        or not time.time() < p.cfg["deadline_unix"] <= time.time() + 86400
    ):
        raise ValueError("exact frozen bounded full-task zero qualification entry required")
    if args.preflight:
        print("FULL_TASK_ZERO18_READY_NO_NEW_FULL_MEAN_OR_PHYSICS", flush=True)
        return
    # Another already mature worker may continue; never overlap full mean constructors.
    mature = Path(p.cfg["mature_worker_root"]) / "course-0-arm-0-complete.json"
    if not mature.is_file():
        raise ValueError("other declared worker must have completed full mean allocation first")
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available before sole new full mean constructor required")
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.full_task_zero18_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        gold_complete_hash=p.gold_complete_hash,
        jobs=[{k: v for k, v in j.items() if k != "gold_trace"} for j in p.jobs],
        main_source_future_used_for_control=False,
        prediction_source_future_used_for_control=False,
        branches_triggered_from_actual_completed_foot_only=True,
        baseline_only=True,
        new_optimizer_updates=0,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    rows, main_steps, prediction_steps = [], 0, 0
    actor = branch = current = None
    history = {}
    try:
        import mujoco
        import torch

        from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
        from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
        from rosclaw_soccer.rsi.causal_impact_replay import CausalImpactReplay
        from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
        from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
        from rosclaw_soccer.rsi.full_task_branch import FullTaskBranch
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        torch.set_num_threads(1)
        print("ALLOCATING_ORIGINAL_FULL_MEAN_FOR_FULL_TASK_ZERO_PARITY", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same complete original mean required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        for job in p.jobs:
            i = job["index"]
            current = dict(index=i, phase="MAIN_CURRENT_BODY")
            write_once(p.root / f"case-{i}-started.json", current)
            case = p.cases[job["source_index"]]
            w = case["receipt"]["witnesses"]
            commitment = load_json_artifact(Path(w["commitment"]["path"]))
            report = load_json_artifact(Path(w["report"]["path"]))["stripped_document"]
            policy, decoder = factory.generate_and_bind_seed(
                seed=report["executed_motor_policy"]["step_motor_proof"]["model"]["seed"]
            )
            if (
                policy["policy_hash"] != job["original_policy_hash"]
                or policy["policy_hash"] != commitment["motor_policy_hash"]
            ):
                raise ValueError("same original sampled motor policy required")
            actor = CausalImpactReplay(
                model,
                world_hash=p.world_hash,
                kp=np.asarray(case["physics"]["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(case["physics"]["canonical_pd_kd"], dtype=np.float64),
                model_root=p.model_root,
                foundation=foundation,
                expected_foundation_hash=commitment["foundation_hash"],
                policy=policy,
                decoder=decoder,
                expected_policy_hash=policy["policy_hash"],
                course=tuple(job["course"]),
                profile=BoundedFirstTouchControl(),
                heading=FootHeadingContactProbe(case["job"]["heading_enabled"]),
                impact_enabled=False,
                constant_forward_approach=True,
            )
            history = {}
            predictions = []
            start = None
            for f in range(300):
                if (
                    time.time() >= p.cfg["deadline_unix"]
                    or main_steps + prediction_steps + 10 > 94840
                ):
                    raise RuntimeError("fixed full-task main/native deadline budget exhausted")
                result = actor.step(f)
                main_steps += 10
                for k, v in result.arrays.items():
                    history.setdefault(k, []).append(v.copy())
                # Comparator only AFTER actual action/physics, never input to either actor.
                require_same_arrays(result.arrays, {k: a[f] for k, a in job["gold_trace"].items()})
                if start is None and f >= 30 and np.any(result.arrays["force_n"][0, :2] > 1):
                    start = f + 1
                    if start != job["comparison_start_after_measured_foot_bin"]:
                        raise ValueError(
                            "own actual completed-foot boundary differs from comparison gold"
                        )
                    for bi in range(2):
                        current = dict(
                            index=i, phase="OWNED_ZERO_FUTURE", branch_index=bi, start_frame=start
                        )
                        branch = FullTaskBranch(
                            actor, world_hash=p.world_hash, action_name="baseline"
                        )
                        future = {}
                        for _ in range(start, 300):
                            if (
                                time.time() >= p.cfg["deadline_unix"]
                                or main_steps + prediction_steps + 10 > 94840
                            ):
                                raise RuntimeError(
                                    "fixed full-task future/native deadline budget exhausted"
                                )
                            predicted = branch.step()
                            prediction_steps += 10
                            for k, v in predicted.arrays.items():
                                future.setdefault(k, []).append(v.copy())
                        arrays = {k: np.asarray(v) for k, v in future.items()}
                        review = branch.review()
                        if (
                            review["new_native_attempted_steps"] != (300 - start) * 10
                            or review["new_native_integrated_steps"] != (300 - start) * 10
                        ):
                            raise ValueError("every remaining predicted native step required")
                        file = p.root / f"case-{i}-branch-{bi}-future.npz"
                        with file.open("xb") as stream:
                            np.savez_compressed(stream, **arrays)
                        # Only own already observed prefix + own simulated future.
                        measured = {
                            k: np.concatenate((np.asarray(history[k]), arrays[k]))
                            for k in (
                                "force_n",
                                "pelvis_z_per_substep_m",
                                "ball_position_after_step_m",
                            )
                        }
                        predicted_outcome = label_first_touch(measured).to_dict()
                        prediction = dict(
                            index=i,
                            branch_index=bi,
                            start_frame=start,
                            review=review,
                            arrays=dict(path=str(file), sha256=digest(file)),
                            predicted_outcome=predicted_outcome,
                            persisted_before_main_advances_frame=start,
                            source_future_used_for_prediction=False,
                        )
                        prediction["report_hash"] = hash_json(prediction)
                        write_once(p.root / f"case-{i}-branch-{bi}-prediction.json", prediction)
                        predictions.append((arrays, prediction))
                        branch = None
                    current = dict(index=i, phase="MAIN_CURRENT_BODY_AFTER_PREDICTIONS")
            trace = {k: np.asarray(v) for k, v in history.items()}
            file = p.root / f"case-{i}-main-trace.npz"
            with file.open("xb") as stream:
                np.savez_compressed(stream, **trace)
            if start is None or len(predictions) != 2:
                raise ValueError("all fixed cases require complete zero/repeat predictions")
            require_same_arrays(trace, job["gold_trace"])
            for future, prediction in predictions:
                require_same_arrays(future, {k: v[start:] for k, v in trace.items()})
                if prediction["predicted_outcome"] != label_first_touch(trace).to_dict():
                    raise ValueError(
                        "prospective full-task M0 prediction differs from actual future"
                    )
            main_review = actor.review()
            if (
                main_review["native_integrated_steps"] != 3000
                or main_review["native_attempted_steps"] != 3000
            ):
                raise ValueError("all actual main reconstruction steps required")
            row = dict(
                index=i,
                source_case_hash=job["source_case_hash"],
                course=job["course"],
                arrays=dict(path=str(file), sha256=digest(file)),
                main_review=main_review,
                predictions=[proof for _, proof in predictions],
                outcome=label_first_touch(trace).to_dict(),
                own_predictions_match_actual_full_future_all_arrays=True,
                source_gold_all300_arrays_match=True,
                repeated_zero_all_future_arrays_match=True,
                new_optimizer_updates=0,
                teacher_qualified=False,
                training_authorized=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"case-{i}-complete.json", row)
            rows.append(row)
            print("FULL_TASK_ZERO_NATIVE_PARITY_CASE_COMPLETE", i, start, flush=True)
            actor, history = None, {}
            del trace, predictions, policy, decoder
            gc.collect()
        factory._stable()
        if (
            main_steps != 54000
            or prediction_steps != 40840
            or len(rows) != 18
            or any(digest(n) != h for n, h in p.pins.items())
        ):
            raise ValueError("whole exact94840-step immutable zero/repeat qualification required")
        result = dict(
            schema="soccer.rsi.full_task_zero18_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            new_main_native_steps=main_steps,
            new_prediction_native_steps=prediction_steps,
            combined_new_native_steps=main_steps + prediction_steps,
            main_reconstruction_runs=18,
            prediction_runs=36,
            executed_control_foundation_pairs=(main_steps + prediction_steps) // 10,
            initialization_foundation_parity_pairs=4,
            new_optimizer_updates=0,
            baseline_whole_task_prediction_qualified_on_this_consumed_scope=True,
            nonzero_action_prediction_qualified=False,
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(p.root / "complete.json", result)
    except BaseException as error:
        if history:
            with (p.root / "failure-main-partial-trace.npz").open("xb") as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in history.items()})
        write_once(
            p.root / "failure.json",
            dict(
                current=current,
                completed_cases=len(rows),
                confirmed_returned_main_steps=main_steps,
                confirmed_returned_prediction_steps=prediction_steps,
                current_main_cumulative_attempted_steps=0
                if actor is None
                else actor.attempted_steps,
                current_branch_cumulative_attempted_steps=0
                if branch is None
                else branch._owned.attempted_steps,
                cumulative_attempt_fields_not_additive_to_confirmed_returned_steps=True,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
