"""Prospective nonzero full-task navigation parity on four consumed contexts.

Four options (zero, repeated zero, two braking caps) are predicted before the
baseline donor continues. Three fresh bodies then execute the matching leases.
No new optimizer, generalization, teacher promotion or autonomous team claim.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_navigation_recovery_inputs import ACTIONS, prepare
from scripts.rsi_read_full_task_zero import same_arrays
from scripts.rsi_stage_release_contact_replay import digest


def build_actor(p, job, factory, foundation, model, *, born_release_capture=False):
    from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
    from rosclaw_soccer.rsi.causal_impact_replay import CausalImpactReplay
    from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
    from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact

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
    return CausalImpactReplay(
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
        born_release_capture=born_release_capture,
    )


def archive(root, name, values):
    path = root / name
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return dict(path=str(path), sha256=digest(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    run_bank(p, preflight=args.preflight)


def run_bank(p, *, preflight, born_release=False):
    """Two explicit source-owned protocols, not a caller-supplied controller."""
    from rosclaw_soccer.rsi.born_postcontact_release import BORN_RELEASE_ACTIONS

    actions = ["baseline", *BORN_RELEASE_ACTIONS] if born_release else ACTIONS
    prediction_actions = actions if born_release else ["baseline", "baseline", *ACTIONS[1:]]
    main_budget, prediction_budget, contexts = (
        (42000, 8960, 1) if born_release else (36000, 14320, 4)
    )
    total_budget = main_budget + prediction_budget
    stem = "soccer.rsi.born_release14" if born_release else "soccer.rsi.navigation_recovery4"
    expected_profile = (
        "full-task-born-release14-v1" if born_release else "full-task-navigation-recovery4-v1"
    )
    born_fields = {
        "born_release_requested_offset_nm",
        "born_release_applied_offset_nm",
        "born_release_forecast_steps",
    }

    def base_arrays(values):
        if not born_release:
            return values
        if {k for k in values if k.startswith("born_release_")} != born_fields:
            raise ValueError("exact predeclared whole-episode born capture namespace required")
        return {k: v for k, v in values.items() if k not in born_fields}

    def indices(ai):
        return [ai] if born_release else [0, 1] if ai == 0 else [ai + 1]

    if (
        Path(__file__).resolve() != p.source / "scripts/rsi_qualify_navigation_recovery.py"
        or not time.time() < p.cfg["deadline_unix"] <= time.time() + 86400
        or p.cfg["profile"] != expected_profile
        or len(p.jobs) != contexts
    ):
        raise ValueError("exact frozen bounded navigation-recovery entry required")
    if preflight:
        print(
            "BORN_RELEASE14_READY_NO_NEW_ALLOCATION_OR_PHYSICS"
            if born_release
            else "NAVIGATION_RECOVERY4_READY_NO_NEW_ALLOCATION_OR_PHYSICS",
            flush=True,
        )
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available and operator-serialized full mean constructor required")
    p.root.mkdir()
    declaration = dict(
        schema=stem + "_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        jobs=[{k: v for k, v in j.items() if k != "gold_trace"} for j in p.jobs],
        prediction_actions=prediction_actions,
        measured_contact_trigger_only=True,
        recorded_future_used_for_control=False,
        new_optimizer_updates=0,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    main_steps = prediction_steps = 0
    actor = branch = current = None
    history = {}
    rows = []

    def budget():
        if (
            time.time() >= p.cfg["deadline_unix"]
            or main_steps + prediction_steps + 10 > total_budget
        ):
            raise RuntimeError("fixed navigation main/prediction native deadline budget exhausted")

    try:
        import mujoco
        import torch

        from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
        from rosclaw_soccer.rsi.born_navigation_recovery import BornNavigationRecovery
        from rosclaw_soccer.rsi.full_task_branch import FullTaskBranch
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        torch.set_num_threads(1)
        print(
            "ALLOCATING_ORIGINAL_FULL_MEAN_FOR_BORN_RELEASE_PARITY"
            if born_release
            else "ALLOCATING_ORIGINAL_FULL_MEAN_FOR_NONZERO_NAVIGATION_PARITY",
            flush=True,
        )
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("complete original full mean required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        for job in p.jobs:
            i = job["index"]
            predictions = []
            actuals = []
            for ai, action in enumerate(actions):
                current = dict(index=i, action_name=action, phase="ACTUAL_MAIN")
                write_once(p.root / f"case-{i}-arm-{ai}-started.json", current)
                clock = time.perf_counter()
                actor = (
                    build_actor(p, job, factory, foundation, model, born_release_capture=True)
                    if born_release
                    else build_actor(p, job, factory, foundation, model)
                )
                construction_seconds = time.perf_counter() - clock
                actual_control_seconds = 0.0
                start = None
                history = {}
                for f in range(300):
                    budget()
                    clock = time.perf_counter()
                    step = actor.step(f)
                    actual_control_seconds += time.perf_counter() - clock
                    main_steps += 10
                    for k, v in step.arrays.items():
                        history.setdefault(k, []).append(v.copy())
                    if start is None:
                        same_arrays(
                            base_arrays(step.arrays),
                            {k: a[f] for k, a in job["gold_trace"].items()},
                        )
                        if born_release and any(np.any(step.arrays[k]) for k in born_fields):
                            raise ValueError("no unobserved pre-foot born action may occur")
                    if start is None and f >= 30 and np.any(step.arrays["force_n"][0, :2] > 1):
                        start = f + 1
                        if start != job["comparison_start_after_measured_foot_bin"]:
                            raise ValueError(
                                "OWN measured completed-foot trigger differs from gold comparator"
                            )
                        if ai == 0:
                            for bi, option in enumerate(prediction_actions):
                                current = dict(
                                    index=i,
                                    action_name=option,
                                    phase="OWN_PREDICTION",
                                    branch_index=bi,
                                )
                                clock = time.perf_counter()
                                branch = FullTaskBranch(
                                    actor, world_hash=p.world_hash, action_name=option
                                )
                                branch_construction_seconds = time.perf_counter() - clock
                                predicted_control_seconds = 0.0
                                future = {}
                                for _ in range(start, 300):
                                    budget()
                                    clock = time.perf_counter()
                                    predicted = branch.step()
                                    predicted_control_seconds += time.perf_counter() - clock
                                    prediction_steps += 10
                                    for k, v in predicted.arrays.items():
                                        future.setdefault(k, []).append(v.copy())
                                values = {k: np.asarray(v) for k, v in future.items()}
                                measured = {
                                    k: np.concatenate((np.asarray(history[k]), values[k]))
                                    for k in history
                                }
                                review = branch.review()
                                if any(
                                    type(review[k]) is not int or review[k] != (300 - start) * 10
                                    for k in (
                                        "new_native_attempted_steps",
                                        "new_native_integrated_steps",
                                    )
                                ):
                                    raise ValueError(
                                        "every predicted remaining physical step required"
                                    )
                                proof = dict(
                                    index=i,
                                    branch_index=bi,
                                    action_name=option,
                                    start_frame=start,
                                    review=review,
                                    timing_seconds=dict(
                                        construction=branch_construction_seconds,
                                        control_steps=predicted_control_seconds,
                                    ),
                                    predicted_outcome=label_first_touch(measured).to_dict(),
                                    arrays=archive(
                                        p.root, f"case-{i}-branch-{bi}-future.npz", values
                                    ),
                                    persisted_before_donor_continues=True,
                                    recorded_future_used_for_prediction=False,
                                )
                                proof["report_hash"] = hash_json(proof)
                                write_once(p.root / f"case-{i}-branch-{bi}-prediction.json", proof)
                                predictions.append((values, proof))
                                branch = None
                            # Advisory only, all three actions execute regardless of ranking.
                            same_arrays(predictions[0][0], predictions[1][0])
                            eligible = list(range(len(actions))) if born_release else [0, 2, 3]
                            selected = max(
                                eligible,
                                key=lambda n: (
                                    predictions[n][1]["predicted_outcome"]["safe"],
                                    predictions[n][1]["predicted_outcome"]["training_reward"],
                                    -n,
                                ),
                            )
                            choice = dict(
                                index=i,
                                selected_branch=selected,
                                selected_action=predictions[selected][1]["action_name"],
                                selection_uses_only_measured_past_and_own_simulated_future=True,
                                advisory_only=True,
                                teacher_qualified=False,
                                hardware_authorized=False,
                            )
                            choice["report_hash"] = hash_json(choice)
                            write_once(p.root / f"case-{i}-prospective-choice.json", choice)
                        else:
                            if born_release:
                                actor.install_born_release(
                                    world_hash=p.world_hash,
                                    action_name=BORN_RELEASE_ACTIONS[action],
                                )
                            else:
                                actor.install_navigation_recovery(
                                    BornNavigationRecovery(start, action)
                                )
                        current = dict(index=i, action_name=action, phase="ACTUAL_AFTER_FOOT")
                if start is None or len(predictions) != len(prediction_actions):
                    raise ValueError(
                        "all declared own future predictions and actual completed foot required"
                    )
                trace = {k: np.asarray(v) for k, v in history.items()}
                if ai == 0:
                    same_arrays(base_arrays(trace), job["gold_trace"])
                for bi in indices(ai):
                    same_arrays(predictions[bi][0], {k: a[start:] for k, a in trace.items()})
                    if (
                        predictions[bi][1]["predicted_outcome"]
                        != label_first_touch(trace).to_dict()
                    ):
                        raise ValueError(
                            "prospective nonzero outcome differs from actual same lease"
                        )
                review = actor.review()
                if (
                    review["native_attempted_steps"] != 3000
                    or review["native_integrated_steps"] != 3000
                ):
                    raise ValueError("whole actual H300 body required")
                row = dict(
                    index=i,
                    arm_index=ai,
                    action_name=action,
                    source_case_hash=job["source_case_hash"],
                    course=job["course"],
                    arrays=archive(p.root, f"case-{i}-arm-{ai}-trace.npz", trace),
                    outcome=label_first_touch(trace).to_dict(),
                    main_review=review,
                    timing_seconds=dict(
                        construction=construction_seconds, control_steps=actual_control_seconds
                    ),
                    matching_prediction_indices=indices(ai),
                    new_optimizer_updates=0,
                    teacher_qualified=False,
                    training_authorized=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                )
                row[
                    "all_future46_arrays_match_prediction"
                    if born_release
                    else "all_future43_arrays_match_prediction"
                ] = True
                row["report_hash"] = hash_json(row)
                write_once(p.root / f"case-{i}-arm-{ai}-complete.json", row)
                actuals.append(row)
                actor, history = None, {}
                del trace
                gc.collect()
                print(
                    "BORN_RELEASE_ACTUAL_PARITY_COMPLETE"
                    if born_release
                    else "NAVIGATION_RECOVERY_ACTUAL_PARITY_COMPLETE",
                    i,
                    action,
                    flush=True,
                )
            rows.append(
                dict(index=i, actuals=actuals, predictions=[proof for _, proof in predictions])
            )
        factory._stable()
        if (
            main_steps != main_budget
            or prediction_steps != prediction_budget
            or len(rows) != contexts
            or any(digest(n) != h for n, h in p.pins.items())
        ):
            raise ValueError(f"whole immutable{total_budget}-step action bank required")
        complete = dict(
            schema=stem + "_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            new_main_native_steps=main_steps,
            new_prediction_native_steps=prediction_steps,
            combined_new_native_steps=main_steps + prediction_steps,
            new_optimizer_updates=0,
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        complete[
            "born_release_prediction_qualified_on_one_consumed_context"
            if born_release
            else "nonzero_navigation_prediction_qualified_on_consumed_four_contexts"
        ] = True
        complete["report_hash"] = hash_json(complete)
        write_once(p.root / "complete.json", complete)
    except BaseException as error:
        if history:
            archive(
                p.root,
                "failure-main-partial-trace.npz",
                {k: np.asarray(v) for k, v in history.items()},
            )
        write_once(
            p.root / "failure.json",
            dict(
                current=current,
                completed_contexts=len(rows),
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
