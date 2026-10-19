"""Full-H300 offline action labels; not a deployed future-aware actor/teacher.

All18 public consumed contexts, all13 named current-contact actions plus
repeated zero. Fourteen already qualified bodies from context1 are reused;
238 new bodies execute their own future Foundation/IK/RNN/planner updates.
The bank is for learning reachability, not a claim of238 new situations.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.impact_torque_screen import CAPS
from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_postcontact_full_bank_inputs import prepare
from scripts.rsi_postcontact_release_trace import check_release_trace
from scripts.rsi_stage_release_contact_replay import digest


def summarize_context(job, rows):
    if (
        len(rows) != 14
        or [r["action_index"] for r in rows] != list(range(14))
        or any(r["context_index"] != job["index"] for r in rows)
        or [r["action_name"] for r in rows] != [*ACTION_NAMES, "baseline"]
    ):
        raise ValueError("all fourteen ordered actions including failures/repeated zero required")
    if any(type(r["outcome"][k]) is not bool for r in rows for k in ("task_pass", "safe")) or any(
        r["outcome"]["task_pass"] and not r["outcome"]["safe"] for r in rows
    ):
        raise ValueError("primitive safe passing outcome booleans required")
    if (
        rows[0]["outcome"] != job["baseline_outcome"]
        or rows[-1]["outcome"] != job["baseline_outcome"]
    ):
        raise ValueError("both whole zero controls must reproduce original baseline")
    eligible = [
        i for i, r in enumerate(rows[:-1]) if r["outcome"]["task_pass"] and r["outcome"]["safe"]
    ]
    preferred = 0
    if not job["baseline_outcome"]["task_pass"] and eligible:
        preferred = min(
            eligible,
            key=lambda i: (
                abs(rows[i]["outcome"]["lateral_60_m"]) / rows[i]["outcome"]["forward_60_m"],
                rows[i]["normalized_action_norm_squared"],
                i,
            ),
        )
    r = dict(
        context_index=job["index"],
        partition=job["partition"],
        source_index=job["source_index"],
        baseline_outcome=job["baseline_outcome"],
        rows=rows,
        observed_safe_passing_action_indices=eligible,
        offline_preferred_action_index=preferred,
        offline_preferred_action_name=rows[preferred]["action_name"],
        offline_selected_outcome=rows[preferred]["outcome"],
        selection_uses_full_simulated_future_outcomes=True,
        runtime_future_outcome_input_authorized=False,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    r["report_hash"] = hash_json(r)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if (
        Path(__file__).resolve() != p.source / "scripts/rsi_postcontact_full_bank.py"
        or not time.time() < p.cfg["deadline_unix"] <= time.time() + 86400
    ):
        raise ValueError("exact frozen bounded whole-H300 bank entry required")
    if args.preflight:
        print("FULL_H300_BANK252_WITH14_REUSED_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available before sole new full mean constructor required")
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.postcontact_full_h300_bank_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        jobs=[{k: v for k, v in j.items() if k != "baseline_trace"} for j in p.jobs],
        reused_authority_hash=p.reused_authority_hash,
        comparison_contexts=18,
        comparison_rollouts=252,
        reused_execution_count=14,
        new_rollout_budget=238,
        new_native_step_budget=714000,
        offline_future_simulation_for_labels_only=True,
        source_future_trace_passed_to_actor=False,
        all_failed_unsafe_and_censored_records_retained=True,
        new_optimizer_updates=0,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    contexts, rows = [], []
    native = new_runs = reused = 0
    current = replay = None
    history = {}
    try:
        import mujoco
        import torch

        from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
        from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
        from rosclaw_soccer.rsi.causal_impact_replay import CausalImpactReplay
        from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
        from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        torch.set_num_threads(1)
        print("ALLOCATING_FULL_ORIGINAL_MEAN_FOR_OFFLINE_H300_ACTION_BANK", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same complete source mean required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        for job in p.jobs:
            case = p.cases[job["source_index"]]
            w = case["receipt"]["witnesses"]
            commitment = load_json_artifact(Path(w["commitment"]["path"]))
            report = load_json_artifact(Path(w["report"]["path"]))["stripped_document"]
            seed = report["executed_motor_policy"]["step_motor_proof"]["model"]["seed"]
            rows = []
            for ai, name in enumerate(p.cfg["action_names"]):
                current = dict(context_index=job["index"], action_index=ai, action_name=name)
                if time.time() >= p.cfg["deadline_unix"]:
                    raise RuntimeError("fixed full-bank wall deadline exhausted")
                write_once(p.root / f"context-{job['index']}-action-{ai}-started.json", current)
                replay = None
                history = {}
                is_reused = job["index"] == p.cfg["reused_authority_context_index"]
                if is_reused:
                    old = p.reused_authority[ai]
                    arrays = old["arrays"]
                    with np.load(arrays["path"], allow_pickle=False) as a:
                        trace = dict(a)
                    review = old["causal_review"]
                    policy_hash = old["original_policy_hash"]
                    reused_hash = old["report_hash"]
                else:
                    start = time.perf_counter()
                    policy, decoder = factory.generate_and_bind_seed(seed=seed)
                    replay = CausalImpactReplay(
                        model,
                        world_hash=p.world_hash,
                        kp=np.asarray(case["physics"]["canonical_pd_kp"], dtype=np.float64),
                        kd=np.asarray(case["physics"]["canonical_pd_kd"], dtype=np.float64),
                        model_root=p.model_root,
                        foundation=foundation,
                        expected_foundation_hash=commitment["foundation_hash"],
                        policy=policy,
                        decoder=decoder,
                        expected_policy_hash=commitment["motor_policy_hash"],
                        course=tuple(job["course"]),
                        profile=BoundedFirstTouchControl(
                            **p.declaration["candidate"]["parameters"]
                        ),
                        heading=FootHeadingContactProbe(case["job"]["heading_enabled"]),
                        impact_enabled=False,
                        early_impact_enabled=False,
                        release_action_name=name,
                    )
                    print(
                        "FULL_H300_BANK_BODY_READY",
                        current,
                        "bind_and_navigation_s",
                        time.perf_counter() - start,
                        flush=True,
                    )
                    for f in range(300):
                        if (
                            time.time() >= p.cfg["deadline_unix"]
                            or native + replay.attempted_steps + 10 > 714000
                        ):
                            raise RuntimeError(
                                "fixed full-H300 bank deadline/native budget exhausted"
                            )
                        result = replay.step(f)
                        for k, v in result.arrays.items():
                            history.setdefault(k, []).append(v.copy())
                    trace = {k: np.asarray(v) for k, v in history.items()}
                    file = p.root / f"context-{job['index']}-action-{ai}-trace.npz"
                    with file.open("xb") as stream:
                        np.savez_compressed(stream, **trace)
                    arrays = dict(path=str(file), sha256=digest(file))
                    review = replay.review()
                    policy_hash = policy["policy_hash"]
                    reused_hash = None
                    del policy, decoder
                if (
                    review["native_integrated_steps"] != 3000
                    or review["forecast_integrated_steps"] != 0
                    or policy_hash != commitment["motor_policy_hash"]
                ):
                    raise ValueError("same whole original zero-forecast model execution required")
                control = check_release_trace(trace, review, action_name=name)
                require_prefix(
                    trace,
                    job["baseline_trace"],
                    control["first_physically_applied_correction_frame"],
                )
                row = dict(
                    current,
                    source_receipt_hash=case["receipt"]["receipt_hash"],
                    course=job["course"],
                    original_policy_hash=policy_hash,
                    arrays=arrays,
                    causal_review=review,
                    control_review=control,
                    outcome=label_first_touch(trace).to_dict(),
                    execution_reused=is_reused,
                    reused_case_hash=reused_hash,
                    normalized_action_norm_squared=float(
                        np.max(
                            np.sum(
                                (trace["postcontact_release_requested_offset_nm"][:, 0] / CAPS)
                                ** 2,
                                axis=1,
                            )
                        )
                    ),
                    new_optimizer_updates=0,
                    teacher_qualified=False,
                    training_authorized=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                )
                row["report_hash"] = hash_json(row)
                write_once(p.root / f"context-{job['index']}-action-{ai}-complete.json", row)
                rows.append(row)
                if is_reused:
                    reused += 1
                else:
                    native += review["native_integrated_steps"]
                    new_runs += 1
                print(
                    "FULL_H300_BANK_ACTION_COMPLETE",
                    current,
                    "reused",
                    is_reused,
                    row["outcome"],
                    flush=True,
                )
                replay = None
                history = {}
                del trace
                gc.collect()
            context = summarize_context(job, rows)
            write_once(p.root / f"context-{job['index']}-complete.json", context)
            contexts.append(context)
        factory._stable()
        if (
            native != 714000
            or new_runs != 238
            or reused != 14
            or len(contexts) != 18
            or any(digest(n) != h for n, h in p.pins.items())
        ):
            raise ValueError("whole immutable252-comparison bank and238-new-body budget required")
        result = dict(
            schema="soccer.rsi.postcontact_full_h300_bank_complete.v1",
            declaration_hash=declaration["report_hash"],
            contexts=contexts,
            comparison_contexts=18,
            comparison_rollouts=252,
            new_rollouts=new_runs,
            reused_execution_count=reused,
            new_native_steps=native,
            forecast_steps=0,
            executed_control_foundation_pairs=71400,
            new_optimizer_updates=0,
            original_baseline_passes=sum(c["baseline_outcome"]["task_pass"] for c in contexts),
            best_observed_safe_pass_contexts=sum(
                c["offline_selected_outcome"]["task_pass"] for c in contexts
            ),
            selection_uses_full_simulated_future_outcomes=True,
            runtime_future_outcome_input_authorized=False,
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(p.root / "complete.json", result)
    except BaseException as error:
        if history:
            with (p.root / "failure-partial-trace.npz").open("xb") as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in history.items()})
        if replay is not None:
            with (p.root / "failure-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, qpos=replay.data.qpos, qvel=replay.data.qvel, ctrl=replay.data.ctrl
                )
        write_once(
            p.root / "failure.json",
            dict(
                current=current,
                completed_contexts=len(contexts),
                current_context_completed_actions=len(rows),
                completed_new_native_steps=native,
                completed_new_rollouts=new_runs,
                reused_execution_count=reused,
                current_attempted_steps=0 if replay is None else replay.attempted_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
