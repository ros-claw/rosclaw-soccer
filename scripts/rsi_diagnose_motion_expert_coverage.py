"""Counterfactual frozen-expert coverage on outcome-blind stratified24.

This is diagnostic reuse of an already consumed near holdout, NOT fresh testing,
online expert selection, a trainable policy or an oracle that can be deployed.
"""

import argparse
import json
from pathlib import Path


def diagnostic_indices():
    return tuple(base + offset for base in range(0, 192, 32) for offset in (0, 8, 16, 24))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "review", "launch", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-inputs-hash",
        "expected-review-hash",
        "expected-launch-hash",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--owners", nargs="+", type=int, required=True)
    a = p.parse_args()
    if (
        not a.owners
        or len(a.owners) != len(set(a.owners))
        or any(i not in range(1, 5) for i in a.owners)
    ):
        raise ValueError("distinct frozen nonbaseline experts1..4 required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_portfolio_common import admit, load_frozen_experts
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, _, pins, _ = admit(a)
    if (
        digest(a.review / "complete.json") != a.expected_review_hash
        or digest(a.launch) != a.expected_launch_hash
    ):
        raise ValueError("authenticated consumed holdout and launch required")
    review = json.loads((a.review / "complete.json").read_text())
    launch = json.loads(a.launch.read_text())
    if (
        review["status"] != "NEAR_HOLDOUT192_RECEIPTS_AUTHENTICATED"
        or review["launch_hash"] != a.expected_launch_hash
        or review["summary"]["net_gain"] != 0
    ):
        raise ValueError("this declared diagnosis follows closed no-gain experiment only")
    pins.update(review["input_file_hashes"])
    pins[str(a.review / "complete.json")] = a.expected_review_hash
    pins[str(a.launch)] = a.expected_launch_hash
    check_pins(pins)
    indices = diagnostic_indices()
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.counterfactual_motion_coverage.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            indices=indices,
            owners=a.owners,
            selection_rule="every stratum offsets0,8,16,24; independent of outcomes",
            reused_consumed_holdout=True,
            fresh_test=False,
            oracle_deployable=False,
            optimizer_steps=0,
            planned_new_native_steps=24 * len(a.owners) * 6000,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    result = run(a, spec, pins, launch, indices, load_frozen_experts(spec))
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="COUNTERFACTUAL_FROZEN_EXPERT_COVERAGE_VERIFIED",
            source_head=a.expected_head,
            results=result,
            new_native_steps=24 * len(a.owners) * 6000,
            optimizer_steps=0,
            reused_consumed_holdout=True,
            fresh_test=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


def run(a, spec, pins, launch, indices, models):
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20, score_full_entry
    from rosclaw_soccer.rsi.recorded_entry_review import replay_prefix
    from rosclaw_soccer.training.motion_portfolio_holdout import motion_portfolio_holdout_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_ppo_rollout import SamplingPolicy
    from scripts.rsi_motion_student_rollout import replay_motion_student, rollout_motion_student
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array
    from scripts.rsi_native_context_recovery_common import digest

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    world = mujoco.MjModel.from_binary_path(spec["world"])
    backend = G1SonicRunupController(
        Path(spec["model_root"]),
        G1SonicRunupConfig(model_variant="low_latency"),
        inference_threads=1,
    )
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(priors) != 1:
        raise ValueError("one original fixed prefix prior required")
    d = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(d["ball_offset_xy_m"]),
        d["historical_frame"],
        d["historical_forward_speed_mps"],
        d["historical_artifact_hash"],
    )
    if prior.contract_hash != d["contract_hash"]:
        raise ValueError("fixed prefix prior differs")
    courses, results = motion_portfolio_holdout_courses(), []
    for index in indices:
        source_jobs = [j for j in launch["jobs"] if index in j["indices"]]
        if len(source_jobs) != 1:
            raise ValueError("unique authenticated source course required")
        original = Path(source_jobs[0]["output"]) / f"course{index:03d}"
        old = json.loads((original / "complete.json").read_text())
        with np.load(original / "baseline/native.npz", allow_pickle=False) as z:
            recorded = {k: z[k][:20].copy() for k in PHYSICAL_FIELDS}

        def body(course=courses[index].course):
            return G1TargetEpisode(
                world,
                world_hash=digest(Path(spec["world"])),
                kp=backend.kp.copy(),
                kd=backend.kd.copy(),
                course=course,
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )

        first = body()
        try:
            nav = G1SonicNavigation(
                Path(spec["model_root"]),
                courses[index].agent_id,
                SonicNavigationConfig(
                    maximum_frames=300,
                    model_variant="low_latency",
                    inference_threads=1,
                    record_default_planner_events=True,
                ),
            )
            prefix = nominal_prefix20(nav, first, prior)
            del nav
            for key in PHYSICAL_FIELDS:
                exact_array(prefix[key], recorded[key])
        except Exception as error:
            write_once(
                a.output / f"failure-prefix{index:03d}.json",
                dict(
                    error=repr(error),
                    native_steps=first.integrated_steps,
                    attempted_steps=first.attempted_steps,
                ),
            )
            raise
        diagnostics = diagnose_entry_prefix(world, prefix)
        for offset, owner in enumerate(a.owners):
            folder = a.output / f"course{index:03d}-owner{owner}"
            folder.mkdir()
            actual, independent, progress = first if offset == 0 else body(), None, {}
            try:
                if offset != 0:
                    replay_prefix(actual, prefix, prior)
                policy = SamplingPolicy(models[owner], seed=92029 + index, stochastic=False)
                trace = rollout_motion_student(
                    policy, FoundationMotorCodec(backend), actual, prefix, progress=progress
                )
                np.savez_compressed(folder / "native.npz", **trace)
                sampled = policy.recording()
                np.savez_compressed(folder / "sampling.npz", **sampled)
                independent = body()
                again = SamplingPolicy(models[owner], seed=92029 + index, stochastic=False)
                proof = replay_motion_student(
                    again, FoundationMotorCodec(backend), independent, trace
                )
                for k, v in again.recording().items():
                    exact_array(v, sampled[k])
                if actual.integrated_steps != 3000 or independent.integrated_steps != 3000:
                    raise ValueError("complete actual H300 and independent replay required")
            except Exception as error:
                write_once(
                    folder / "failure.json",
                    dict(
                        error=repr(error),
                        original_native_steps=actual.integrated_steps,
                        original_attempted_steps=actual.attempted_steps,
                        independent_native_steps=independent.integrated_steps if independent else 0,
                        independent_attempted_steps=independent.attempted_steps
                        if independent
                        else 0,
                    ),
                )
                raise
            score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
            row = dict(
                course_index=index,
                stratum=courses[index].stratum,
                owner=owner,
                baseline_success=old["variants"][0]["causal_qualified"],
                deployed_portfolio_success=old["variants"][1]["causal_qualified"],
                deployed_owner=old["variants"][1]["portfolio_audit"]["selected_owner"],
                causal_qualified=bool(
                    score["qualified"]
                    and diagnostics["prefix_physical_safe"]
                    and diagnostics["post_boundary_control_can_cause_first_touch"]
                ),
                score=score,
                prefix_diagnostics=diagnostics,
                replay=proof,
                new_native_steps=6000,
                native_hash=digest(folder / "native.npz"),
                sampling_hash=digest(folder / "sampling.npz"),
                foundation_prefix_neural_planning_shared_per_course=True,
                foundation_prefix_all_physical_fields_match_authenticated_baseline=True,
                recorded_prefix_physically_reexecuted=True,
            )
            write_once(folder / "complete.json", row)
            write_once(a.output / f"progress{index:03d}-owner{owner}.json", row)
            results.append(row)
            print("EXPERT_COVERAGE_VERIFIED", index, owner, row["causal_qualified"], flush=True)
    return results


if __name__ == "__main__":
    main()
