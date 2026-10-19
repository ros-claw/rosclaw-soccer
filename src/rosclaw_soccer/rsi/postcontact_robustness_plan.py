"""Forty public initial states, four matched control arms, not new G0 data.

No outcomes enter this declaration. New owned-branch teacher execution needs
its own admission/prospective verification; this plan grants no motion or
learning authority and is not the sealed Fresh exam.
"""

import hashlib
from typing import Any

from rosclaw_soccer.rsi.balanced_first_touch_curriculum import balanced_public_first_touch_jobs
from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_course_preflight import (
    FirstTouchCourse,
    preflight_receiving_courses,
)

COURSE_GENERATOR_SEEDS = tuple(range(202610200, 202610210))
ARMS = (
    ("heading_off_original", False, False),
    ("heading_off_postcontact", False, True),
    ("heading_on_original", True, False),
    ("heading_on_postcontact", True, True),
)


def declare_robustness40(*, mean_model_hash: str) -> dict[str, Any]:
    if (
        type(mean_model_hash) is not str
        or len(mean_model_hash) != 71
        or not mean_model_hash.startswith("sha256:")
        or any(c not in "0123456789abcdef" for c in mean_model_hash[7:])
    ):
        raise ValueError("exact frozen full mean model hash required")
    jobs = balanced_public_first_touch_jobs(COURSE_GENERATOR_SEEDS, repeats=1)
    seeds: list[int] = []
    strata = [0, 0, 0, 0]
    records: list[dict[str, Any]] = []
    launches: list[FirstTouchCourse] = []
    for job in jobs:
        bucket = validate_public_course(job.course)
        payload = f"soccer.rsi.postcontact_four_arm40.v1:{job.seed}:{job.lane}"
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
        seeds.append(seed)
        strata[bucket] += 1
        launches.append(FirstTouchCourse("cpu.first_touch.0", seed, job.course))
        records.append(
            dict(
                index=len(records),
                original_generator_seed=job.seed,
                lane=job.lane,
                source_cluster=job.group // 4,
                stratum=bucket,
                course=list(job.course),
                sampling_seed=seed,
                arms=[
                    dict(name=name, heading_enabled=heading, postcontact_exit_enabled=teacher)
                    for name, heading, teacher in ARMS
                ],
                same_sampling_seed_all_four_arms=True,
                all_success_failure_censored_unsafe_outcomes_required=True,
            )
        )
    if len(records) != 40 or strata != [10] * 4 or len(set(seeds)) != 40:
        raise ValueError("whole balanced40 unique-state/seed declaration required")
    preflight = preflight_receiving_courses(tuple(launches))
    report = dict(
        schema="soccer.rsi.postcontact_owned_robustness40_plan.v1",
        partition="DECLARED_PUBLIC_MECHANISM_EXPANSION_NOT_FRESH",
        mean_model_hash=mean_model_hash,
        generator_seeds=list(COURSE_GENERATOR_SEEDS),
        receiving_course_preflight_hash=preflight,
        records=records,
        strata_counts=strata,
        unique_initial_states=40,
        generator_seed_clusters=10,
        headings_per_initial_state=2,
        paired_control_contexts=80,
        new_native_rollouts=160,
        new_native_step_budget=480000,
        maximum_forecast_step_budget=44800,
        combined_step_budget=524800,
        official_horizon_frames=300,
        post_contact_frames=60,
        maximum_postcontact_action_frames=3,
        baseline_source_executions_reused=0,
        four_control_arms_not_four_independent_sampling_draws=True,
        old_G0_bank_unchanged=True,
        all_courses_preflighted=True,
        implementation_profile="explicit new owned-feedback branch teacher, not default change",
        selected_window_parity_required_before_collection=True,
        independent_prospective_alignment_and_full_task_verification_required=True,
        heading_action_is_fixed_by_arm_not_future_outcome=True,
        privilege_ceiling=(
            "SIM current-truth teacher; student observation must be separately approved"
        ),
        private_fresh_accessed=False,
        global_ancestry_separation_proven=False,
        new_optimizer_updates=0,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    return report
