"""Declare and execute a separate paired192 near holdout after candidate freeze."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("declare", "run"), required=True)
    for name in ("inputs", "core-source", "router-training", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-inputs-hash",
        "router-unit",
        "expected-router-receipt-hash",
        "expected-router-source",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--expected-plan-hash")
    p.add_argument("--indices", nargs="+", type=int)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, _, pins, _ = admit(a)
    wait_for_terminal_collection(a.router_unit)
    if digest(a.router_training / "complete.json") != a.expected_router_receipt_hash:
        raise ValueError("frozen router receipt differs")
    tc = json.loads((a.router_training / "complete.json").read_text())
    td = json.loads((a.router_training / "declaration.json").read_text())
    if (
        tc["source_head"] != a.expected_router_source
        or td["source_head"] != a.expected_router_source
        or tc["inputs_hash"] != a.expected_inputs_hash
        or tc["optimizer_steps"] != 2000
        or tc["status"] != "MOTION_ROUTER_TRAINED_REQUIRES_EXAM"
        or digest(a.router_training / "router.pt") != tc["checkpoint_hash"]
    ):
        raise ValueError("fixed original router and memory roster required")
    sources = [
        Path(k).parent.parent
        for k in td["input_file_hashes"]
        if k.endswith("/scripts/rsi_train_motion_router.py")
    ]
    if (
        len(sources) != 1
        or subprocess.check_output(
            ["git", "-C", str(sources[0]), "rev-parse", "HEAD"], text=True
        ).strip()
        != a.expected_router_source
    ):
        raise ValueError("one immutable original candidate source required")
    for folder in ("src", "scripts"):
        for old in (sources[0] / folder).rglob("*.py"):
            rel = old.relative_to(sources[0])
            if str(rel) != "src/rosclaw_soccer/training/receiving_course_preflight.py" and digest(
                old
            ) != digest(Path.cwd() / rel):
                raise ValueError("holdout changed frozen candidate/control code: " + str(rel))
    pins.update(td["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.router_training.iterdir() if f.is_file()})
    from rosclaw_soccer.training.motion_portfolio_holdout import motion_portfolio_holdout_courses
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = motion_portfolio_holdout_courses()
    bank_hash = preflight_receiving_courses(courses)  # ENTIRE192 before model/worker allocation
    check_pins(pins)
    if a.mode == "declare":
        if a.plan is not None or a.indices is not None:
            raise ValueError("declaration always contains full192")
        a.output.mkdir(parents=True, exist_ok=False)
        write_once(
            a.output / "declaration.json",
            dict(
                schema="soccer.rsi.near_motion_holdout_plan.v1",
                source_head=a.expected_head,
                input_file_hashes=pins,
                inputs_hash=a.expected_inputs_hash,
                frozen_router_hash=tc["checkpoint_hash"],
                preflight_hash=bank_hash,
                candidate_frozen_before_course_generation=True,
                original_rsi_private_fresh=False,
                optimizer_steps=0,
                new_native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        write_once(
            a.output / "courses.json",
            dict(courses=[asdict(c) for c in courses], preflight_hash=bank_hash),
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NEAR_PORTFOLIO_HOLDOUT_DECLARED",
                source_head=a.expected_head,
                courses=192,
                preflight_hash=bank_hash,
                courses_hash=digest(a.output / "courses.json"),
                frozen_router_hash=tc["checkpoint_hash"],
                new_native_steps=0,
                optimizer_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        return
    if (
        a.plan is None
        or not a.expected_plan_hash
        or digest(a.plan / "complete.json") != a.expected_plan_hash
        or not a.indices
        or len(a.indices) != len(set(a.indices))
        or any(i not in range(192) for i in a.indices)
    ):
        raise ValueError("closed whole192 plan and disjoint declared subset required")
    plan = json.loads((a.plan / "complete.json").read_text())
    if (
        plan["source_head"] != a.expected_head
        or plan["preflight_hash"] != bank_hash
        or plan["frozen_router_hash"] != tc["checkpoint_hash"]
        or digest(a.plan / "courses.json") != plan["courses_hash"]
    ):
        raise ValueError("sealed near holdout plan differs")
    pins.update({str(f): digest(f) for f in a.plan.iterdir() if f.is_file()})
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.near_motion_holdout_paired.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            inputs_hash=a.expected_inputs_hash,
            plan_hash=a.expected_plan_hash,
            preflight_hash=bank_hash,
            indices=a.indices,
            paired_variants=["baseline", "portfolio"],
            foundation_prefix_neural_math_shared_between_variants=True,
            foundation_prefix_physically_reexecuted_for_every_body=True,
            independent_replay_recomputes_student_not_foundation_prefix_networks=True,
            no_parameter_updates=True,
            original_rsi_private_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    result = run(a, spec, pins, courses, tc)
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="NEAR_HOLDOUT_PAIRED_NATIVE_VERIFIED",
            source_head=a.expected_head,
            results=result,
            new_native_steps=sum(r["new_native_steps"] for r in result),
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


def run(a, spec, pins, courses, tc):
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
    from rosclaw_soccer.rsi.motion_capability_router import create_motion_capability_router
    from rosclaw_soccer.rsi.recorded_entry_review import replay_prefix
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_frozen_motion_portfolio import FrozenMotionPortfolio
    from scripts.rsi_motion_portfolio_common import load_frozen_experts
    from scripts.rsi_motion_ppo_rollout import SamplingPolicy
    from scripts.rsi_motion_student_rollout import replay_motion_student, rollout_motion_student
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array
    from scripts.rsi_native_context_recovery_common import digest

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = torch.load(a.router_training / "router.pt", map_location="cpu", weights_only=True)
    if (
        payload["source_head"] != a.expected_router_source
        or payload["inputs_hash"] != a.expected_inputs_hash
        or payload["expert_checkpoint_hashes"] != [e["checkpoint_hash"] for e in spec["experts"]]
    ):
        raise ValueError("fixed holdout router checkpoint metadata differs")
    router = create_motion_capability_router(experts=5)
    router.load_state_dict(payload["state_dict"], strict=True)
    router.eval().requires_grad_(False)
    models = load_frozen_experts(spec)
    portfolio = FrozenMotionPortfolio(router, models)
    world = mujoco.MjModel.from_binary_path(spec["world"])
    model_root = Path(spec["model_root"])
    backend = G1SonicRunupController(
        model_root, G1SonicRunupConfig(model_variant="low_latency"), inference_threads=1
    )
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(priors) != 1:
        raise ValueError("one original fixed-prefix prior required")
    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("original prefix affordance differs")
    results = []

    class CheckedPrefixBody:
        def __init__(self, actual, prefix):
            self.actual, self.prefix = actual, prefix

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def step(self, frame, target):
            row = self.actual.step(frame, target)
            for key in PHYSICAL_FIELDS:
                exact_array(row[key], self.prefix[key][frame : frame + 1])
            return row

    for index in a.indices:
        course = courses[index]
        folder = a.output / f"course{index:03d}"
        folder.mkdir()

        def body(course=course):
            return G1TargetEpisode(
                world,
                world_hash=digest(Path(spec["world"])),
                kp=backend.kp.copy(),
                kd=backend.kd.copy(),
                course=course.course,
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )

        first = body()
        try:
            nav = G1SonicNavigation(
                model_root,
                course.agent_id,
                SonicNavigationConfig(
                    maximum_frames=300,
                    model_variant="low_latency",
                    inference_threads=1,
                    record_default_planner_events=True,
                ),
            )
            prefix = nominal_prefix20(nav, first, prior)
            del nav
        except Exception as error:
            write_once(
                folder / "failure-prefix.json",
                dict(
                    error=repr(error),
                    native_steps=first.integrated_steps,
                    attempted_steps=first.attempted_steps,
                ),
            )
            raise
        variants = []
        for variant, model in (("baseline", models[0]), ("portfolio", portfolio)):
            part = folder / variant
            part.mkdir()
            actual = first if variant == "baseline" else body()
            sampling = SamplingPolicy(model, seed=92029 + index, stochastic=False)
            progress = {}
            independent = None
            try:
                if variant == "portfolio":
                    replay_prefix(CheckedPrefixBody(actual, prefix), prefix, prior)
                trace = rollout_motion_student(
                    sampling, FoundationMotorCodec(backend), actual, prefix, progress=progress
                )
                np.savez_compressed(part / "native.npz", **trace)
                sampled = sampling.recording()
                np.savez_compressed(part / "sampling.npz", **sampled)
                again = SamplingPolicy(model, seed=92029 + index, stochastic=False)
                independent = body()
                proof = replay_motion_student(
                    again, FoundationMotorCodec(backend), independent, trace
                )
                for k, v in again.recording().items():
                    exact_array(v, sampled[k])
            except Exception as error:
                rows = progress.get("rows", [])
                if rows:
                    np.savez_compressed(
                        part / "partial.npz",
                        **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
                    )
                write_once(
                    part / "failure.json",
                    dict(
                        error=repr(error),
                        original_native_steps=actual.integrated_steps,
                        original_attempted_steps=actual.attempted_steps,
                        independent_native_steps=independent.integrated_steps
                        if independent is not None
                        else 0,
                        independent_attempted_steps=independent.attempted_steps
                        if independent is not None
                        else 0,
                    ),
                )
                raise
            if actual.integrated_steps != 3000 or independent.integrated_steps != 3000:
                raise ValueError("full H300 every paired native body required")
            diagnostics = diagnose_entry_prefix(world, {k: v[:20] for k, v in trace.items()})
            score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
            causal = bool(
                score["qualified"]
                and diagnostics["prefix_physical_safe"]
                and diagnostics["post_boundary_control_can_cause_first_touch"]
            )
            row = dict(
                variant=variant,
                score=score,
                prefix_diagnostics=diagnostics,
                causal_qualified=causal,
                native_steps=6000,
                replay=proof,
                native_hash=digest(part / "native.npz"),
                sampling_hash=digest(part / "sampling.npz"),
            )
            if variant == "portfolio":
                left, right = portfolio.audits[-2:]
                if {k: v for k, v in left.items() if k != "stream_index"} != {
                    k: v for k, v in right.items() if k != "stream_index"
                }:
                    raise ValueError(
                        "paired holdout owner/memory inference not independently exact"
                    )
                row["portfolio_audit"] = right
            write_once(part / "complete.json", row)
            variants.append(row)
        outcome = dict(
            course=asdict(course),
            variants=variants,
            new_native_steps=12000,
            rescue=not variants[0]["causal_qualified"] and variants[1]["causal_qualified"],
            regression=variants[0]["causal_qualified"] and not variants[1]["causal_qualified"],
        )
        write_once(folder / "complete.json", outcome)
        write_once(a.output / f"progress{index:03d}.json", outcome)
        results.append(outcome)
        print(
            "NEAR_HOLDOUT_PAIRED_VERIFIED",
            index,
            course.stratum,
            [r["causal_qualified"] for r in variants],
            flush=True,
        )
    return results


if __name__ == "__main__":
    main()
