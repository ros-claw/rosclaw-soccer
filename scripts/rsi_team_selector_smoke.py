"""Paired eight-body SIM smoke of all three learned receiving selectors.

Two consumed parent-success courses only; no fresh gate or policy promotion.
Original parent and all seeds are executed in a shared physical world, with
the nine physical trace arrays persisted and hash checked after each run.
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import rsi_r1_velocity_gated_fresh_v238 as prior
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_measured_skill_router_v207 import clean as is_clean
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_smooth_residual_development_v261 import _run_smooth
from rsi_r1_velocity_gated_fresh_v238 import ANCHOR

from rosclaw_soccer.rsi.team_coverage_actor import TeamCoverageActor
from rosclaw_soccer.rsi.team_coverage_motor import TeamCoverageMotor
from rosclaw_soccer.rsi.team_receiving_selector import TeamReceivingSelector
from rosclaw_soccer.rsi.team_receiving_selector_motor import TeamReceivingSelectorMotor
from rosclaw_soccer.rsi.team_trace_capture import extract_team_trace
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean, source_pins

TRAINING_HASH = "sha256:341d75e84e3c8575371388bdca5bc65d230870a037633c2ee4bf2adf555070db"
PARENT_HASH = "sha256:52f70e48bb39f050350ee103c72bb3000aea47a3f594aa96f35434d9b008c282"


def run_job(job):
    arm, course, task, offset, threshold, model, trace_path = job
    prior.THRESHOLD = -0.48
    prior.RADIUS = 0.019
    original_simulate = prior.simulate_r0_receiving_course
    holder = []

    def capture(*args, **kwargs):
        result, trace = original_simulate(*args, **kwargs)
        arrays = {key: np.asarray(trace[key]) for key in PHYSICAL_KEYS}
        arrays.update(extract_team_trace(trace, require_tactical=True))
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("non-object physical trace arrays required")
        np.savez_compressed(trace_path, **arrays)
        return result, trace

    prior.simulate_r0_receiving_course = capture
    try:
        if arm == "parent":
            row = _run_smooth(((0.7645, 0.7660), offset, threshold, task))
        else:
            parent_weights = task[-2]
            refined = replace(
                parent_weights,
                output_bias=tuple(
                    float(x) for x in np.asarray(parent_weights.output_bias) + np.asarray(offset)
                ),
            )
            if arm.startswith("probe-"):
                selected = int(arm.removeprefix("probe-"))
                model = dict(model)
                model.update(
                    center=[0.0] * 10,
                    scale=[1.0] * 10,
                    weight1=np.zeros((32, 10)).tolist(),
                    bias1=[0.0] * 32,
                    weight2=np.zeros((16, 32)).tolist(),
                    bias2=[1.0 if i == selected else 0.0 for i in range(16)],
                )
                threshold = 0.775
                prior.THRESHOLD = -0.55
                prior.RADIUS = 0.022
            learned_coverage = arm.startswith("learned-")
            selector = (TeamCoverageActor if learned_coverage else TeamReceivingSelector).from_dict(
                model
            )

            def factory(*args, **kwargs):
                actor_type = TeamCoverageMotor if learned_coverage else TeamReceivingSelectorMotor
                actor = actor_type(
                    *args,
                    **kwargs,
                    **(
                        {"coverage_actor": selector} if learned_coverage else {"selector": selector}
                    ),
                    maximum_relative_lateral_feature=threshold,
                    protection_radius=0.002,
                    refined_neural_weights=refined,
                    gain_lateral_start=0.7645,
                    gain_lateral_full=0.7660,
                )
                holder.append(actor)
                return actor

            prior.ReceivingVelocityGatedPrecontactMotor = factory
            row = prior._run_velocity(task)
            if len(holder) != 1:
                raise ValueError("exactly one controlled receiving motor required")
            row["selector_action"] = (
                holder[0].coverage_action if learned_coverage else holder[0].selected_action
            )
            row["selector_out_of_support"] = (
                holder[0].coverage_fallback
                if learned_coverage
                else holder[0].selector_out_of_support
            )
        with np.load(trace_path, allow_pickle=False) as f:
            material = {key: f[key].tolist() for key in PHYSICAL_KEYS}
        if hash_json(material) != row["physical_trace_hash"]:
            raise ValueError("raw trace hash differs from actual rollout")
        row["trace_file"] = trace_path.name
        row["trace_file_hash"] = digest(trace_path)
        row["arm"] = arm
        row["course_seed"] = course.seed
        return row
    finally:
        prior.simulate_r0_receiving_course = original_simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "asset-root",
        "policy",
        "parent-report",
        "right-report",
        "refine-report",
        "lateral-report",
        "v205-dir",
        "map-report",
        "v214-report",
        "v215-dir",
        "v216-report",
        "v218-dir",
        "v220-report",
        "v224-report",
        "v229-dir",
        "v233-dir",
        "v236-dir",
        "v237-dir",
        "v254-dir",
        "v257-dir",
        "v261-dir",
        "training-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--full-retention", action="store_true")
    parser.add_argument("--smoke-proof", type=Path)
    parser.add_argument("--repeat-proof", type=Path)
    parser.add_argument("--coverage-probe", action="store_true")
    parser.add_argument("--coverage-learned-dir", type=Path)
    parser.add_argument("--coverage-audit", type=Path)
    parser.add_argument("--retention-proof", type=Path)
    parser.add_argument("--retention-audit", type=Path)
    args = parser.parse_args()
    clean(args.expected_head)
    learned_coverage = args.coverage_learned_dir is not None
    coverage_mode = args.coverage_probe or learned_coverage
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence directory required")
    if coverage_mode:
        if (
            args.coverage_probe
            and learned_coverage
            or args.full_retention
            or args.smoke_proof is not None
            or args.repeat_proof is not None
        ):
            raise ValueError("coverage probe is a separate development experiment")
        if (
            args.retention_proof is None
            or args.retention_audit is None
            or digest(args.retention_proof)
            != "sha256:04390ab231f102c01d10b62dd433fd3b84f1b92777a77e75ebc46fb7742d6790"
            or digest(args.retention_audit)
            != "sha256:4e0940b666cdf09d0d10d5dccf264dcd1c66c43ebee98f32773535a706eda880"
        ):
            raise ValueError("exact failed paired retention and independent audit required")
        retention = json.loads(args.retention_proof.read_text())
        audit = json.loads(args.retention_audit.read_text())
        if (
            retention["status"] != "TEAM_RECEIVING_EIGHT_BODY_FULL_CONSUMED_NOT_PROMOTED"
            or audit["status"] != "TEAM_RETENTION_TRACE_SELECTOR_READBACK_NOT_REINTEGRATED"
        ):
            raise ValueError("full failed retention lineage required")
    elif args.retention_proof is not None or args.retention_audit is not None:
        raise ValueError("retention proofs apply to coverage probe only")
    if learned_coverage:
        if (
            args.coverage_audit is None
            or digest(args.coverage_audit)
            != "sha256:314e90c233e14bc131c06763cd5318888eb4da4188dd1a3939efeb536a382769"
        ):
            raise ValueError("exact independently rescored coverage audit required")
    elif args.coverage_audit is not None:
        raise ValueError("coverage audit applies only to learned coverage")
    if args.full_retention:
        smoke_hash = "sha256:c17b889563ab27fa762929bf159204b565b19e44e043c66fbfc2b60ef2bc0793"
        if (
            args.smoke_proof is None
            or args.repeat_proof is None
            or args.smoke_proof.resolve() == args.repeat_proof.resolve()
            or digest(args.smoke_proof) != smoke_hash
            or digest(args.repeat_proof) != smoke_hash
        ):
            raise ValueError(
                "two distinct, matching original current-environment smoke reports required"
            )
        smoke = json.loads(args.smoke_proof.read_text())
        if len(smoke["results"]) != 2 or any(
            not r["parent_success"] or not all(r["retention"].values()) for r in smoke["results"]
        ):
            raise ValueError("all three seed smoke retention checks required")
    elif args.smoke_proof is not None or args.repeat_proof is not None:
        raise ValueError("smoke proof applies to full retention only")
    train_path = args.training_dir / "complete.json"
    if digest(train_path) != TRAINING_HASH:
        raise ValueError("exact complete three-seed training required")
    training = json.loads(train_path.read_text())
    if training["status"] != "TEAM_RECEIVING_OFFLINE_ACTORS_TRAINED_NOT_EVALUATED" or [
        r["seed"] for r in training["records"]
    ] != [17, 43, 71]:
        raise ValueError("all three unpromoted models required")
    v261 = _checked(args.v261_dir / "report.json")
    if v261["report_hash"] != PARENT_HASH or v261["selected_ramp"] != [0.7645, 0.766]:
        raise ValueError("exact original smooth parent required")
    reports = {
        name: _checked(getattr(args, name) / "report.json")
        for name in ("v229_dir", "v233_dir", "v236_dir", "v237_dir", "v254_dir", "v257_dir")
    }
    v224 = _checked(args.v224_report)
    common, (base, _, _), lineage = context(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v205_dir,
        args.map_report,
        args.v214_report,
        args.v215_dir,
        args.v216_report,
        args.v218_dir,
        args.v220_report,
    )
    if lineage["v220_report_hash"] != v224["v220_report_hash"]:
        raise ValueError("exact historical neural lineage required")
    base_bias = np.asarray(base.output_bias) + np.asarray(v224["best_offset"])
    pre = replace(base, output_bias=tuple(float(x) for x in base_bias))
    candidate = replace(
        base,
        output_bias=tuple(
            float(x) for x in base_bias + np.asarray(reports["v236_dir"]["best_offset"])
        ),
    )
    v229 = reports["v229_dir"]
    neural = _load_policy(args.v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    anchor = next(
        row for row in reports["v233_dir"]["candidates"] if row["source_episode"] == ANCHOR
    )
    post = replace(
        neural,
        output_bias=tuple(
            float(x) for x in np.asarray(neural.output_bias) + np.asarray(anchor["latent_offset"])
        ),
    )
    states = tuple(tuple(float(x) for x in row) for row in reports["v237_dir"]["activation_states"])
    threshold = float(reports["v254_dir"]["learned_lateral_threshold"])
    offset = tuple(float(x) for x in reports["v257_dir"]["selected_offset"])
    baseline = v261["results"][0]["rows"]
    if len(baseline) != 55:
        raise ValueError("complete original 55-course retention cohort required")
    coverage_seeds = (223003, 238004, 252008, 255004, 258005, 258007)
    chosen = (
        [next(row for row in baseline if row["course"]["seed"] == seed) for seed in coverage_seeds]
        if coverage_mode
        else baseline
        if args.full_retention
        else [baseline[0], baseline[53]]
    )
    courses = tuple(ReceivingCourse(**row["course"]) for row in chosen)
    if (
        not args.full_retention
        and not coverage_mode
        and [course.seed for course in courses] != [223001, 259011]
    ):
        raise ValueError("exact preregistered consumed smoke courses required")
    bank_hash = preflight_receiving_courses(courses)
    if args.full_retention and bank_hash != v261["course_bank_hash"]:
        raise ValueError("exact original complete retention bank required")
    models = {}
    pins = {
        str(train_path): digest(train_path),
        str(args.v261_dir / "report.json"): digest(args.v261_dir / "report.json"),
        str(args.policy): digest(args.policy),
    }
    if args.full_retention:
        pins[str(args.smoke_proof)] = smoke_hash
        pins[str(args.repeat_proof)] = smoke_hash
    if coverage_mode:
        pins[str(args.retention_proof)] = digest(args.retention_proof)
        pins[str(args.retention_audit)] = digest(args.retention_audit)
        matched = {r["course_seed"]: r for r in retention["results"]}
        for seed in coverage_seeds:
            group = matched[seed]
            parent_row = next(r for r in group["rows"] if r["arm"] == "parent")
            if (
                group["parent_success"]
                or parent_row["activation_enabled"]
                or parent_row["protected_episode"]
            ):
                raise ValueError("known unprotected action-gated failed training courses required")
    for row in training["records"]:
        path = args.training_dir / row["model_path"]
        if path.is_symlink() or digest(path) != row["model_hash"]:
            raise ValueError("exact seed checkpoint required")
        models[str(row["seed"])] = json.loads(path.read_text())
        TeamReceivingSelector.from_dict(models[str(row["seed"])])
        pins[str(path)] = row["model_hash"]
    if learned_coverage:
        learned_path = args.coverage_learned_dir / "complete.json"
        if (
            learned_path.is_symlink()
            or digest(learned_path)
            != "sha256:cd9e8cd98f4a6d19dc9a47ef72e80893d556338bea8256b06fd81004af3dbeaa"
        ):
            raise ValueError("exact three-seed coverage actor training required")
        learned = json.loads(learned_path.read_text())
        if learned["status"] != "TEAM_COVERAGE_SUPERVISED_ACTORS_NOT_EVALUATED" or [
            r["seed"] for r in learned["records"]
        ] != [17, 43, 71]:
            raise ValueError("three unpromoted offline actors required")
        pins[str(learned_path)] = digest(learned_path)
        pins[str(args.coverage_audit)] = digest(args.coverage_audit)
        for row in learned["records"]:
            path = args.coverage_learned_dir / row["model_path"]
            if (
                path.is_symlink()
                or path.parent != args.coverage_learned_dir
                or digest(path) != row["model_hash"]
            ):
                raise ValueError("exact confined coverage model required")
            models[f"learned-{row['seed']}"] = json.loads(path.read_text())
            TeamCoverageActor.from_dict(models[f"learned-{row['seed']}"])
            pins[str(path)] = row["model_hash"]
    for name in (
        "parent_report",
        "right_report",
        "refine_report",
        "lateral_report",
        "map_report",
        "v214_report",
        "v216_report",
        "v220_report",
        "v224_report",
    ):
        path = getattr(args, name)
        pins[str(path)] = digest(path)
    for name in (
        "v205_dir",
        "v215_dir",
        "v218_dir",
        "v229_dir",
        "v233_dir",
        "v236_dir",
        "v237_dir",
        "v254_dir",
        "v257_dir",
    ):
        for path in getattr(args, name).glob("*"):
            if path.is_file() and path.suffix in (".json", ".npz"):
                pins[str(path)] = digest(path)
    for folder in (
        args.asset_root / "g1_description",
        args.asset_root / "policy/robonaldo",
        args.asset_root / "policy/loco_mode",
    ):
        if not folder.is_dir():
            raise ValueError("qualified robot, locomotion and ball assets required")
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                pins[str(path)] = digest(path)
    source_pins(pins)
    check_pins(pins)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=args.expected_head,
            training_hash=TRAINING_HASH,
            learned_coverage_training_hash=digest(learned_path) if learned_coverage else None,
            parent_report_hash=PARENT_HASH,
            course_bank_hash=bank_hash,
            course_seeds=[c.seed for c in courses],
            arms=(
                ["parent", *(f"probe-{i}" for i in range(16))]
                if args.coverage_probe
                else ["parent", "learned-17", "learned-43", "learned-71"]
                if learned_coverage
                else ["parent", "17", "43", "71"]
            ),
            physical_runs=(17 if args.coverage_probe else 4) * len(courses),
            expanded_candidate_activation=(
                dict(
                    max_lateral=0.775,
                    min_relative_vx=-0.55,
                    activation_radius=0.022,
                    protected_episode_still_preserved=True,
                )
                if coverage_mode
                else None
            ),
            source_pins=pins,
            raw_trace_keys=PHYSICAL_KEYS,
            full_team_snapshot_fields=(
                "time, ball pose/velocity/contact, robot collision, each player's "
                "pelvis, joint position/velocity/torque/margin, feet and tactical decisions"
            ),
            historical_physical_hash_covers_only_raw_trace_keys=True,
            partition=(
                "CONSUMED_COVERAGE_PROBE"
                if args.coverage_probe
                else "CONSUMED_COVERAGE_LEARNED"
                if learned_coverage
                else "CONSUMED_FULL_RETENTION"
                if args.full_retention
                else "CONSUMED_SMOKE_ONLY"
            ),
            fresh_exam=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    jobs = []
    for course in courses:
        task = (common[0], common[1], course, *common[2:], pre, post, candidate, states)
        arms = (
            ["parent", *(f"probe-{i}" for i in range(16))]
            if args.coverage_probe
            else ["parent", "learned-17", "learned-43", "learned-71"]
            if learned_coverage
            else ["parent", "17", "43", "71"]
        )
        for arm in arms:
            jobs.append(
                (
                    arm,
                    course,
                    task,
                    offset,
                    threshold,
                    models["17"] if arm.startswith("probe-") else models.get(arm),
                    args.output / f"{course.seed}-{arm}.npz",
                )
            )
    rows = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            write_once(args.output / f"{row['course_seed']}-{row['arm']}.json", row)
            rows.append(row)
    check_pins(pins)
    summary = []
    for old, course in zip(chosen, courses, strict=True):
        group = [r for r in rows if r["course_seed"] == course.seed]
        parent = next(r for r in group if r["arm"] == "parent")
        retained = {
            r["arm"]: bool(
                not (is_clean(parent) and parent["controlled_reception"])
                or (is_clean(r) and r["controlled_reception"])
            )
            for r in group
            if r["arm"] != "parent"
        }
        summary.append(
            dict(
                course_seed=course.seed,
                parent_matches_historical_trace=(
                    parent["physical_trace_hash"] == old["physical_trace_hash"]
                ),
                parent_matches_current_retention=(
                    parent["physical_trace_hash"]
                    == next(r for r in matched[course.seed]["rows"] if r["arm"] == "parent")[
                        "physical_trace_hash"
                    ]
                    if coverage_mode
                    else None
                ),
                parent_success=bool(is_clean(parent) and parent["controlled_reception"]),
                retention=retained,
                rows=group,
            )
        )
    all_historical_exact = all(item["parent_matches_historical_trace"] for item in summary)
    compared_arms = (
        [f"probe-{i}" for i in range(16)]
        if args.coverage_probe
        else ["learned-17", "learned-43", "learned-71"]
        if learned_coverage
        else ["17", "43", "71"]
    )
    per_seed_retained = {
        arm: all(item["retention"][arm] for item in summary) for arm in compared_arms
    }
    write_once(
        args.output / "complete.json",
        dict(
            status=(
                "TEAM_RECEIVING_COVERAGE_PROBE_NOT_PROMOTED"
                if args.coverage_probe
                else "TEAM_RECEIVING_COVERAGE_LEARNED_NOT_PROMOTED"
                if learned_coverage
                else "TEAM_RECEIVING_EIGHT_BODY_FULL_CONSUMED_NOT_PROMOTED"
                if args.full_retention
                else "TEAM_RECEIVING_EIGHT_BODY_CONSUMED_SMOKE_NOT_PROMOTED"
            ),
            source_head=args.expected_head,
            training_hash=TRAINING_HASH,
            learned_coverage_training_hash=digest(learned_path) if learned_coverage else None,
            parent_report_hash=PARENT_HASH,
            course_bank_hash=bank_hash,
            results=summary,
            physical_runs=(17 if args.coverage_probe else 4) * len(courses),
            coverage_probe=args.coverage_probe,
            coverage_learned=learned_coverage,
            full_retention=args.full_retention,
            historical_parent_all_exact=all_historical_exact,
            current_parent_success_retained_by_seed=per_seed_retained,
            coverage_action_success_counts=(
                {
                    arm: sum(
                        is_clean(r) and r["controlled_reception"] for r in rows if r["arm"] == arm
                    )
                    for arm in compared_arms
                }
                if args.coverage_probe
                else None
            ),
            coverage_action_unsafe_counts=(
                {
                    arm: sum(
                        not r["safe"] or bool(r["fault_agents"]) for r in rows if r["arm"] == arm
                    )
                    for arm in compared_arms
                }
                if args.coverage_probe
                else None
            ),
            fresh_authorized=False,
            fresh_exam=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
