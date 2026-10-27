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
            selector = TeamReceivingSelector.from_dict(model)
            prior.RADIUS = 0.019

            def factory(*args, **kwargs):
                actor = TeamReceivingSelectorMotor(
                    *args,
                    **kwargs,
                    selector=selector,
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
            row["selector_action"] = holder[0].selected_action
            row["selector_out_of_support"] = holder[0].selector_out_of_support
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
    args = parser.parse_args()
    clean(args.expected_head)
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence directory required")
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
    chosen = baseline if args.full_retention else [baseline[0], baseline[53]]
    courses = tuple(ReceivingCourse(**row["course"]) for row in chosen)
    if not args.full_retention and [course.seed for course in courses] != [223001, 259011]:
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
    for row in training["records"]:
        path = args.training_dir / row["model_path"]
        if path.is_symlink() or digest(path) != row["model_hash"]:
            raise ValueError("exact seed checkpoint required")
        models[str(row["seed"])] = json.loads(path.read_text())
        TeamReceivingSelector.from_dict(models[str(row["seed"])])
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
            parent_report_hash=PARENT_HASH,
            course_bank_hash=bank_hash,
            course_seeds=[c.seed for c in courses],
            arms=["parent", "17", "43", "71"],
            physical_runs=4 * len(courses),
            source_pins=pins,
            raw_trace_keys=PHYSICAL_KEYS,
            full_team_snapshot_fields=(
                "time, ball pose/velocity/contact, robot collision, each player's "
                "pelvis, joint position/velocity/torque/margin, feet and tactical decisions"
            ),
            historical_physical_hash_covers_only_raw_trace_keys=True,
            partition="CONSUMED_FULL_RETENTION" if args.full_retention else "CONSUMED_SMOKE_ONLY",
            fresh_exam=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    jobs = []
    for course in courses:
        task = (common[0], common[1], course, *common[2:], pre, post, candidate, states)
        for arm in ("parent", "17", "43", "71"):
            jobs.append(
                (
                    arm,
                    course,
                    task,
                    offset,
                    threshold,
                    models.get(arm),
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
                parent_success=bool(is_clean(parent) and parent["controlled_reception"]),
                retention=retained,
                rows=group,
            )
        )
    all_historical_exact = all(item["parent_matches_historical_trace"] for item in summary)
    per_seed_retained = {
        arm: all(item["retention"][arm] for item in summary) for arm in ("17", "43", "71")
    }
    write_once(
        args.output / "complete.json",
        dict(
            status=(
                "TEAM_RECEIVING_EIGHT_BODY_FULL_CONSUMED_NOT_PROMOTED"
                if args.full_retention
                else "TEAM_RECEIVING_EIGHT_BODY_CONSUMED_SMOKE_NOT_PROMOTED"
            ),
            source_head=args.expected_head,
            training_hash=TRAINING_HASH,
            parent_report_hash=PARENT_HASH,
            course_bank_hash=bank_hash,
            results=summary,
            physical_runs=4 * len(courses),
            full_retention=args.full_retention,
            historical_parent_all_exact=all_historical_exact,
            current_parent_success_retained_by_seed=per_seed_retained,
            fresh_authorized=False,
            fresh_exam=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
