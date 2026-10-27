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
from rosclaw_soccer.rsi.team_coverage_admission import assess_coverage_retention
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
    parser.add_argument("--coverage-full-retention", action="store_true")
    parser.add_argument("--protected-smoke", action="store_true")
    parser.add_argument("--protection-proof", type=Path)
    parser.add_argument("--protection-audit", type=Path)
    parser.add_argument("--coverage-audit", type=Path)
    parser.add_argument("--learned-proof", type=Path)
    parser.add_argument("--learned-audit", type=Path)
    parser.add_argument("--neighbor-bank", type=Path)
    parser.add_argument("--neighbor-bank-hash")
    parser.add_argument("--radius-guard-dir", type=Path)
    parser.add_argument("--radius-guard-hash")
    parser.add_argument("--guard-proof", type=Path)
    parser.add_argument("--guard-audit", type=Path)
    parser.add_argument("--full-retention-report", type=Path)
    parser.add_argument("--full-retention-audit", type=Path)
    parser.add_argument("--admission-proof", type=Path)
    parser.add_argument("--admission-hash")
    parser.add_argument("--retention-proof", type=Path)
    parser.add_argument("--retention-audit", type=Path)
    args = parser.parse_args()
    clean(args.expected_head)
    learned_coverage = args.coverage_learned_dir is not None
    coverage_mode = args.coverage_probe or learned_coverage
    neighbor = args.neighbor_bank is not None
    guard_mode = args.radius_guard_dir is not None
    if guard_mode and (
        not (neighbor or args.coverage_full_retention) or not args.radius_guard_hash
    ):
        raise ValueError("failure-derived guard requires paired neighbor or full retention")
    if not guard_mode and args.radius_guard_hash is not None:
        raise ValueError("guard hash requires an explicit guard directory")
    if args.coverage_full_retention and not learned_coverage:
        raise ValueError("full learned retention requires trained coverage actors")
    if args.protected_smoke and (
        not learned_coverage or args.coverage_full_retention or args.coverage_probe or neighbor
    ):
        raise ValueError("protected smoke requires only the frozen learned actors")
    if neighbor and (not learned_coverage or args.coverage_full_retention or args.coverage_probe):
        raise ValueError("neighbor development requires the frozen learned actors only")
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
    if args.coverage_full_retention:
        if guard_mode:
            if (
                args.guard_proof is None
                or args.guard_audit is None
                or args.guard_proof.is_symlink()
                or args.guard_audit.is_symlink()
                or digest(args.guard_proof)
                != "sha256:170488c6794d087ab75a2dfc36913e9b9326a621cdcd2f5679795b3f80b912d4"
                or digest(args.guard_audit)
                != "sha256:d75fa447bea470c64a14701a3a478340799903ae1d7c440283404ef61db0e23d"
            ):
                raise ValueError("exact audited posthoc failure guard development required")
            guard_proof = json.loads(args.guard_proof.read_text())
            guard_audit = json.loads(args.guard_audit.read_text())
            if (
                guard_proof["status"] != "TEAM_RECEIVING_NEIGHBOR_RADIUS_GUARD_NOT_PROMOTED"
                or guard_audit["status"]
                != "TEAM_COVERAGE_NEIGHBOR_WINDOW_RECOMPUTED_NOT_REINTEGRATED"
                or guard_audit["source_hash"] != digest(args.guard_proof)
                or guard_audit["models_hash"] != args.radius_guard_hash
                or guard_audit["posthoc_failure_guard"] is not True
                or set(guard_audit["candidates"]) != {"learned-17", "learned-43", "learned-71"}
                or any(
                    v["gains"] < 2 or v["losses"] or v["new_recorded_hazards"]
                    for v in guard_audit["candidates"].values()
                )
            ):
                raise ValueError("all seed posthoc guard development gates required")
        if (
            args.protection_proof is None
            or args.protection_audit is None
            or args.protection_proof.is_symlink()
            or args.protection_audit.is_symlink()
            or digest(args.protection_proof)
            != "sha256:58dda99ebe752368bf02eba2c3507eedcea1f7fcc74420df2a5c98907353a8f8"
            or digest(args.protection_audit)
            != "sha256:646f8d52e0ca4dbc0ea63018172653b8bb1fdedea818cb6f15a1f27fa4a1f4dd"
        ):
            raise ValueError("exact audited protection-order physical smoke required")
        protection = json.loads(args.protection_proof.read_text())
        protection_audit = json.loads(args.protection_audit.read_text())
        if (
            protection["status"] != "TEAM_RECEIVING_PROTECTION_ORDER_SMOKE_NOT_PROMOTED"
            or protection_audit["source_hash"] != digest(args.protection_proof)
            or protection_audit["protected_smoke"] is not True
            or len(protection_audit["rows"]) != 8
        ):
            raise ValueError("complete independently rescored protection-order smoke required")
        if (
            args.learned_proof is None
            or args.learned_audit is None
            or args.learned_proof.is_symlink()
            or args.learned_audit.is_symlink()
            or digest(args.learned_proof)
            != "sha256:2c5e91667e279c623ff1ba05f29b7d183f314d134e4cada1c3b24f1433f3d3f9"
            or digest(args.learned_audit)
            != "sha256:559063c0fabfd5c8493ed8a262950b22e11e396d40ec683115b84138253260d7"
        ):
            raise ValueError("exact learned-course simulation and audit required")
        learned_proof = json.loads(args.learned_proof.read_text())
        learned_audit = json.loads(args.learned_audit.read_text())
        if (
            learned_proof["status"] != "TEAM_RECEIVING_COVERAGE_LEARNED_NOT_PROMOTED"
            or learned_audit["source_hash"] != digest(args.learned_proof)
            or learned_audit["learned_coverage"] is not True
            or len(learned_audit["rows"]) != 24
            or any(not g["parent_matches_current_retention"] for g in learned_proof["results"])
            or any(
                sum(
                    is_clean(r) and r["controlled_reception"]
                    for g in learned_proof["results"]
                    for r in g["rows"]
                    if r["arm"] == arm
                )
                != 5
                or any(
                    not is_clean(r)
                    for g in learned_proof["results"]
                    for r in g["rows"]
                    if r["arm"] == arm
                )
                for arm in ("learned-17", "learned-43", "learned-71")
            )
        ):
            raise ValueError("all seed training-course physical and safety gates required")
    elif args.learned_proof is not None or args.learned_audit is not None:
        raise ValueError("learned proof applies only to full learned retention")
    if not args.coverage_full_retention and (args.protection_proof or args.protection_audit):
        raise ValueError("protection proof applies only to full learned retention")
    if not (guard_mode and args.coverage_full_retention) and (args.guard_proof or args.guard_audit):
        raise ValueError("guard proof applies only to guarded full retention")
    if neighbor:
        required = (
            args.neighbor_bank,
            args.full_retention_report,
            args.full_retention_audit,
            args.admission_proof,
        )
        if (
            any(path is None or path.is_symlink() for path in required)
            or not args.neighbor_bank_hash
            or not args.admission_hash
            or digest(args.neighbor_bank) != args.neighbor_bank_hash
            or digest(args.admission_proof) != args.admission_hash
        ):
            raise ValueError("pinned new development and audited admission required")
        manifest = json.loads(args.neighbor_bank.read_text())
        admission = json.loads(args.admission_proof.read_text())
        if (
            manifest.get("schema") != "soccer.coverage_neighbor_development.v1"
            or manifest.get("status") != "PREDECLARED_NOT_EXECUTED"
            or manifest.get("fresh_exam") is not False
            or manifest.get("promotion_authorized") is not False
            or manifest.get("hardware_authorized") is not False
            or len(manifest.get("courses", [])) != 12
            or admission.get("status") != "TEAM_COVERAGE_RETENTION_ASSESSED_NOT_PROMOTED"
            or digest(args.full_retention_report) != admission["source_hash"]
            or digest(args.full_retention_audit) != admission["audit_hash"]
        ):
            raise ValueError("unopened development bank and bound admission required")
        full_report = json.loads(args.full_retention_report.read_text())
        full_audit = json.loads(args.full_retention_audit.read_text())
        reassessed = assess_coverage_retention(full_report, full_audit, admission["source_hash"])
        if (
            reassessed != admission["result"]
            or not reassessed["development_generalization_admitted"]
        ):
            raise ValueError("all seeded nonregression, gain and hazard gates required")
        if (
            full_report.get("learned_coverage_training_hash")
            != "sha256:cd9e8cd98f4a6d19dc9a47ef72e80893d556338bea8256b06fd81004af3dbeaa"
        ):
            raise ValueError("admitted and tested model lineage must match")
    elif any(
        value is not None
        for value in (
            args.neighbor_bank_hash,
            args.full_retention_report,
            args.full_retention_audit,
            args.admission_proof,
            args.admission_hash,
        )
    ):
        raise ValueError("neighbor evidence only applies to development")
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
        [dict(course=course, physical_trace_hash=None) for course in manifest["courses"]]
        if neighbor
        else [
            next(row for row in baseline if row["course"]["seed"] == seed)
            for seed in (223003, 259010)
        ]
        if args.protected_smoke
        else [
            next(row for row in baseline if row["course"]["seed"] == seed)
            for seed in coverage_seeds
        ]
        if coverage_mode and not args.coverage_full_retention
        else baseline
        if args.full_retention or args.coverage_full_retention
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
    if (
        neighbor
        and bank_hash != "sha256:60a30fe74e052a141dd295f00b74cec71ef8c708612adfbd6c068f3c34359398"
    ):
        raise ValueError("exact preregistered twelve-course development bank required")
    if (args.full_retention or args.coverage_full_retention) and bank_hash != v261[
        "course_bank_hash"
    ]:
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
    if args.coverage_full_retention:
        pins[str(args.learned_proof)] = digest(args.learned_proof)
        pins[str(args.learned_audit)] = digest(args.learned_audit)
        pins[str(args.protection_proof)] = digest(args.protection_proof)
        pins[str(args.protection_audit)] = digest(args.protection_audit)
        if guard_mode:
            pins[str(args.guard_proof)] = digest(args.guard_proof)
            pins[str(args.guard_audit)] = digest(args.guard_audit)
    if neighbor:
        for path in required:
            pins[str(path)] = digest(path)
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
    if guard_mode:
        guard_path = args.radius_guard_dir / "complete.json"
        if guard_path.is_symlink() or digest(guard_path) != args.radius_guard_hash:
            raise ValueError("exact failure-derived guard receipt required")
        guard = json.loads(guard_path.read_text())
        if (
            guard["status"] != "TEAM_COVERAGE_FAILURE_DERIVED_RADIUS_NOT_EVALUATED"
            or guard["models_hash"] != digest(learned_path)
            or guard["development_hash"]
            != "sha256:6c013e4f8a469f0ed7496eb0e60abb9e88aaa1c467160e81fd65fa41b2e7300e"
            or guard["audit_hash"]
            != "sha256:e73f4bbeb86fe86e4cb8b339f4f9a804b0bd7dc386d67b0f09c331726ebfba2f"
            or [r["seed"] for r in guard["records"]] != [17, 43, 71]
            or guard["promotion_authorized"] is not False
        ):
            raise ValueError("bound failed development and three original actors required")
        pins[str(guard_path)] = args.radius_guard_hash
        for record in guard["records"]:
            path = args.radius_guard_dir / record["model_path"]
            arm = f"learned-{record['seed']}"
            if (
                path.is_symlink()
                or path.parent != args.radius_guard_dir
                or digest(path) != record["model_hash"]
            ):
                raise ValueError("exact confined radius-guard actor required")
            value = json.loads(path.read_text())
            old = models[arm]
            if {k: v for k, v in value.items() if k != "radius"} != {
                k: v for k, v in old.items() if k != "radius"
            } or not 0 < value["radius"] < old["radius"]:
                raise ValueError("guard may only reduce original actor support radius")
            TeamCoverageActor.from_dict(value)
            models[arm] = value
            pins[str(path)] = record["model_hash"]
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
            effective_coverage_model_hash=digest(guard_path)
            if guard_mode
            else digest(learned_path)
            if learned_coverage
            else None,
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
                else "CONSUMED_PROTECTION_ORDER_SMOKE"
                if args.protected_smoke
                else "CONSUMED_NEIGHBOR_FAILURE_RADIUS_GUARD"
                if guard_mode and neighbor
                else "CONSUMED_FULL_RETENTION_FAILURE_RADIUS_GUARD"
                if guard_mode
                else "NEW_NEIGHBOR_DEVELOPMENT_NOT_FRESH"
                if neighbor
                else "CONSUMED_COVERAGE_LEARNED_FULL_RETENTION"
                if args.coverage_full_retention
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
                    if coverage_mode and not neighbor
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
                else "TEAM_RECEIVING_PROTECTION_ORDER_SMOKE_NOT_PROMOTED"
                if args.protected_smoke
                else "TEAM_RECEIVING_NEIGHBOR_RADIUS_GUARD_NOT_PROMOTED"
                if guard_mode and neighbor
                else "TEAM_RECEIVING_COVERAGE_RADIUS_FULL_RETENTION_NOT_PROMOTED"
                if guard_mode
                else "TEAM_RECEIVING_NEIGHBOR_DEVELOPMENT_NOT_PROMOTED"
                if neighbor
                else "TEAM_RECEIVING_COVERAGE_LEARNED_FULL_RETENTION_NOT_PROMOTED"
                if args.coverage_full_retention
                else "TEAM_RECEIVING_COVERAGE_LEARNED_NOT_PROMOTED"
                if learned_coverage
                else "TEAM_RECEIVING_EIGHT_BODY_FULL_CONSUMED_NOT_PROMOTED"
                if args.full_retention
                else "TEAM_RECEIVING_EIGHT_BODY_CONSUMED_SMOKE_NOT_PROMOTED"
            ),
            source_head=args.expected_head,
            training_hash=TRAINING_HASH,
            learned_coverage_training_hash=digest(learned_path) if learned_coverage else None,
            effective_coverage_model_hash=digest(guard_path)
            if guard_mode
            else digest(learned_path)
            if learned_coverage
            else None,
            parent_report_hash=PARENT_HASH,
            course_bank_hash=bank_hash,
            results=summary,
            physical_runs=(17 if args.coverage_probe else 4) * len(courses),
            coverage_probe=args.coverage_probe,
            coverage_learned=learned_coverage,
            coverage_full_retention=args.coverage_full_retention,
            protected_smoke=args.protected_smoke,
            neighbor_development=neighbor,
            failure_derived_radius_guard=guard_mode,
            full_retention=args.full_retention,
            historical_parent_all_exact=all_historical_exact,
            current_parent_success_retained_by_seed=per_seed_retained,
            current_parent_success_count=sum(item["parent_success"] for item in summary),
            candidate_success_count={
                arm: sum(is_clean(r) and r["controlled_reception"] for r in rows if r["arm"] == arm)
                for arm in compared_arms
            },
            candidate_safety_failure_count={
                arm: sum(
                    not r["safe"]
                    or bool(r["fault_agents"])
                    or bool(r["own_nonfoot_frames"])
                    or r["active_substeps"] > 32
                    for r in rows
                    if r["arm"] == arm
                )
                for arm in compared_arms
            },
            candidate_unclean_count={
                arm: sum(not is_clean(r) for r in rows if r["arm"] == arm) for arm in compared_arms
            },
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
