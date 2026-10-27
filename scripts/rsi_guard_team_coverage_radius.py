"""Shrink candidate SIM action support using an audited failed development case."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.team_coverage_actor import TeamCoverageActor
from rosclaw_soccer.rsi.team_coverage_admission import recorded_hazard
from rosclaw_soccer.rsi.team_coverage_radius import radius_below_failure
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "development", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("models-hash", "development-hash", "audit-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    paths = (
        args.models / "complete.json",
        args.development / "complete.json",
        args.audit / "complete.json",
    )
    hashes = (args.models_hash, args.development_hash, args.audit_hash)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True))
    ):
        raise ValueError("exact unpromoted models, development, audit and unused output required")
    training, report, audit = (json.loads(p.read_text()) for p in paths)
    if (
        training["status"] != "TEAM_COVERAGE_SUPERVISED_ACTORS_NOT_EVALUATED"
        or report["status"] != "TEAM_RECEIVING_NEIGHBOR_DEVELOPMENT_NOT_PROMOTED"
        or report["physical_runs"] != 48
        or audit["status"] != "TEAM_COVERAGE_NEIGHBOR_WINDOW_RECOMPUTED_NOT_REINTEGRATED"
        or audit["source_hash"] != args.development_hash
        or audit["models_hash"] != args.models_hash
        or audit["global_safety_not_independently_recomputed"] is not True
    ):
        raise ValueError("audited neighbor failure and original three actors required")
    models, model_pins = {}, {}
    for row in training["records"]:
        path = args.models / row["model_path"]
        if path.is_symlink() or path.parent != args.models or digest(path) != row["model_hash"]:
            raise ValueError("exact confined original actor required")
        value = json.loads(path.read_text())
        model_pins[str(path)] = row["model_hash"]
        models[f"learned-{row['seed']}"] = (row, value, TeamCoverageActor.from_dict(value))
    if set(models) != {"learned-17", "learned-43", "learned-71"} or len(report["results"]) != 12:
        raise ValueError("all three actors and twelve paired courses required")
    failures = {arm: [] for arm in models}
    for group in report["results"]:
        arms = {r["arm"]: r for r in group["rows"]}
        if len(arms) != 4 or set(arms) != {"parent", *models}:
            raise ValueError("complete paired development outcomes required")
        for arm, (_, _, actor) in models.items():
            row = arms[arm]
            if recorded_hazard(row) and not recorded_hazard(arms["parent"]):
                if row["selector_action"] == 16:
                    raise ValueError(
                        "fallback cannot be used as evidence for an intervention hazard"
                    )
                x = np.asarray(row["activation_features"], dtype=np.float64)
                distance = float(np.min(np.linalg.norm(actor.support - x, axis=1)))
                failures[arm].append(
                    dict(
                        course_seed=group["course_seed"],
                        action=row["selector_action"],
                        distance=distance,
                        physical_trace_hash=row["physical_trace_hash"],
                    )
                )
    if any(len(rows) != 1 or rows[0]["course_seed"] != 410001 for rows in failures.values()):
        raise ValueError("all seeded models must reproduce the same single new hazard")
    distances = [rows[0]["distance"] for rows in failures.values()]
    radius = radius_below_failure(float(min(distances)), prior_radius=0.004)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            models_hash=args.models_hash,
            development_hash=args.development_hash,
            audit_hash=args.audit_hash,
            source_head=args.expected_head,
            nearest_failure=failures,
            derived_radius=radius,
            rule="largest 0.00005-grid radius strictly below nearest new recorded hazard",
            post_hoc_development=True,
            optimizer_updates=0,
            motor_safety_certified=False,
            hardware_authorized=False,
        ),
    )
    records = []
    for record, original, actor in models.values():
        changed = dict(original, radius=radius)
        candidate = TeamCoverageActor.from_dict(changed)
        if any(candidate.select(tuple(x))[0] != actor.select(tuple(x))[0] for x in actor.support):
            raise ValueError("all original six training state choices must be retained")
        path = args.output / f"seed-{record['seed']}.json"
        write_once(path, changed)
        records.append(
            dict(
                seed=record["seed"],
                model_path=path.name,
                model_hash=digest(path),
                original_model_hash=record["model_hash"],
            )
        )
    if any(digest(p) != h for p, h in zip(paths, hashes, strict=True)) or any(
        digest(Path(p)) != h for p, h in model_pins.items()
    ):
        raise ValueError("source changed during guard construction")
    write_once(
        args.output / "complete.json",
        dict(
            status="TEAM_COVERAGE_FAILURE_DERIVED_RADIUS_NOT_EVALUATED",
            source_head=args.expected_head,
            models_hash=args.models_hash,
            development_hash=args.development_hash,
            audit_hash=args.audit_hash,
            radius=radius,
            failures=failures,
            records=records,
            post_hoc_development=True,
            optimizer_updates=0,
            physical_evaluation=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
