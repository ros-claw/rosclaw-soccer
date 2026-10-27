"""Conservative, auditable admission of SIM coverage actors to development."""

ARMS = ("learned-17", "learned-43", "learned-71")


def clean_reception(row: dict) -> bool:
    return bool(
        row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
        and row["active_substeps"] <= 32
        and row["controlled_reception"]
    )


def recorded_hazard(row: dict) -> bool:
    return bool(
        not row["safe"]
        or row["fault_agents"]
        or row["own_nonfoot_frames"]
        or row["active_substeps"] > 32
    )


def assess_coverage_retention(report: dict, audit: dict, source_hash: str) -> dict:
    if (
        report.get("status") != "TEAM_RECEIVING_COVERAGE_LEARNED_FULL_RETENTION_NOT_PROMOTED"
        or report.get("physical_runs") != 220
        or report.get("promotion_authorized") is not False
        or report.get("fresh_exam") is not False
        or audit.get("status") != "TEAM_COVERAGE_RETENTION_WINDOW_RECOMPUTED_NOT_REINTEGRATED"
        or audit.get("source_hash") != source_hash
        or audit.get("trace_count") != 220
        or audit.get("global_safety_not_independently_recomputed") is not True
    ):
        raise ValueError("exact complete unpromoted report and limited audit required")
    groups = report.get("results", [])
    audit_rows = audit.get("rows", [])
    indexed = {(r["course_seed"], r["arm"]): r for r in audit_rows}
    if (
        len(groups) != 55
        or len({g["course_seed"] for g in groups}) != 55
        or len(audit_rows) != 220
        or len(indexed) != 220
    ):
        raise ValueError("55 unique paired courses and 220 audited rows required")
    stats = {
        arm: {"success": 0, "gains": 0, "losses": 0, "new_recorded_hazards": 0} for arm in ARMS
    }
    parent_success = 0
    for group in groups:
        seed = group["course_seed"]
        rows = {r["arm"]: r for r in group["rows"]}
        if (
            len(group["rows"]) != 4
            or set(rows) != {"parent", *ARMS}
            or group.get("parent_matches_current_retention") is not True
        ):
            raise ValueError("unique paired actors and exact current parent required")
        parent = rows["parent"]
        p = clean_reception(parent)
        parent_success += p
        for arm, row in rows.items():
            proof = indexed[(seed, arm)]
            if (
                row["course_seed"] != seed
                or proof["physical_trace_hash"] != row["physical_trace_hash"]
                or proof["raw_trace_hash"] != row["trace_file_hash"]
                or proof["controlled_reception"] != row["controlled_reception"]
                or proof["source_global_safety"] != row["safe"]
                or proof["source_fault_agents"] != row["fault_agents"]
            ):
                raise ValueError("audited outcome differs from source")
            if arm == "parent":
                continue
            ok = clean_reception(row)
            stats[arm]["success"] += ok
            stats[arm]["gains"] += ok and not p
            stats[arm]["losses"] += p and not ok
            stats[arm]["new_recorded_hazards"] += recorded_hazard(row) and not recorded_hazard(
                parent
            )
    passed = all(
        v["gains"] >= 2 and v["losses"] == 0 and v["new_recorded_hazards"] == 0
        for v in stats.values()
    )
    return dict(
        parent_success=parent_success,
        candidates=stats,
        development_generalization_admitted=passed,
        independent_substep_safety_certified=False,
        fresh_exam_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
