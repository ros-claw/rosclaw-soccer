"""Construct paired SIM training examples without discarding unsafe outcomes.

The audit recomputes reception from snapshots; global safety remains a source
receipt, not an independently verified substep certificate. No training or
promotion occurs here.
"""

import numpy as np


def build_coverage_dataset(report: dict, audit: dict) -> dict[str, np.ndarray]:
    if (
        report.get("status") != "TEAM_RECEIVING_COVERAGE_PROBE_NOT_PROMOTED"
        or report.get("promotion_authorized") is not False
        or report.get("fresh_exam") is not False
        or audit.get("status") != "TEAM_RECEIVING_WINDOW_RECOMPUTED_NOT_REINTEGRATED"
        or audit.get("coverage") is not True
        or audit.get("global_safety_not_independently_recomputed") is not True
    ):
        raise ValueError("unpromoted coverage and explicit audit limits required")
    groups = report.get("results", [])
    audited = {(x["course_seed"], x["arm"]): x for x in audit.get("rows", [])}
    if len(groups) != 6 or len(audited) != 102 or len(audit.get("rows", [])) != 102:
        raise ValueError("six paired courses and 102 unique audited outcomes required")
    features, success, safety, enabled, parent_success, seeds = [], [], [], [], [], []
    for group in groups:
        seed = group["course_seed"]
        rows = {r["arm"]: r for r in group["rows"]}
        if len(rows) != 17 or set(rows) != {"parent", *(f"probe-{i}" for i in range(16))}:
            raise ValueError("one parent and sixteen distinct actions required")
        base = rows["parent"]
        x = np.asarray(base["activation_features"], dtype=np.float64)
        if x.shape != (10,) or not np.isfinite(x).all():
            raise ValueError("finite measured state required")
        outcomes, guards, active = [], [], []
        for arm in ("parent", *(f"probe-{i}" for i in range(16))):
            row, proof = rows[arm], audited[(seed, arm)]
            if (
                row["course_seed"] != seed
                or not np.array_equal(x, np.asarray(row["activation_features"]))
                or row["trace_file_hash"] != proof["raw_trace_hash"]
                or row["physical_trace_hash"] != proof["original_physical_hash"]
                or row["controlled_reception"] != proof["controlled_reception"]
                or row["safe"] != proof["source_global_safety"]
            ):
                raise ValueError("paired state, audit or original source receipt mismatch")
            outcomes.append(bool(proof["controlled_reception"]))
            guards.append(bool(row["safe"] and not row["fault_agents"]))
            active.append(bool(row["activation_enabled"] and not row["protected_episode"]))
        seeds.append(seed)
        features.append(x)
        success.append(outcomes[1:])
        safety.append(guards[1:])
        enabled.append(active[1:])
        parent_success.append(outcomes[0] and guards[0])
    if len(set(seeds)) != 6:
        raise ValueError("duplicate course seeds")
    return {
        "course_seed": np.asarray(seeds, dtype=np.int64),
        "features": np.asarray(features),
        "controlled_reception": np.asarray(success, dtype=bool),
        "source_safe_no_fault": np.asarray(safety, dtype=bool),
        "activation_enabled_unprotected": np.asarray(enabled, dtype=bool),
        "parent_safe_controlled": np.asarray(parent_success, dtype=bool),
    }
