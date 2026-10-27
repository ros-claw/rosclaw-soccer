import copy

import pytest

from rosclaw_soccer.rsi.team_coverage_dataset import build_coverage_dataset


def paired():
    report = {
        "status": "TEAM_RECEIVING_COVERAGE_PROBE_NOT_PROMOTED",
        "promotion_authorized": False,
        "fresh_exam": False,
        "results": [],
    }
    audit = {
        "status": "TEAM_RECEIVING_WINDOW_RECOMPUTED_NOT_REINTEGRATED",
        "coverage": True,
        "global_safety_not_independently_recomputed": True,
        "rows": [],
    }
    for seed in range(6):
        group = {"course_seed": seed, "rows": []}
        for arm in ("parent", *(f"probe-{i}" for i in range(16))):
            row = {
                "course_seed": seed,
                "arm": arm,
                "activation_features": [float(seed)] * 10,
                "trace_file_hash": f"file-{seed}-{arm}",
                "physical_trace_hash": f"physics-{seed}-{arm}",
                "controlled_reception": arm == "probe-1",
                "safe": arm != "probe-2",
                "fault_agents": [],
                "activation_enabled": arm != "parent",
                "protected_episode": False,
            }
            group["rows"].append(row)
            audit["rows"].append(
                {
                    "course_seed": seed,
                    "arm": arm,
                    "raw_trace_hash": row["trace_file_hash"],
                    "original_physical_hash": row["physical_trace_hash"],
                    "controlled_reception": row["controlled_reception"],
                    "source_global_safety": row["safe"],
                }
            )
        report["results"].append(group)
    return report, audit


def test_coverage_keeps_success_and_unsafe_separately():
    report, audit = paired()
    data = build_coverage_dataset(report, audit)
    assert data["features"].shape == (6, 10)
    assert data["controlled_reception"].sum() == 6
    assert (~data["source_safe_no_fault"]).sum() == 6
    assert not data["parent_safe_controlled"].any()


@pytest.mark.parametrize("mutation", ["trace", "state", "duplicate", "audit_limit"])
def test_coverage_rejects_unpaired_or_overclaimed_data(mutation):
    report, audit = paired()
    if mutation == "trace":
        audit["rows"][1]["raw_trace_hash"] = "different"
    elif mutation == "state":
        report["results"][0]["rows"][1]["activation_features"][0] = 9
    elif mutation == "duplicate":
        audit["rows"][1] = copy.deepcopy(audit["rows"][0])
    else:
        audit["global_safety_not_independently_recomputed"] = False
    with pytest.raises(ValueError):
        build_coverage_dataset(report, audit)
