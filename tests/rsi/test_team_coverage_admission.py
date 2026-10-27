import pytest

from rosclaw_soccer.rsi.team_coverage_admission import assess_coverage_retention


def evidence():
    report = dict(
        status="TEAM_RECEIVING_COVERAGE_LEARNED_FULL_RETENTION_NOT_PROMOTED",
        physical_runs=220,
        promotion_authorized=False,
        fresh_exam=False,
        results=[],
    )
    audit = dict(
        status="TEAM_COVERAGE_RETENTION_WINDOW_RECOMPUTED_NOT_REINTEGRATED",
        source_hash="sha256:source",
        trace_count=220,
        global_safety_not_independently_recomputed=True,
        rows=[],
    )
    for seed in range(55):
        group = dict(course_seed=seed, parent_matches_current_retention=True, rows=[])
        for arm in ("parent", "learned-17", "learned-43", "learned-71"):
            good = seed < (25 if arm == "parent" else 27)
            row = dict(
                course_seed=seed,
                arm=arm,
                safe=True,
                fault_agents=[],
                first_foot_frame=20 if good else None,
                own_nonfoot_frames=[],
                active_substeps=0,
                controlled_reception=good,
                physical_trace_hash=f"p-{seed}-{arm}",
                trace_file_hash=f"f-{seed}-{arm}",
            )
            group["rows"].append(row)
            audit["rows"].append(
                dict(
                    course_seed=seed,
                    arm=arm,
                    physical_trace_hash=row["physical_trace_hash"],
                    raw_trace_hash=row["trace_file_hash"],
                    controlled_reception=good,
                    source_global_safety=True,
                    source_fault_agents=[],
                )
            )
        report["results"].append(group)
    return report, audit


def test_admission_separates_missing_contact_from_safety():
    report, audit = evidence()
    result = assess_coverage_retention(report, audit, "sha256:source")
    assert result["development_generalization_admitted"]
    assert result["parent_success"] == 25
    assert result["candidates"]["learned-17"]["gains"] == 2
    assert result["candidates"]["learned-17"]["new_recorded_hazards"] == 0
    assert not result["fresh_exam_authorized"]


def test_admission_rejects_new_hazards_and_wrong_proof():
    report, audit = evidence()
    report["results"][40]["rows"][1]["own_nonfoot_frames"] = [21]
    assert not assess_coverage_retention(report, audit, "sha256:source")[
        "development_generalization_admitted"
    ]
    audit["rows"][0]["raw_trace_hash"] = "changed"
    with pytest.raises(ValueError, match="audited outcome"):
        assess_coverage_retention(report, audit, "sha256:source")
