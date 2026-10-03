import copy

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_seal_recovered_sampling_collection import recovered_summary
from tests.rsi.test_independent_motor_learning_continuation import complete


def seal(value):
    value.pop("report_hash", None)
    value["report_hash"] = hash_json(value)
    return value


def inputs():
    summary, declared = complete()
    declared.update(
        schema="soccer.rsi.smooth_memory_exploration_commitment.v1",
        behavior_kind="OUTPUT_MEMORY_CURRENT_PARENT_AR1",
        sampling_rho=0.9,
        std_raw=0.1,
        execution_timeout_s=600,
        source_commit="original-worker",
        core_commit="original-core",
    )
    for i, row in enumerate(summary["rows"]):
        row["parent_report_hash"] = f"sha256:{234 + i * 2:064x}"
        row["greedy"] = dict(report_hash=f"sha256:{235 + i * 2:064x}")
        for s, sample in enumerate(row["samples"]):
            sample["report_hash"] = f"sha256:{i * 16 + s:064x}"
    started = seal(
        dict(
            schema="soccer.rsi.explicit_sampling_shard_recovery.v1",
            original_commitment_hash=hash_json(declared),
            actual_worker_commit="original-worker",
            actual_core_commit="original-core",
            gpu=0,
            course_indices=[0, 4, 8, 12],
            automatic_retry=False,
            reused_reports_are_not_new_physics=True,
            original_failed_run_must_be_preserved=True,
            completed_reports_require_full_reaudit=True,
            promotion_authorized=False,
            hardware_authorized=False,
        )
    )
    finished = seal(
        dict(
            schema="soccer.rsi.explicit_sampling_shard_recovery_result.v1",
            source_commitment_hash=started["report_hash"],
            completed_course_indices=[0, 4, 8, 12],
            original_failed_log_preserved=True,
            original_existing_bytes_preserved=True,
            entire_collection_sealed=False,
            learning_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
    )
    terminal = dict(
        schema="soccer.rsi.interrupted_independent_sampling_status.v1",
        collector_source_commit="original-worker",
        core_commit="original-core",
        collector_exit_code=1,
        completed_exploration_samples=179,
        required_exploration_samples=208,
        missing_exploration_samples=29,
        complete_course_rows=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
        failed_native_course=declared["courses"][8],
        failed_native_sample=3,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return declared, summary["rows"], started, finished, terminal


def test_complete_recovery_preserves_failure_and_exact_new_vs_reused_counts():
    values = inputs()
    before = copy.deepcopy(values)
    result = recovered_summary(*values)
    assert values == before
    provenance = result["recovery_provenance"]
    assert provenance["new_physical_executions_during_recovery"] == 31
    assert provenance["reused_physical_executions"] == 203
    assert provenance["clean_original_collector_success"] is False
    assert provenance["independent_audit_required"] is True
    assert result["high_quality_samples"] == 104
    assert result["report_hash"] == hash_json(
        {k: v for k, v in result.items() if k != "report_hash"}
    )


@pytest.mark.parametrize(
    "fault",
    [
        "rows",
        "sample",
        "label",
        "duplicate",
        "hash",
        "startseal",
        "finishseal",
        "link",
        "source",
        "core",
        "indices",
        "retry",
        "preserved",
        "authorized",
        "clean",
        "partial",
        "stream",
        "reward",
        "rho",
        "gpu_bool",
    ],
)
def test_partial_forged_changed_or_authorized_recovery_is_rejected(fault):
    declared, rows, started, finished, terminal = inputs()
    if fault == "rows":
        rows.pop()
    elif fault == "sample":
        rows[3]["samples"].pop()
    elif fault == "label":
        rows[3]["samples"][2]["high_quality"] = 1
    elif fault == "duplicate":
        rows[0]["samples"][1]["report_hash"] = rows[0]["samples"][0]["report_hash"]
    elif fault == "hash":
        rows[0]["parent_report_hash"] = "missing"
    elif fault == "startseal":
        started["gpu"] = 3
    elif fault == "finishseal":
        finished["original_existing_bytes_preserved"] = False
    elif fault == "link":
        finished["source_commitment_hash"] = "other"
    elif fault == "source":
        started["actual_worker_commit"] = "different"
    elif fault == "core":
        started["actual_core_commit"] = "different"
    elif fault == "indices":
        finished["completed_course_indices"] = [0, 4, 8]
    elif fault == "retry":
        started["automatic_retry"] = True
    elif fault == "preserved":
        finished["original_failed_log_preserved"] = False
    elif fault == "authorized":
        finished["learning_authorized"] = True
    elif fault == "clean":
        terminal["collector_exit_code"] = 0
    elif fault == "partial":
        terminal["completed_exploration_samples"] = 180
    elif fault == "stream":
        declared["exploration_stream"] = 1
    elif fault == "reward":
        declared["std_raw"] = 0.05
    elif fault == "rho":
        declared["sampling_rho"] = 0.5
    else:
        started["gpu"] = False
    if fault not in ("startseal", "finishseal"):
        seal(started)
        if fault != "link":
            finished["source_commitment_hash"] = started["report_hash"]
        seal(finished)
    with pytest.raises(ValueError):
        recovered_summary(declared, rows, started, finished, terminal)
