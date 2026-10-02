import copy
import os

import pytest

from scripts.rsi_continue_independent_motor_learning import checked_collection, process_start


def complete():
    declaration = dict(
        samples_per_course=16,
        exploration_stream=2,
        exploration_stream_namespace="STREAM_STRIDE_20000000",
        partition="TRAIN_CONSUMED",
        sampling_construction="EXACT_CACHED_COMPLETE_MEAN_V1",
        courses=[[20261100 + i, 2] for i in range(13)],
        sampling_view_hashes=[f"sha256:{i:064x}" for i in range(208)],
        promotion_authorized=False,
        hardware_authorized=False,
    )
    rows = [
        dict(
            index=i,
            seed=seed,
            lane=lane,
            samples=[
                dict(
                    sample=s,
                    view_hash=declaration["sampling_view_hashes"][i * 16 + s],
                    high_quality=s % 2 == 0,
                )
                for s in range(16)
            ],
        )
        for i, (seed, lane) in enumerate(declaration["courses"])
    ]
    return dict(
        schema="soccer.rsi.smooth_memory_failure_exploration.v1",
        commitment=declaration,
        independent_contexts=13,
        exploration_executions=208,
        physical_executions=234,
        rows=rows,
        promotion_authorized=False,
        hardware_authorized=False,
    ), declaration


def test_complete_success_and_failure_bank_required():
    summary, declared = complete()
    checked_collection(summary, declared)
    assert any(not s["high_quality"] for r in summary["rows"] for s in r["samples"])


@pytest.mark.parametrize(
    "fault", ["count", "float", "sample", "row", "seed", "view", "authority", "stream"]
)
def test_partial_reordered_or_relabelled_collection_rejected(fault):
    summary, declared = complete()
    if fault == "count":
        summary["exploration_executions"] -= 1
    elif fault == "float":
        summary["physical_executions"] = 234.0
    elif fault == "sample":
        summary["rows"][2]["samples"].pop()
    elif fault == "row":
        summary["rows"].reverse()
    elif fault == "seed":
        declared["courses"][2] = declared["courses"][0]
    elif fault == "view":
        declared["sampling_view_hashes"][1] = declared["sampling_view_hashes"][0]
    elif fault == "authority":
        summary["hardware_authorized"] = True
    else:
        declared["exploration_stream"] = 1
    with pytest.raises(ValueError):
        checked_collection(summary, declared)


def test_actual_linux_process_start_and_exited_identity():
    before = process_start(os.getpid())
    assert before is not None and before.isdecimal()
    assert process_start(os.getpid()) == before
    assert process_start(2**31 - 1) is None


def test_changed_commitment_is_not_implicitly_accepted():
    summary, declared = complete()
    changed = copy.deepcopy(declared)
    changed["extra"] = "different source commitment"
    with pytest.raises(ValueError):
        checked_collection(summary, changed)
