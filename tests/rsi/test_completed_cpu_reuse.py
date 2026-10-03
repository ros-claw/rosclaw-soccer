import copy

import pytest

from scripts.rsi_continue_memory_validation import check_cpu_reuse


def complete():
    raw = dict(
        step_model_hash="sha256:" + "a" * 64,
        seed=20261177,
        lane=0,
        report_hash="sha256:" + "b" * 64,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    review = dict(
        schema="soccer.rsi.cpu_motor_transfer_review.v1",
        physical_substeps=3000,
        reviewed_report_hash=raw["report_hash"],
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=True,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return raw, review, copy.deepcopy(review)


def test_complete_same_candidate_full_replay_can_be_reused():
    raw, review, replay = complete()
    check_cpu_reuse(raw, review, replay, model_hash=raw["step_model_hash"], course=(20261177, 0))


@pytest.mark.parametrize(
    "fault", ["model", "course", "replay", "frames", "float", "source", "authority", "motor"]
)
def test_partial_or_merely_hash_matching_reuse_rejected(fault):
    raw, review, replay = complete()
    if fault == "model":
        raw["step_model_hash"] = "sha256:" + "c" * 64
    elif fault == "course":
        raw["seed"] += 1
    elif fault == "replay":
        replay["changed"] = True
    elif fault == "frames":
        review["physical_substeps"] = replay["physical_substeps"] = 2999
    elif fault == "float":
        review["physical_substeps"] = replay["physical_substeps"] = 3000.0
    elif fault == "source":
        review["reviewed_report_hash"] = replay["reviewed_report_hash"] = "sha256:" + "d" * 64
    elif fault == "authority":
        raw["hardware_authorized"] = True
    else:
        review["neural_target_reconstructed"] = replay["neural_target_reconstructed"] = False
    with pytest.raises(ValueError, match="CPU replay"):
        check_cpu_reuse(raw, review, replay, model_hash="sha256:" + "a" * 64, course=(20261177, 0))
