import copy
import subprocess
import sys

import pytest
from test_joint_delivery_teacher import row
from test_stochastic_refinement_pilot import fixture as old_fixture

from rosclaw_soccer.rsi.refinement_sampling_exam import (
    paired_rewards,
    sampling_exam_config,
    verify_sampling_exam,
)
from rosclaw_soccer.rsi.stochastic_refinement_pilot import pilot_config


def fixture():
    old, courses = old_fixture()
    cfg = sampling_exam_config(25)
    rows = []
    for i in range(48):
        r = copy.deepcopy(old["records"][i // 2])
        r.update(index=i, arm=cfg["arms"][i % 2], sampling_seed=cfg["seeds"][(i // 2) % 8])
        r["outcome"] = row(0.6 if i % 2 == 0 else 0.3, 2.7)["outcome"]
        rows.append(r)
    return dict(
        config=cfg, records=rows, counts=dict(native_steps=186720, actor_calls=37344)
    ), courses


def test_new_noise_complete_paired48_not_generalization_or_training():
    r, courses = fixture()
    verify_sampling_exam(r, courses)
    assert not set(r["config"]["seeds"]) & set(pilot_config(25)["seeds"])
    result = paired_rewards(r["records"])
    assert result["paired_noise_samples"] == 24 and result["unique_consumed_TRAIN_initials"] == 3
    assert result["positive_pairs"] == 24 and result["negative_pairs"] == 0
    assert result["mean_paired_reward_difference"] > 0
    for k in (
        "generalization_proven",
        "deterministic_mean_improvement_proven",
        "promotion_authorized",
        "goal_complete",
    ):
        assert result[k] is False


@pytest.mark.parametrize(
    "fault", ["missing", "arm", "seed", "course", "count", "teacher", "credit", "tail"]
)
def test_mixed_incomplete_or_wrong_arms_rejected(fault):
    r, courses = fixture()
    if fault == "missing":
        r["records"].pop()
    elif fault == "arm":
        r["records"][1]["arm"] = "old"
    elif fault == "seed":
        r["records"][1]["sampling_seed"] += 1
    elif fault == "course":
        r["records"][1]["course"] = courses[1]
    elif fault == "count":
        r["counts"]["native_steps"] += 10
    elif fault == "teacher":
        r["records"][1]["external_teacher"] = True
    elif fault == "credit":
        r["records"][1]["PPO_actor_rows"] = 289
    else:
        r["records"][1]["stand_controls"] = 99
    with pytest.raises(ValueError):
        verify_sampling_exam(r, courses)
    if fault in ("missing", "arm", "seed", "course"):
        with pytest.raises(ValueError):
            paired_rewards(r["records"])


def test_failure_kept_with_actual_clocks_and_negative_reward():
    r, courses = fixture()
    x = r["records"][1]
    x.update(
        termination="BODY_FAILURE",
        controls=70,
        stochastic_controls=70,
        stand_controls=0,
        observed_recovery_controls=0,
        total_observed_controls=70,
        native_steps=700,
        actor_calls=140,
    )
    x["outcome"]["whole_body_healthy"] = False
    r["counts"] = {k: sum(x[k] for x in r["records"]) for k in ("native_steps", "actor_calls")}
    verify_sampling_exam(r, courses)
    assert paired_rewards(r["records"])["negative_pairs"] == 1


def test_incomplete_audit_args_never_allocate(tmp_path):
    output = tmp_path / "unused"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_run_refinement_sampling_exam",
            "--experience-root",
            "x",
            "--own-root",
            "x",
            "--training-root",
            "x",
            "--output",
            str(output),
            "--own-hash",
            "h",
            "--training-hash",
            "h",
            "--expected-head",
            "h",
            "--state-id",
            "25",
            "--audit-source-root",
            "x",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and not output.exists()
