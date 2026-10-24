import copy
import subprocess
import sys

import pytest

from rosclaw_soccer.rsi.stochastic_refinement_pilot import pilot_config, verify_study


def fixture():
    cfg = pilot_config(25)
    courses = [
        dict(state_id=25, task=1, assisted=False, ball_planar_offset_world=[x, 0.0])
        for x in (0.0, -0.005, 0.005)
    ]
    rows = []
    for i in range(24):
        rows.append(
            dict(
                index=i,
                course=copy.deepcopy(courses[i // 8]),
                sampling_seed=cfg["seeds"][i % 8],
                sampling_profile="contact-v1",
                sampling_law="box-truncated-ar1",
                controller_kind="pure-refinement-stochastic-native-v1",
                assisted=False,
                external_teacher=False,
                authority_rejected=False,
                all_actor_actions_actually_executed=True,
                natural_terminal_observed=True,
                deterministic_recovery_excluded_from_actor_rows=True,
                requires_whole_native_and_law_audit=True,
                PPO_actor_rows=0,
                neural_actor_optimizer_updates=0,
                stochastic_controls=289,
                motion_reference_controls=289,
                controls=389,
                stand_controls=100,
                observed_recovery_controls=100,
                total_observed_controls=389,
                native_steps=3890,
                actor_calls=778,
                termination="FULL_MOTION_PLUS100NN_RECOVERY",
            )
        )
    report = dict(config=cfg, records=rows, counts=dict(native_steps=93360, actor_calls=18672))
    return report, courses


def test_finite_exact_3initials_8seeds_no_teacher_or_optimizer():
    report, courses = fixture()
    verify_study(report, courses)
    c = report["config"]
    assert len(set(c["seeds"])) == 8 and c["new_optimizer_updates"] == 0
    assert c["maximum_native_steps"] == 96000 and c["Fresh_cases_read"] == 0


@pytest.mark.parametrize(
    "fault", ["missing", "seed", "teacher", "credit", "count", "course", "clock", "tail"]
)
def test_partial_mixed_or_teacher_data_not_onpolicy_calibration(fault):
    report, courses = fixture()
    r = report["records"][0]
    if fault == "missing":
        report["records"].pop()
    elif fault == "seed":
        r["sampling_seed"] += 1
    elif fault == "teacher":
        r["external_teacher"] = True
    elif fault == "credit":
        r["PPO_actor_rows"] = 289
    elif fault == "count":
        report["counts"]["native_steps"] += 10
    elif fault == "course":
        r["course"] = courses[1]
    elif fault == "clock":
        r["controls"] = 400
    else:
        r["stand_controls"] = 99
    with pytest.raises(ValueError):
        verify_study(report, courses)


def test_observed_body_failure_is_kept_not_dropped_or_padded():
    report, courses = fixture()
    r = report["records"][0]
    r.update(
        termination="BODY_FAILURE",
        controls=70,
        stochastic_controls=70,
        stand_controls=0,
        observed_recovery_controls=0,
        total_observed_controls=70,
        native_steps=700,
        actor_calls=140,
    )
    report["counts"] = dict(
        native_steps=sum(x["native_steps"] for x in report["records"]),
        actor_calls=sum(x["actor_calls"] for x in report["records"]),
    )
    verify_study(report, courses)


@pytest.mark.parametrize(
    "extra",
    [
        ["--audit-source-root", "x"],
        ["--producer-head", "h"],
        ["--audit-source-root", "x", "--producer-head", "h", "--preflight-only"],
    ],
)
def test_invalid_audit_cli_never_creates_output(tmp_path, extra):
    output = tmp_path / "unused"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_run_stochastic_refinement_pilot",
            "--expert-own-root",
            "x",
            "--expert-own-hash",
            "h",
            "--expected-head",
            "h",
            "--state-id",
            "25",
            "--output",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and not output.exists()
