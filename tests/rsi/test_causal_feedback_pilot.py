import copy
import subprocess
import sys

import pytest
from test_joint_delivery_teacher import row

from rosclaw_soccer.rsi.causal_feedback_pilot import (
    pilot_config,
    selection,
    teacher_qualified,
    verify_study,
)


def fixture():
    baseline = row(0.03, 3.6)
    baseline["course"] = dict(
        state_id=26, task=0, assisted=False, ball_planar_offset_world=[0.0, 0.0]
    )
    variants = [
        baseline["course"] | dict(ball_planar_offset_world=[x, 0.0]) for x in (-0.005, 0.005)
    ]
    records = []
    for i in range(6):
        o = row(0.05, 2.9 if i // 2 == 1 else 3.2)
        records.append(
            o
            | dict(
                index=i,
                course=copy.deepcopy(baseline["course"]),
                feedback_variant=1 + i // 2,
                prefix_identity_fields=1,
                neural_actor_optimizer_updates=0,
            )
        )
    selected = selection(baseline, records)
    for i in range(6, 10):
        records.append(
            row(0.06, 2.8)
            | dict(
                index=i,
                course=copy.deepcopy(variants[(i - 6) // 2]),
                feedback_variant=selected["variant"],
                prefix_identity_fields=None,
                neural_actor_optimizer_updates=0,
            )
        )
    return dict(config=pilot_config(26), records=records, selected=selected), baseline, variants


def test_whole_fixed_ten_world_study_and_unchanged_thresholds():
    report, baseline, variants = fixture()
    verify_study(report, baseline, variants)
    assert report["selected"]["variant"] == 2
    assert teacher_qualified(report)
    for s in (25, 26):
        cfg = pilot_config(s)
        assert cfg["maximum_worlds"] == 10
        assert cfg["maximum_native_steps"] == 40000
        assert cfg["new_optimizer_updates"] == cfg["Fresh_cases_read"] == 0


@pytest.mark.parametrize(
    "fault", ["missing", "variant", "course", "repeat", "index", "prefix", "winner", "optimizer"]
)
def test_no_dropped_rows_mixed_courses_or_unrepeated_winner(fault):
    report, baseline, variants = fixture()
    if fault == "missing":
        report["records"].pop()
    elif fault == "variant":
        report["records"][6]["feedback_variant"] = 3
    elif fault == "course":
        report["records"][8]["course"] = variants[0]
    elif fault == "repeat":
        report["records"][3]["outcome"]["incoming_arrival_speed_mps"] += 0.01
    elif fault == "index":
        report["records"][9]["index"] = 0
    elif fault == "prefix":
        report["records"][0]["prefix_identity_fields"] = 0
    elif fault == "winner":
        report["selected"]["variant"] = 1
    else:
        report["records"][0]["neural_actor_optimizer_updates"] = 1
    with pytest.raises(ValueError):
        verify_study(report, baseline, variants)


def test_one_validation_failure_forbids_teacher_qualification():
    report, _, _ = fixture()
    report["records"][8]["outcome"] = row(0.2, 2.8)["outcome"]
    assert not teacher_qualified(report)


@pytest.mark.parametrize(
    "extra",
    [["--source-root", "x"], ["--source-root", "x", "--source-hash", "h", "--preflight-only"]],
)
def test_bad_audit_cli_cannot_create_output(tmp_path, extra):
    output = tmp_path / "unused"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_run_causal_feedback_pilot",
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
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2 and not output.exists()
