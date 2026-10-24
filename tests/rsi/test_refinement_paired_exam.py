import copy

import pytest

from rosclaw_soccer.rsi.refinement_paired_exam import exam_config, verify_exam


def fixture():
    courses = [dict(assisted=False, state_id=25, offset=x) for x in (0, -0.005, 0.005)]
    cfg = exam_config(25)
    rows = [
        dict(
            index=i,
            arm=cfg["arms"][i % 4],
            course=courses[i // 4],
            controls=389,
            native_steps=3890,
            actor_calls=778,
            stand_controls=100,
            termination="FULL_MOTION_PLUS100NN_RECOVERY",
            neural_actor_optimizer_updates=0,
            outcome=dict(strict=False, error=0.3 if i % 2 else 0.6),
        )
        for i in range(12)
    ]
    return dict(
        config=cfg, records=rows, counts=dict(native_steps=46680, actor_calls=9336)
    ), courses


def test_exact_complete_finite_paired_exam():
    report, courses = fixture()
    verify_exam(report, courses)


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm", "old"),
        ("controls", 401),
        ("native_steps", 4000),
        ("actor_calls", 1),
        ("stand_controls", 99),
        ("neural_actor_optimizer_updates", 1),
        ("termination", "SKIPPED"),
        ("sampling_seed", 1),
        ("feedback_variant", 0),
        ("student_prefix_controls", 0),
    ],
)
def test_incorrect_provenance_or_clocks_rejected(field, value):
    report, courses = fixture()
    report["records"][1][field] = value
    with pytest.raises(ValueError):
        verify_exam(report, courses)


def test_repeat_mismatch_or_missing_world_rejected():
    report, courses = fixture()
    report["records"][3]["outcome"]["error"] = 0.2
    with pytest.raises(ValueError, match="repeats"):
        verify_exam(report, courses)
    report, courses = fixture()
    report["records"].pop()
    with pytest.raises(ValueError):
        verify_exam(report, courses)


def test_actual_failure_terminal_not_padded_or_discarded():
    report, courses = fixture()
    for i in (1, 3):
        report["records"][i].update(
            termination="BODY_FAILURE",
            controls=12,
            stand_controls=0,
            native_steps=120,
            actor_calls=24,
        )
    report["counts"] = dict(native_steps=39140, actor_calls=7828)
    verify_exam(report, courses)
    wrong = copy.deepcopy(report)
    wrong["counts"]["native_steps"] += 1
    with pytest.raises(ValueError):
        verify_exam(wrong, courses)


@pytest.mark.parametrize("state", [True, 24, 27, 25.0])
def test_other_unregistered_state_rejected(state):
    with pytest.raises(ValueError):
        exam_config(state)
