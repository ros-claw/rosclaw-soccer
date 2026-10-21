import copy

import pytest

from rosclaw_soccer.rsi.paid_successor_curriculum import (
    curriculum_cases,
    qualified_additional_teacher,
    validate_train_curriculum,
)


def data_fixture(*, expanded=False):
    cases = [2, 3, 8, 9, 10, 11] if expanded else [2, 3, 10, 11]
    declaration = dict(teacher_cases=cases, DEV_episodes_fit=0)
    if expanded:
        declaration["curriculum_protocol"] = "successor-six-teacher-v2"
    rows = [dict(case=i, skill=2, role="teacher", source_path=f"teacher/{i}") for i in cases]
    rows += [
        dict(case=i, skill=k, role="protected_success", source_path=f"success/{k}/{i}")
        for k, n in enumerate((16, 14, 6))
        for i in range(n)
    ]
    for i, row in enumerate(rows):
        row.update(rows=2, start=i * 2, stop=(i + 1) * 2)
    data = dict(
        trajectories=rows,
        rows=len(rows) * 2,
        teacher_rows=len(cases) * 2,
        protected_rows=72,
        protected_successes_by_kick=[16, 14, 6],
        bank="train",
        DEV_episodes_fit=0,
    )
    return data, declaration


@pytest.mark.parametrize("expanded", [False, True])
def test_four_and_six_teacher_banks_have_distinct_declared_identities(expanded):
    data, declaration = data_fixture(expanded=expanded)
    validate_train_curriculum(data, declaration)
    assert len(curriculum_cases(declaration)) == (6 if expanded else 4)


@pytest.mark.parametrize("fault", ["drop", "role", "case", "DEV", "clock", "protocol"])
def test_training_rejects_missing_successes_or_undeclared_extra_examples(fault):
    data, declaration = data_fixture(expanded=True)
    if fault == "drop":
        data["trajectories"].pop()
    elif fault == "role":
        data["trajectories"][0]["role"] = "protected_success"
    elif fault == "case":
        data["trajectories"][0]["case"] = 16
    elif fault == "DEV":
        data["bank"] = "dev"
    elif fault == "clock":
        data["trajectories"][1]["start"] += 1
    else:
        declaration["curriculum_protocol"] = "successor-four-teacher-v1"
    with pytest.raises(ValueError):
        validate_train_curriculum(data, declaration)


def extension_fixture():
    cases = [dict(source=f"TRAIN{i}") for i in range(16)]
    selected = [
        dict(
            case=i,
            kick=dict(body_failed=False, forbidden=False),
            stand=dict(healthy=True, stability_observed=True),
        )
        for i in (8, 9)
    ]
    report = dict(
        status="PAID_SUCCESSOR_CONTACT_TEACHER_CLOSED_PENDING_AUDIT",
        input_file_hashes={"parent": "p"},
        cem_distribution_updates=2,
        actual_native_steps=171600,
        actual_neural_rows=17160,
        searches=[dict(pair=[8, 9], selected_rank=[1, 1, 1, 2, 2, 10.0], selected=selected)],
        hardware_authorized=False,
        promotion_authorized=False,
    )
    review = dict(
        status="PAID_SUCCESSOR_CONTACT_ALL_CANDIDATES_NATIVE_NEURAL_IK_CEM_AUDIT_PASS",
        input_file_hashes={"teacher": "t"},
        counts=dict(branches=52, own_neural=17160),
        own_native_steps=171600,
        hardware_authorized=False,
        promotion_authorized=False,
        summaries=[
            dict(
                pair=[8, 9],
                selected=[dict(quality=True, stand_healthy=True, stand_stable=True)] * 2,
            )
        ],
    )
    declaration = dict(source_cases=cases, pairs=[[8, 9]], random_seed=20261022)
    args = dict(
        report_path="teacher", report_hash="t", parent_path="parent", parent_hash="p", cases=cases
    )
    return report, review, declaration, args


def test_only_independently_audited_train005_successes_can_extend_dataset():
    report, review, declaration, args = extension_fixture()
    assert [
        r["case"] for r in qualified_additional_teacher(report, review, declaration, **args)
    ] == [8, 9]


@pytest.mark.parametrize("fault", ["hash", "failed", "stand", "seed", "incomplete"])
def test_additional_teacher_cannot_smuggle_failures_or_dev_labels(fault):
    report, review, declaration, args = copy.deepcopy(extension_fixture())
    if fault == "hash":
        review["input_file_hashes"] = {}
    elif fault == "failed":
        report["searches"][0]["selected"][0]["kick"]["forbidden"] = True
    elif fault == "stand":
        review["summaries"][0]["selected"][0]["stand_stable"] = False
    elif fault == "seed":
        declaration["random_seed"] = 20261023
    else:
        review["counts"]["branches"] = 51
    with pytest.raises(ValueError):
        qualified_additional_teacher(report, review, declaration, **args)
