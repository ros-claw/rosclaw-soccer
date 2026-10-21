import pytest

from rosclaw_soccer.rsi.paid_student_evidence import qualify_student_worlds


def fixture():
    report = dict(
        status="PAID_SUCCESSOR_NEURAL_STUDENT_LIFECYCLE_CLOSED_PENDING_AUDIT",
        declaration_hash="d",
        source_head="head",
        bank="train",
        worlds=[dict(case=i) for i in range(16)],
        summary=dict(stable=0),
        all_first_and_full_repeat_arrays_exact=True,
        actual_native_steps=1000,
        actual_neural_rows=100,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration = dict(
        source_head="head",
        bank="train",
        cases=[{}] * 16,
        skills=3,
        repeats=2,
        body_resets=0,
        standing_controls=100,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    audit = dict(
        status="PAID_SUCCESSOR_STUDENT_ALL_NATIVE_PRIMITIVES_AND_OWN_NEURAL_AUDIT_PASS",
        input_file_hashes={"report": "r"},
        summary=dict(stable=0),
        counts=dict(worlds=32, native=1000, neural=100, own_neural=100),
        own_native_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    args = dict(report_path="report", report_hash="r", declaration_hash="d")
    return report, declaration, audit, args


def test_even_failed_physical_outcomes_are_valid_for_exact_replay_not_promotion():
    r, d, a, args = fixture()
    qualify_student_worlds(r, d, a, **args)
    assert r["summary"]["stable"] == 0 and not r["promotion_authorized"]


@pytest.mark.parametrize("fault", ["hash", "declaration", "rows", "bank", "reset", "hardware"])
def test_replay_admission_rejects_unbound_or_incomplete_worlds(fault):
    r, d, a, args = fixture()
    if fault == "hash":
        a["input_file_hashes"] = {}
    elif fault == "declaration":
        r["declaration_hash"] = "changed"
    elif fault == "rows":
        a["counts"]["own_neural"] = 99
    elif fault == "bank":
        d["bank"] = "dev"
    elif fault == "reset":
        d["body_resets"] = 1
    else:
        d["hardware_authorized"] = True
    with pytest.raises(ValueError):
        qualify_student_worlds(r, d, a, **args)
