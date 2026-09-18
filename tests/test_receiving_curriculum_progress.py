import pytest

from rosclaw_soccer.training.receiving_curriculum_progress import receiving_curriculum_progress
from rosclaw_soccer.training.role_receiving_courses import ROSTER

HASH = "sha256:" + "a" * 64


def rows():
    return [
        dict(
            course_id=f"{a}-{speed}-{i}",
            agent_id=a,
            speed_mps=speed,
            policy_hash=HASH,
            safe=True,
            controlled_reception=True,
        )
        for a in ROSTER
        for speed in (0.75, 1.25)
        for i in range(4)
    ]


def report(data):
    return receiving_curriculum_progress(data, policy_hash=HASH, speeds_mps=(0.75, 1.25))


def test_all_roles_and_speeds_required():
    assert report(rows())["ready_for_next_curriculum"]
    assert not report([])["ready_for_next_curriculum"]
    assert not report([r for r in rows() if r["agent_id"] != "blue.goalkeeper"])[
        "ready_for_next_curriculum"
    ]


def test_high_aggregate_cannot_hide_one_failed_role_or_unsafe_capture():
    data = rows()
    data[0]["controlled_reception"] = False
    assert not report(data)["ready_for_next_curriculum"]
    data[0].update(controlled_reception=True, safe=False)
    assert not report(data)["ready_for_next_curriculum"]


def test_duplicate_and_mixed_policy_fail_closed():
    with pytest.raises(ValueError):
        report(rows() + [rows()[0]])
    data = rows()
    data[0]["policy_hash"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError):
        report(data)


@pytest.mark.parametrize("value", [1, "true", None])
def test_boolean_evidence_must_be_explicit(value):
    data = rows()
    data[0]["safe"] = value
    with pytest.raises(ValueError):
        report(data)


def test_cannot_silently_relax_graduation_threshold():
    with pytest.raises(ValueError):
        receiving_curriculum_progress(
            rows(), policy_hash=HASH, speeds_mps=(0.75, 1.25), target_fraction=0.5
        )


@pytest.mark.parametrize("field,value", [("agent_id", []), ("speed_mps", []), ("speed_mps", True)])
def test_malformed_identity_is_rejected(field, value):
    data = rows()
    data[0][field] = value
    with pytest.raises(ValueError):
        report(data)


def test_summary_never_authorizes_promotion():
    result = report(rows())
    assert result["ready_for_next_curriculum"]
    assert not result["promotion_eligible"]
    assert result["activation_ceiling"] == "SIM_ONLY"
