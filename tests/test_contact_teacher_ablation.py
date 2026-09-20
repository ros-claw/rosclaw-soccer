import pytest

from rosclaw_soccer.training.contact_teacher_ablation import ContactTeacherSuppression


def test_one_player_removal_only_at_declared_boundary():
    contract = ContactTeacherSuppression("blue.playmaker", 30)
    assert contract.suppressed_agent(29) is None
    assert contract.suppressed_agent(30) == "blue.playmaker"
    assert contract.suppressed_agent(31) == "blue.playmaker"
    assert contract.contract_hash != ContactTeacherSuppression("blue.playmaker", 31).contract_hash
    assert contract.contract_hash != ContactTeacherSuppression("red.playmaker", 30).contract_hash


@pytest.mark.parametrize(
    "identity,start",
    [
        ("", 0),
        ("Bad identity", 0),
        ("blue.playmaker", -1),
        ("blue.playmaker", True),
        ("blue.playmaker", 1001),
    ],
)
def test_bad_contract_rejected(identity, start):
    with pytest.raises(ValueError):
        ContactTeacherSuppression(identity, start)


@pytest.mark.parametrize("frame", [-1, True, 0.5])
def test_bad_frame_rejected(frame):
    with pytest.raises(ValueError):
        ContactTeacherSuppression("blue.playmaker", 0).suppressed_agent(frame)
