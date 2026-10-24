from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.pass_teacher_coverage import teacher_coverage
from rosclaw_soccer.rsi.pass_teacher_transfer_plan import EXPANDED_STATES


def outcome(controlled, strict=False):
    return dict(
        whole_body_healthy=True,
        passer_full100NNrecovery=True,
        passer_clean_foot=True,
        B_excluded_contact=False,
        B_legal_functional_receive_before_deadline=True,
        combined_precision_and_functional_receive=strict,
        incoming=dict(
            A_first_foot_strike_substep=900,
            B_first_ball_contact_substep=1100 if controlled else None,
        ),
    )


def fixture():
    baseline, selected, transfers = [], [], []
    for i, (s, t) in enumerate((s, t) for s in range(20, 30) for t in (0, 1)):
        baseline.append(dict(state_id=s, task=t, outcome=outcome(i < 6)))
        row = dict(state_id=s, task=t, outcome=outcome(i < 8), transfer_selected=t == 1)
        selected.append(row)
        if s in EXPANDED_STATES:
            transfers.append(deepcopy(row))
    return baseline, selected, transfers


def test_qualifies_only_functional_teacher_not_precision_or_policy():
    r = teacher_coverage(*fixture())
    assert r["baseline_controlled"] == 6 and r["selected_controlled"] == 8
    assert r["net_controlled_change"] == 2
    assert r["functional_correction_teacher_qualified"] is True
    assert r["strict_composite"] == 0
    assert r["learned_policy_improvement_verified"] is False
    assert r["correction_bank_ready"] is False


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "regression", "unsafe", "fallback", "fake_precision"]
)
def test_invalid_or_insufficient_coverage_never_qualifies(defect):
    b, s, t = fixture()
    if defect == "missing":
        s.pop()
    elif defect == "duplicate":
        s[-1] = deepcopy(s[0])
    elif defect == "regression":
        s[0]["outcome"] = outcome(False)
    elif defect == "unsafe":
        s[-1]["outcome"]["passer_clean_foot"] = False
    elif defect == "fallback":
        s[2]["transfer_selected"] = 1
    else:
        s[0]["outcome"]["combined_precision_and_functional_receive"] = 1
    if defect in ("regression", "unsafe"):
        assert teacher_coverage(b, s, t)["functional_correction_teacher_qualified"] is False
    else:
        with pytest.raises(ValueError):
            teacher_coverage(b, s, t)


def test_near_target_without_actual_contact_not_full_chain():
    b, s, t = fixture()
    s[-1]["outcome"]["combined_precision_and_functional_receive"] = True
    r = teacher_coverage(b, s, t)
    assert r["strict_composite"] == 1 and r["actual_contact_and_strict"] == 0
