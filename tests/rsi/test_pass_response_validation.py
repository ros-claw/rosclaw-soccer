import copy

import pytest

from rosclaw_soccer.rsi.pass_response_validation import response_validation


def proposals():
    return [
        dict(
            model_candidate_index=i,
            predicted_response=[0.0, 0.0, 0.0, 0.0],
            actual_response=[0.2, 0.0, 0.0, 0.4],
        )
        for i in range(6)
    ]


def test_prediction_not_physical_success():
    result = response_validation(anchor=[0.1, 0.0, 0.0, 0.0], proposals=proposals())
    assert result["predicted_improvements"] == 6
    assert result["actual_improvements"] == 0
    assert result["predicted_improvement_but_actual_not"] == 6
    assert result["coordinate_rmse"] == pytest.approx([0.2, 0.0, 0.0, 0.4])
    assert result["student_improvement_verified"] is False


def test_missing_not_zero_or_improvement():
    data = proposals()
    data[0]["actual_response"] = None
    result = response_validation(anchor=[0.1, 0.0, 0.0, 0.0], proposals=data)
    assert result["observed"] == 5 and result["missing"] == 1
    assert result["rows"][0]["actual_cost"] is None
    assert result["rows"][0]["actual_improvement"] is None


def test_slow_ball_is_not_penalized_for_not_being_accelerated():
    data = proposals()
    for row in data:
        row["actual_response"] = [0.0, 0.0, 0.0, -1.0]
    old = copy.deepcopy(data)
    result = response_validation(anchor=[0.1, 0.0, 0.0, 0.0], proposals=data)
    assert result["actual_improvements"] == 6
    assert result["rows"][0]["actual_cost"] == 0
    assert old == data


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "bool", "shape"])
def test_invalid_declared_data_rejected(defect):
    data = proposals()
    if defect == "missing":
        data.pop()
    elif defect == "duplicate":
        data[-1]["model_candidate_index"] = 0
    elif defect == "nan":
        data[0]["actual_response"][0] = float("nan")
    elif defect == "bool":
        data[0]["actual_response"][0] = True
    else:
        data[0]["predicted_response"].pop()
    with pytest.raises(ValueError):
        response_validation(anchor=[0.0, 0.0, 0.0, 0.0], proposals=data)
