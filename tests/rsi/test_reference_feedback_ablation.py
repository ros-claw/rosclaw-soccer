"""Fixed numerical diagnostic gate; no physical or learning successes claimed."""

import ast
import copy
import inspect

import pytest

from scripts import rsi_fit_reference_feedback_ablation as stage


def diagnosed():
    rows = []
    for i in range(16):
        rows.append(
            dict(
                eligible_for_original_fit=i in (1, 3, 9, 15),
                partition="validation" if i >= 12 else "train",
                original_fit_raw_mse_reproduced=True,
                measured_body_navigation_and_force_inputs_bit_exact_between_paths=True,
                errors={
                    k: dict(mean_squared_reference_error_rad2=v)
                    for k, v in (
                        ("teacher_forced_projected", 0.001),
                        ("teacher_forced_raw", 0.001001),
                        ("self_fed_projected", 0.002),
                    )
                },
            )
        )
    return dict(rows=rows)


def test_whole_mechanism_gate_does_not_mutate_or_allocate():
    evidence = diagnosed()
    before = copy.deepcopy(evidence)
    assert stage.require_mechanism(evidence) is None
    assert evidence == before


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_failure",
        "eligibility",
        "split",
        "raw_parity",
        "same_body",
        "small_feedback",
        "projection",
        "nan",
        "zero",
    ],
)
def test_ablation_cannot_be_justified_by_selective_or_wrong_mechanism(tamper):
    v = diagnosed()
    r = v["rows"][1]
    if tamper == "missing_failure":
        v["rows"].pop(0)
    elif tamper == "eligibility":
        v["rows"][0]["eligible_for_original_fit"] = True
    elif tamper == "split":
        r["partition"] = "validation"
    elif tamper == "raw_parity":
        v["rows"][0]["original_fit_raw_mse_reproduced"] = False
    elif tamper == "same_body":
        r["measured_body_navigation_and_force_inputs_bit_exact_between_paths"] = False
    elif tamper == "small_feedback":
        r["errors"]["self_fed_projected"]["mean_squared_reference_error_rad2"] = 0.0012
    elif tamper == "projection":
        r["errors"]["teacher_forced_raw"]["mean_squared_reference_error_rad2"] = 0.002
    elif tamper == "nan":
        r["errors"]["self_fed_projected"]["mean_squared_reference_error_rad2"] = float("nan")
    else:
        r["errors"]["teacher_forced_projected"]["mean_squared_reference_error_rad2"] = 0
    with pytest.raises(ValueError):
        stage.require_mechanism(v)


def test_runner_has_no_physical_step_state_or_foundation_calls():
    tree = ast.parse(inspect.getsource(stage))
    names = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not names.intersection({"MjData", "mj_step", "mj_forward", "qualified_episode"})
    assert "from_binary_path" in names  # MJB is static joint-limit evidence only.
