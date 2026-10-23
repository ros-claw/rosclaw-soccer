import pytest

from rosclaw_soccer.rsi.pass_trial_logging import trial_progress


def test_sealed_case_and_outcome_rows_never_enter_coarse_progress():
    r = dict(
        index=3,
        controls=389,
        native_steps=3890,
        actor_calls=778,
        course=dict(target=[9.8, 1.1, 0.11], state_id=33, private_seed=123),
        outcome=dict(private_coordinate=3.14159),
        artifacts=dict(private_path="private"),
        future_nested_private_field={"leak": "no"},
    )
    public = trial_progress(r, include_course=False)
    assert public == dict(index=3, controls=389, native_steps=3890, actor_calls=778)
    assert trial_progress(r, include_course=True) == {
        k: v for k, v in r.items() if k != "artifacts"
    }
    assert "artifacts" in r


@pytest.mark.parametrize("flag", [None, 0, 1, "false"])
def test_ambiguous_disclosure_flag_rejected(flag):
    with pytest.raises(ValueError):
        trial_progress({}, include_course=flag)
