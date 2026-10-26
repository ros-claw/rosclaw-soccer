import numpy as np
import pytest
from test_shared_cftrack_forecast import setup

from rosclaw_soccer.rsi.cftrack_contact_generator import generate_cftrack_contact_plan
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash


@pytest.mark.parametrize("incomplete", [True, False])
def test_missing_parent_evidence_abstains_without_actual_mutation(
    tmp_path, monkeypatch, incomplete
):
    p, f = setup(tmp_path)

    def forecast(**kwargs):
        assert kwargs == {"controls": 160}
        return dict(
            controls=4 if incomplete else 160,
            body_failed=False,
            runtime_failure=None,
            ball_forces=np.zeros((1600, 1, 3)),
        )

    monkeypatch.setattr(f, "forecast", forecast)
    before = cftrack_state_hash(p.policy)
    plan, report, _ = generate_cftrack_contact_plan(p, f)
    assert plan is None
    assert report["status"] == (
        "ABSTAIN_INCOMPLETE_PARENT" if incomplete else "ABSTAIN_NO_PREDICTED_FOOT_CONTACT"
    )
    assert cftrack_state_hash(p.policy) == before and p.world.native_steps == 0


def test_unowned_forecaster_rejected_before_any_prediction(tmp_path):
    p, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="own pre-proposal"):
        generate_cftrack_contact_plan(p, object())


def test_mutated_actual_history_is_terminal_not_silently_repaired(tmp_path, monkeypatch):
    p, f = setup(tmp_path)

    def forecast(**kwargs):
        p.policy.action[0] = 1
        return dict(controls=4, body_failed=False, runtime_failure=None)

    monkeypatch.setattr(f, "forecast", forecast)
    with pytest.raises(ValueError, match="changed actual policy"):
        generate_cftrack_contact_plan(p, f)
    assert p.failed and p.policy.action[0] == 1 and p.world.native_steps == 0
