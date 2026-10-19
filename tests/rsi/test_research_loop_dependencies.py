import copy

import pytest

from rosclaw_soccer.rsi.research_loop_dependencies import annotate_dependencies


def setup():
    campaigns = {
        "collect": dict(phase="RUNNING", service=dict(MainPID="12", LoadState="loaded")),
        "verify": dict(phase="RUNNING", service=dict(MainPID="13", LoadState="loaded")),
    }
    plan = dict(
        schema="soccer.research.observed_pipeline_dependencies.v1",
        dependencies={
            "collect": [],
            "verify": [dict(unit="collect", milestone="terminal_complete")],
        },
    )
    return campaigns, plan


def test_waiting_process_is_not_mislabelled_as_training():
    campaigns, plan = setup()
    before = copy.deepcopy(campaigns)
    output = annotate_dependencies(campaigns, plan)
    assert campaigns == before
    assert output["verify"]["declared_work_role"] == "WAITING_FOR_DECLARED_PREDECESSOR"
    assert output["verify"]["dependency_contents_verified"] is False


def test_complete_file_without_terminal_process_does_not_release_wait():
    campaigns, plan = setup()
    campaigns["collect"]["artifacts"] = dict(complete_present=True)
    assert annotate_dependencies(campaigns, plan)["verify"]["pending_prerequisites"]
    campaigns["collect"]["service"].update(MainPID="0", ActiveState="active", Result="success")
    result = annotate_dependencies(campaigns, plan)
    assert not result["verify"]["pending_prerequisites"]
    assert result["verify"]["dependency_contents_verified"] is False


def test_dependency_failure_demands_review_not_retry():
    campaigns, plan = setup()
    campaigns["collect"]["phase"] = "FAILED"
    result = annotate_dependencies(campaigns, plan)["verify"]
    assert result["declared_work_role"] == "DEPENDENCY_FAILED_NEEDS_AGENT_REVIEW"
    assert result["failed_prerequisites"]


def test_constructor_milestone_allows_completed_factory_with_running_rollouts():
    campaigns, plan = setup()
    plan["dependencies"]["verify"][0]["milestone"] = "factory-contract.json"
    campaigns["collect"]["artifacts"] = dict(checkpoint_names=["factory-contract.json"])
    assert not annotate_dependencies(campaigns, plan)["verify"]["pending_prerequisites"]


@pytest.mark.parametrize("defect", ["cycle", "unknown", "duplicate", "path", "missing"])
def test_ambiguous_or_unbounded_plans_rejected(defect):
    campaigns, plan = setup()
    if defect == "cycle":
        plan["dependencies"]["collect"] = [dict(unit="verify", milestone="terminal_complete")]
    elif defect == "unknown":
        plan["dependencies"]["verify"][0]["unit"] = "missing"
    elif defect == "duplicate":
        plan["dependencies"]["verify"] *= 2
    elif defect == "path":
        plan["dependencies"]["verify"][0]["milestone"] = "../../other"
    else:
        del plan["dependencies"]["collect"]
    with pytest.raises(ValueError):
        annotate_dependencies(campaigns, plan)
