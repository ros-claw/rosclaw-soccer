import pytest

from rosclaw_soccer.rsi.research_loop_dispatch import dispatch_state


def test_all_closed_requires_successor_not_running_claim():
    result = dispatch_state({"a": {"phase": "STAGE_ARTIFACT_READY_FOR_READBACK"}})
    assert result["no_live_declared_work"]
    assert result["state"] == "AGENT_SUCCESSOR_REQUIRED"
    assert result["agent_attention_required"]
    assert not result["agent_wakeup_provided"]
    assert not result["goal_complete"]


@pytest.mark.parametrize("phase", ["FAILED", "NO_OBSERVED_CPU_PROGRESS", None])
def test_failure_not_hidden_by_another_live_job(phase):
    result = dispatch_state({"a": {"phase": phase}, "b": {"phase": "RUNNING"}})
    assert result["state"] == "AGENT_REVIEW_REQUIRED"
    assert result["live_units"] == ["b"]
    assert not result["physics_retry_authorized"]


def test_live_worker_is_not_goal_success():
    result = dispatch_state({"a": {"phase": "RUNNING"}})
    assert result["state"] == "DECLARED_WORK_LIVE"
    assert not result["no_live_declared_work"]
    assert not result["scientific_successor_automated"]


def test_empty_declared_queue_requires_successor():
    result = dispatch_state({})
    assert result["state"] == "AGENT_SUCCESSOR_REQUIRED"
    assert result["no_live_declared_work"]
    assert result["agent_attention_required"]
    assert not result["agent_wakeup_provided"]
    assert not result["scientific_successor_automated"]


@pytest.mark.parametrize("campaigns", [None, [], {str(i): {} for i in range(129)}])
def test_bounded_queue(campaigns):
    with pytest.raises(ValueError):
        dispatch_state(campaigns)
