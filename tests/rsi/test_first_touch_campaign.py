import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict

import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_campaign import (
    FirstTouchCampaign,
    load_verified_predecessor,
    summarize_first_touch_candidate,
)
from rosclaw_soccer.rsi.first_touch_campaign_worker import (
    disk_admission,
    prepare,
    propose_next,
    require_same_action_functions,
)
from rosclaw_soccer.sim.contracts import hash_bytes


def declaration():
    return dict(
        mode="SIM_ONLY",
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        max_attempts=3,
        max_generations=4,
        plateau_generations=3,
    )


def receipt(root):
    path = root / "fixture.bin"
    if not path.exists():
        path.write_bytes(b"test-only physical witness")
    return dict(
        physical_substeps=3000,
        foundation_calls=300,
        sampling_draws=270,
        promotion_authorized=False,
        hardware_authorized=False,
        witnesses={
            name: dict(path=str(path), sha256=hash_bytes(path.read_bytes()))
            for name in (
                "trace",
                "report",
                "independent_review",
                "foundation_review",
                "commitment",
                "world",
                "proof_payload",
            )
        },
    )


def test_restart_preserves_completed_jobs_and_charges_interrupted_attempt(tmp_path):
    campaign = FirstTouchCampaign(tmp_path, declaration())
    assert campaign.begin("a") == 0
    campaign.complete("a", receipt(tmp_path))
    campaign.begin("b")
    campaign.db.close()
    recovered = FirstTouchCampaign(tmp_path, declaration())
    recovered.recover_interrupted_attempt()
    assert recovered.state["completed"]["a"] == receipt(tmp_path)
    assert recovered.begin("b") == 2
    recovered.complete("b", receipt(tmp_path))
    with pytest.raises(ValueError, match="budget"):
        recovered.begin("c")
    with pytest.raises(ValueError, match="twice"):
        recovered.begin("a")


def test_contract_change_cannot_resume_under_old_successes(tmp_path):
    FirstTouchCampaign(tmp_path, declaration()).db.close()
    changed = declaration() | dict(max_attempts=99)
    with pytest.raises(ValueError, match="explicit successor"):
        FirstTouchCampaign(tmp_path, changed)


def test_tampered_event_stops_instead_of_repairing_it(tmp_path):
    FirstTouchCampaign(tmp_path, declaration()).db.close()
    with sqlite3.connect(tmp_path / "campaign.sqlite") as db:
        db.execute("UPDATE events SET body='{}'")
    with pytest.raises(ValueError, match="integrity"):
        FirstTouchCampaign(tmp_path, declaration())


@pytest.mark.parametrize("fault", ["short_physics", "bool_count", "hardware", "wrong_job"])
def test_incomplete_or_unbound_receipt_cannot_complete_work(tmp_path, fault):
    campaign = FirstTouchCampaign(tmp_path, declaration())
    campaign.begin("a")
    value = receipt(tmp_path)
    if fault == "short_physics":
        value["physical_substeps"] = 2999
    elif fault == "bool_count":
        value["sampling_draws"] = True
    elif fault == "hardware":
        value["hardware_authorized"] = True
    with pytest.raises(ValueError):
        campaign.complete("b" if fault == "wrong_job" else "a", value)
    assert campaign.state["completed"] == {}


def test_restart_does_not_trust_completed_witness_after_modification(tmp_path):
    campaign = FirstTouchCampaign(tmp_path, declaration())
    campaign.begin("a")
    campaign.complete("a", receipt(tmp_path))
    campaign.db.close()
    (tmp_path / "fixture.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="witness bytes changed"):
        FirstTouchCampaign(tmp_path, declaration())


def test_three_no_gain_generations_trigger_explicit_plateau(tmp_path):
    campaign = FirstTouchCampaign(tmp_path, declaration())
    assert (
        campaign.finish_generation(dict(paired_development_gain=False))
        == "AUTOMATIC_NEXT_GENERATION"
    )
    assert (
        campaign.finish_generation(dict(paired_development_gain=False))
        == "AUTOMATIC_NEXT_GENERATION"
    )
    assert (
        campaign.finish_generation(dict(paired_development_gain=False))
        == "PLATEAU_RESEARCH_REQUIRED"
    )


def test_quality_gain_does_not_overrule_safety_or_retention():
    base = [dict(safe=True, task_pass=False, training_reward=-3.0)] * 4
    outcomes = [dict(safe=True, task_pass=True, training_reward=1.0)] * 3 + [
        dict(safe=False, task_pass=False, training_reward=-10.0)
    ]
    result = summarize_first_touch_candidate(outcomes, base)
    assert result["rescue"] == 3
    assert result["paired_development_gain"] is False
    assert result["teacher_qualified"] is False


def test_actual_process_kill_leaves_recoverable_attempt_not_success(tmp_path):
    code = (
        "import json,os,signal,sys; from pathlib import Path; "
        "from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign; "
        "c=FirstTouchCampaign(Path(sys.argv[1]),json.loads(sys.argv[2])); "
        "c.begin('interrupted'); os.kill(os.getpid(),signal.SIGKILL)"
    )
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), json.dumps(declaration())])
    assert result.returncode < 0
    recovered = FirstTouchCampaign(tmp_path, declaration())
    assert recovered.state["attempts"] == 1
    assert recovered.state["completed"] == {}
    recovered.recover_interrupted_attempt()
    assert recovered.begin("interrupted") == 1


def test_evidence_reuse_does_not_consume_another_physics_attempt(tmp_path):
    campaign = FirstTouchCampaign(tmp_path, declaration())
    campaign.begin("original")
    campaign.complete(
        "original", dict(receipt(tmp_path), evaluation_identity="same-state-and-policy")
    )
    adopted = campaign.reuse("next-generation-elite", "original", "same-state-and-policy")
    assert adopted["reused_from_job"] == "original"
    assert campaign.state["attempts"] == 1
    with pytest.raises(ValueError, match="identity"):
        campaign.reuse("different", "original", "changed-state")


def test_next_generation_is_conditioned_on_measured_failures():
    default = asdict(BoundedFirstTouchControl())
    different = asdict(BoundedFirstTouchControl(closing_gain_s_inv=2.4))
    population = [
        dict(parameters=default, swing="legacy"),
        dict(parameters=different, swing="bounded_contact_diagnostic"),
    ]
    summaries = [
        dict(safety_regression=1, regression=0, high_quality=4, safe_states=3, mean_reward=2.0),
        dict(safety_regression=0, regression=0, high_quality=1, safe_states=4, mean_reward=1.0),
    ]
    proposed = propose_next(population, summaries, seed=42)
    assert proposed[0] == population[1]
    assert proposed == propose_next(population, summaries, seed=42)
    assert len(proposed) == 4
    for candidate in proposed:
        BoundedFirstTouchControl(**candidate["parameters"])


def test_terminal_plateau_cannot_be_unfrozen_by_restart_or_begin(tmp_path):
    campaign = FirstTouchCampaign(tmp_path, declaration())
    for _ in range(3):
        campaign.finish_generation(dict(paired_development_gain=False))
    campaign.db.close()
    restarted = FirstTouchCampaign(tmp_path, declaration())
    with pytest.raises(ValueError, match="terminal"):
        restarted.recover_interrupted_attempt()
    with pytest.raises(ValueError, match="terminal"):
        restarted.begin("cannot-repeat-plateau")


def test_non_sim_configuration_rejected_before_importing_preparation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(mode="REAL", hardware_authorized=True)))
    with pytest.raises(ValueError, match="SIM-only"):
        prepare(path)


def stopped_predecessor(tmp_path, phase="STOPPED_OPERATOR_REQUEST"):
    root = tmp_path / "old"
    old = FirstTouchCampaign(root, declaration())
    old.begin("already-verified")
    old.complete("already-verified", dict(receipt(root), evaluation_identity="same-state"))
    old.state["populations"] = [
        [dict(parameters=asdict(BoundedFirstTouchControl()), swing="legacy")]
    ]
    old.state["phase"] = phase
    old.record("STOPPED_TEST_FIXTURE")
    tip, commitment = old.tip, old.state["declaration_hash"]
    old.db.close()
    path = root / "campaign.sqlite"
    return dict(
        ledger_path=str(path),
        ledger_hash=hash_bytes(path.read_bytes()),
        ledger_tip=tip,
        declaration_hash=commitment,
        evidence_root=str(root),
    )


def test_explicit_successor_adopts_completed_work_and_attempt_budget(tmp_path):
    predecessor = stopped_predecessor(tmp_path)
    successor_declaration = dict(declaration(), predecessor=predecessor, reserved_disk_gib=0)
    campaign = FirstTouchCampaign(tmp_path / "new", successor_declaration)
    campaign.adopt_predecessor()
    assert campaign.state["attempts"] == 1
    assert len(campaign.state["completed"]) == 1
    assert len(campaign.state["populations"]) == 1
    assert campaign.state["adopted_predecessor_tip"] == predecessor["ledger_tip"]
    assert campaign.begin("next") == 1
    campaign.db.close()
    restarted = FirstTouchCampaign(tmp_path / "new", successor_declaration)
    restarted.adopt_predecessor()  # idempotent, never overwrite the next attempt
    assert restarted.state["attempts"] == 2
    assert restarted.state["active"]["job"] == "next"


@pytest.mark.parametrize(
    "phase", ["FATAL_CONTRACT_VIOLATION", "PLATEAU_RESEARCH_REQUIRED", "WAITING_RESOURCES"]
)
def test_successor_cannot_unfreeze_fatal_or_live_predecessor(tmp_path, phase):
    with pytest.raises(ValueError, match="quiescent"):
        load_verified_predecessor(stopped_predecessor(tmp_path, phase))


@pytest.mark.parametrize("fault", ["digest", "tip", "declaration", "witness"])
def test_successor_refuses_changed_predecessor_or_evidence(tmp_path, fault):
    predecessor = stopped_predecessor(tmp_path)
    if fault == "witness":
        (tmp_path / "old/fixture.bin").write_bytes(b"changed")
    else:
        key = {"digest": "ledger_hash", "tip": "ledger_tip", "declaration": "declaration_hash"}[
            fault
        ]
        predecessor[key] = "sha256:" + "0" * 64
    campaign = FirstTouchCampaign(tmp_path / "new", dict(declaration(), predecessor=predecessor))
    with pytest.raises(ValueError):
        campaign.adopt_predecessor()
    assert campaign.state["completed"] == {}


def test_disk_capacity_policy_can_be_explicitly_disabled():
    assert disk_admission(free_bytes=0, reserved_gib=0)
    assert disk_admission(free_bytes=100 * 1024**3, reserved_gib=0)
    assert not disk_admission(free_bytes=100 * 1024**3, reserved_gib=101)
    with pytest.raises(ValueError):
        disk_admission(free_bytes=0, reserved_gib=-1)
    with pytest.raises(ValueError):
        disk_admission(free_bytes=0, reserved_gib=False)


def test_operational_migration_must_not_change_physics_or_search_functions(tmp_path):
    previous, current = tmp_path / "previous.py", tmp_path / "current.py"
    source = "def qualified_episode():\n    return 300\ndef propose_next():\n    return 4\n"
    previous.write_text(source)
    current.write_text("import sqlite3\n" + source)
    require_same_action_functions(previous, current)
    current.write_text(source.replace("return 300", "return 600"))
    with pytest.raises(ValueError, match="arithmetic"):
        require_same_action_functions(previous, current)
