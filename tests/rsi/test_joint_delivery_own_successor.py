import json

import pytest

from rosclaw_soccer.rsi.joint_delivery_teacher import pilot_config
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_run_joint_delivery_own_successor import audit_command


def arguments(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    flags = dict(hardware_authorized=False, promotion_authorized=False, goal_complete=False)
    d = dict(
        **flags,
        config=pilot_config(25),
        input_file_hashes={},
        source_head="producer",
        expert_own_hash="expert",
        expert_own_root=str(tmp_path / "expert"),
        preflight_only=False,
    )
    dp = root / "declaration.json"
    dp.write_text(json.dumps(d))
    r = dict(
        **d,
        status="JOINT_DELIVERY_TEACHER38_COLLECTED_PENDING_ALL_OWN",
        declaration_hash=digest(dp),
        records=[{}] * 38,
        optimizer_updates=0,
        counts=dict(native_steps=150000, actor_calls=30000),
        Fresh_verified=False,
        team_game_verified=False,
    )
    (root / "complete.json").write_text(json.dumps(r))
    return dict(
        source_root=root,
        producer_head="producer",
        expert_own_root=tmp_path / "expert",
        expert_own_hash="expert",
        state_id=25,
        expected_head="auditor",
        output=tmp_path / "unused",
    )


def test_successor_only_runs_the_declared_independent_replay(tmp_path):
    a = arguments(tmp_path)
    c = audit_command(**a)
    assert c[2:4] == ["-m", "scripts.rsi_run_joint_delivery_pilot"]
    assert c[c.index("--producer-head") + 1] == "producer"
    assert c[c.index("--expected-head") + 1] == "auditor"
    assert not any(k in c for k in ("--threshold", "--horizon", "--promote", "--preflight-only"))


@pytest.mark.parametrize(
    "fault",
    ["count", "head", "expert", "config", "budget", "permission", "updates", "failed", "exists"],
)
def test_incomplete_or_changed_pilot_cannot_launch_replay(tmp_path, fault):
    a = arguments(tmp_path)
    p = a["source_root"] / "complete.json"
    r = json.loads(p.read_text())
    if fault == "count":
        r["records"].pop()
    elif fault == "head":
        r["source_head"] = "different"
    elif fault == "expert":
        r["expert_own_hash"] = "different"
    elif fault == "config":
        r["config"]["maximum_worlds"] = 40
    elif fault == "budget":
        r["counts"]["native_steps"] = 152001
    elif fault == "permission":
        r["promotion_authorized"] = True
    elif fault == "updates":
        r["optimizer_updates"] = 1
    elif fault == "failed":
        (a["source_root"] / "failed.json").write_text("{}")
    else:
        a["output"].mkdir()
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        audit_command(**a)
