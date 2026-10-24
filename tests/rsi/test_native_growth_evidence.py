import copy
import json

import pytest

from rosclaw_soccer.rsi.dual_motor_checkpoint import digest
from rosclaw_soccer.rsi.native_growth_evidence import qualified_native_growth_pair


def fixture(tmp_path):
    flags = dict(hardware_authorized=False, promotion_authorized=False, goal_complete=False)
    kind = "committed-neural-options-v1"
    passed = dict(
        **flags,
        status="TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH",
        training_hash="training",
        producer_hash="pass-source",
        records=[dict(index=i) for i in range(80)],
        summary=dict(controlled=1),
        counts=dict(native_steps=100),
    )
    old_records = [
        dict(context=i // 2, repeat=i % 2, healthy_quality=True, result=dict(body_failed=False))
        for i in range(44)
    ]
    retained = dict(
        **flags,
        status="DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS",
        training_hash="training",
        source_hash="old-source",
        checkpoint_kind=kind,
        checkpoint_hash="checkpoint",
        worlds=44,
        independent_whole_world_replay=True,
        counts=dict(native=100, neural=10),
        results=[
            dict(context=x["context"], repeat=x["repeat"], quality=True, body_failed=False)
            for x in old_records
        ],
    )
    pd = dict(
        **flags,
        protocol="dual256.team_pass_unassisted_TRAIN80.v1",
        training_hash="training",
        checkpoint_kind=kind,
        maximum_worlds=80,
        maximum_native_steps=320000,
        maximum_actor_calls=64000,
        formal_pass_receive_controls=300,
        required_A_neural_recovery_controls=100,
        teacher_enabled=False,
    )
    rd = dict(
        **flags,
        protocol="dual256.unassisted_TRAIN20_and_consumed_run2.v1",
        training_hash="training",
        checkpoint_kind=kind,
        contexts=[{}] * 22,
        repeats=2,
        hidden_width=256,
        maximum_native_steps=176000,
        maximum_actor_decisions=17600,
        online_teacher_enabled=False,
        Fresh_cases=0,
    )
    pins = {}
    for name, declared in (("pass", pd), ("old", rd)):
        root = tmp_path / name
        root.mkdir()
        p = root / "declaration.json"
        p.write_text(json.dumps(declared))
        pins[str(p)] = digest(p)
    source = dict(
        **flags,
        training_hash="training",
        checkpoint_kind=kind,
        records=[dict(**r, artifacts={}) for r in passed["records"]],
        summary=copy.deepcopy(passed["summary"]),
        counts=copy.deepcopy(passed["counts"]),
        declaration_hash=pins[str(tmp_path / "pass/declaration.json")],
    )
    old_source = dict(
        **flags,
        training_hash="training",
        checkpoint_hash="checkpoint",
        records=old_records,
        whole_repeats_exact=True,
        actual_native_steps=100,
        actual_actor_decisions=10,
        declaration_hash=pins[str(tmp_path / "old/declaration.json")],
    )
    trained = dict(**flags, checkpoint_kind=kind, checkpoint_hash="checkpoint")
    docs = {
        "training": (tmp_path / "training/complete.json", trained),
        "pass-source": (tmp_path / "pass/complete.json", source),
        "old-source": (tmp_path / "old/complete.json", old_source),
    }
    return passed, retained, kind, pins, docs


def test_source_own_pair_requires_both_unchanged_physical_protocols(tmp_path):
    p, r, kind, pins, docs = fixture(tmp_path)
    trained, scope = qualified_native_growth_pair(
        p, r, kind=kind, pins=pins, resolve=docs.__getitem__
    )
    assert trained is docs["training"][1] and len(scope["contexts"]) == 22


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "source-result",
        "summary",
        "count",
        "retention-result",
        "own-clock",
        "kind",
        "promotion",
        "horizon",
        "teacher",
    ],
)
def test_incomplete_or_inconsistent_physical_evidence_is_rejected(tmp_path, fault):
    p, r, kind, pins, docs = fixture(tmp_path)
    if fault == "short":
        p["records"].pop()
    elif fault == "source-result":
        docs["pass-source"][1]["records"][0]["index"] = 79
    elif fault == "summary":
        p["summary"]["controlled"] = 80
    elif fault == "count":
        r["counts"]["native"] = 99
    elif fault == "retention-result":
        r["results"][0]["quality"] = False
    elif fault == "own-clock":
        r["independent_whole_world_replay"] = False
    elif fault == "kind":
        r["checkpoint_kind"] = "other"
    elif fault == "promotion":
        docs["training"][1]["promotion_authorized"] = True
    else:
        path = tmp_path / "pass/declaration.json"
        d = json.loads(path.read_text())
        d["formal_pass_receive_controls" if fault == "horizon" else "teacher_enabled"] = (
            360 if fault == "horizon" else True
        )
        path.write_text(json.dumps(d))
        pins[str(path)] = digest(path)
        docs["pass-source"][1]["declaration_hash"] = digest(path)
    with pytest.raises(ValueError):
        qualified_native_growth_pair(p, r, kind=kind, pins=pins, resolve=docs.__getitem__)
