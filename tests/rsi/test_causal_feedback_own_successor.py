import json

import pytest

from rosclaw_soccer.rsi.causal_feedback_pilot import pilot_config
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_run_causal_feedback_own_successor import resolve_source


def source(tmp_path):
    cfg = pilot_config(25)
    declaration = dict(
        source_head="head",
        config=cfg,
        expert_own_hash="old-own",
        expert_own_root="old-root",
        preflight_only=False,
        source_hash=None,
        input_file_hashes={},
    )
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration))
    complete = dict(
        status="CAUSAL_FOOT_BALL_TRAIN10_COLLECTED_PENDING_OWN",
        source_head="head",
        config=cfg,
        expert_own_hash="old-own",
        declaration_hash=digest(path),
        input_file_hashes={},
        records=[{} for _ in range(10)],
        counts=dict(native_steps=40000, actor_calls=8000),
        optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
        Fresh_verified=False,
        team_game_verified=False,
    )
    return complete


def resolve(root):
    from pathlib import Path

    return resolve_source(
        root, producer_head="head", expert_root=Path("old-root"), expert_hash="old-own", state=25
    )


def test_closed_source_binding_is_not_a_physics_qualification(tmp_path):
    complete = source(tmp_path)
    p = tmp_path / "complete.json"
    p.write_text(json.dumps(complete))
    assert resolve(tmp_path) == digest(p)


@pytest.mark.parametrize(
    "fault",
    ["failed", "partial", "head", "permission", "teacher", "counts", "declaration", "missing"],
)
def test_no_successor_on_unclosed_or_conflicting_study(tmp_path, fault):
    complete = source(tmp_path)
    if fault == "failed":
        (tmp_path / "failed.json").write_text("{}")
    elif fault == "partial":
        complete["records"].pop()
    elif fault == "head":
        complete["source_head"] = "other"
    elif fault == "permission":
        complete["hardware_authorized"] = True
    elif fault == "teacher":
        complete["teacher_qualified"] = True
    elif fault == "counts":
        complete["counts"]["native_steps"] = 40001
    elif fault == "declaration":
        complete["declaration_hash"] = "other"
    if fault != "missing":
        (tmp_path / "complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError):
        resolve(tmp_path)
