from dataclasses import replace

import pytest
from test_s199_independent_agent_cells import _cells, _observations

from rosclaw_soccer.growth.independent_agent_cell import build_team_coordination_frame
from rosclaw_soccer.growth.pass_readiness import reconcile_pass_readiness
from rosclaw_soccer.growth.role_self_model import SoccerSkill, TacticalIntent, TeamRoleRoster


def fixture():
    cells = _cells()
    obs = _observations(cells, possession_agent_id="red.playmaker")
    decisions = tuple(c.decide(o) for c, o in zip(cells, obs, strict=True))
    return cells, obs, decisions


def test_accepted_frame_is_numerically_identical():
    cells, obs, ds = fixture()
    result = reconcile_pass_readiness(cells=cells, observations=obs, decisions=ds)
    assert all(a is b for a, b in zip(ds, result, strict=True))


def test_recovering_receiver_is_not_forced_to_receive():
    cells, obs, ds = fixture()
    sender = next(d for d in ds if d.intent is TacticalIntent.PASS)
    target = next(i for i, c in enumerate(cells) if c.agent_id == sender.target_agent_id)
    obs = tuple(
        replace(o, self_state=replace(o.self_state, stable=False)) if i == target else o
        for i, o in enumerate(obs)
    )
    ds = tuple(c.decide(o) for c, o in zip(cells, obs, strict=True))
    # Preserve an already proposed pass; do not let the passer infer recovery.
    ds = tuple(sender if d.agent_id == sender.agent_id else d for d in ds)
    result = reconcile_pass_readiness(cells=cells, observations=obs, decisions=ds)
    assert result[target].intent is TacticalIntent.RECOVER
    assert next(d for d in result if d.agent_id == sender.agent_id).intent is TacticalIntent.HOLD
    roster = TeamRoleRoster(match_id="readiness.test", agents=tuple(c.self_model for c in cells))
    frame = build_team_coordination_frame(
        roster=roster, cells=cells, observations=obs, decisions=result, frame_index=0
    )
    assert not frame.pass_receive_handshakes


def test_goalkeeper_save_cannot_be_overridden_by_pass():
    cells, obs, ds = fixture()
    source = next(i for i, c in enumerate(cells) if c.agent_id == "red.playmaker")
    target = next(i for i, c in enumerate(cells) if c.agent_id == "red.goalkeeper")
    altered = list(ds)
    altered[source] = cells[source]._decision(
        obs[source],
        TacticalIntent.PASS,
        SoccerSkill.LEAD_PASS,
        obs[target].self_state.position_m,
        cells[target].agent_id,
        0.9,
    )
    altered[target] = cells[target]._decision(
        obs[target],
        TacticalIntent.SAVE,
        SoccerSkill.SAVE,
        obs[target].self_state.position_m,
        None,
        0.9,
    )
    result = reconcile_pass_readiness(cells=cells, observations=obs, decisions=tuple(altered))
    assert result[source].intent is TacticalIntent.HOLD
    assert result[target] is altered[target]
    reverse = reconcile_pass_readiness(
        cells=tuple(reversed(cells)),
        observations=tuple(reversed(obs)),
        decisions=tuple(reversed(altered)),
    )
    assert {d.agent_id: d.decision_hash for d in result} == {
        d.agent_id: d.decision_hash for d in reverse
    }


@pytest.mark.parametrize("fault", ["duplicate", "binding", "opponent", "missing_receiver"])
def test_bad_contracts_still_fail_closed(fault):
    cells, obs, ds = fixture()
    source = next(i for i, d in enumerate(ds) if d.intent is TacticalIntent.PASS)
    altered = list(ds)
    if fault == "duplicate":
        altered[0] = altered[1]
    elif fault == "binding":
        altered[source] = replace(altered[source], observation_hash="sha256:" + "0" * 64)
    else:
        altered[source] = replace(
            altered[source], target_agent_id="blue.finisher" if fault == "opponent" else None
        )
    with pytest.raises(ValueError):
        reconcile_pass_readiness(cells=cells, observations=obs, decisions=tuple(altered))
