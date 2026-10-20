import copy
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
from rosclaw_soccer.rsi.bilateral_entry import (
    choose_bilateral_candidate,
    contact_foot_context,
    observed_contact_foot,
)


def prior():
    return SkillEntryAffordance((0.21, 0.034), 150, 1.7, "sha256:" + "a" * 64)


def test_context_keeps_default_identity_and_mirrors_only_reference():
    source = np.zeros((993, 36), dtype=np.float64)
    source[:, 2:4] = [0.8, 1]
    source[:, 7] = 0.1
    affordance = prior()
    right, entry, report = contact_foot_context(source, affordance, "right")
    assert right is source and entry is affordance and report is None
    saved = source.copy()
    left, entry, report = contact_foot_context(source, affordance, "left")
    np.testing.assert_array_equal(source, saved)
    assert entry.ball_offset_xy_m == (0.21, -0.034)
    assert entry.contract_hash != affordance.contract_hash
    assert report["mirrored_prior_is_executed_left_motion"] is False
    np.testing.assert_array_equal(left[:, 13], source[:, 7])
    with pytest.raises(ValueError, match="exactly once"):
        contact_foot_context(left, entry, "left")


@pytest.mark.parametrize("foot", [None, True, "LEFT", "both"])
def test_bad_foot_before_context_access(foot):
    with pytest.raises(ValueError, match="explicit"):
        contact_foot_context(None, None, foot)


def rows():
    data = [
        dict(law_index=law, offset_root_m=list(offset), score=dict(qualified=False, cost=None))
        for law in (0, 3, 4)
        for offset in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025))
    ]
    return data + [
        dict(
            law_index=law,
            offset_root_m=[0.0] * 3,
            contact_reference_foot="left",
            score=dict(qualified=False, cost=None),
        )
        for law in (0, 3, 4)
    ]


def test_fixed_nine_options_and_same_cost_no_mutation():
    data = rows()
    previous = copy.deepcopy(data)
    assert choose_bilateral_candidate(data) == 0
    assert data == previous
    data[3]["score"] = dict(qualified=True, cost=0.2)
    data[7]["score"] = dict(qualified=True, cost=0.1)
    assert choose_bilateral_candidate(data) == 7
    data[2]["score"] = dict(qualified=True, cost=0.1)
    assert choose_bilateral_candidate(data) == 2


@pytest.mark.parametrize("kind", ["short", "order", "offset", "foot", "mode", "nan", "bool"])
def test_invalid_candidate_contract(kind):
    data = rows()
    if kind == "short":
        data.pop()
    elif kind == "order":
        data[8]["law_index"] = 0
    elif kind == "offset":
        data[6]["offset_root_m"][1] = -0.025
    elif kind == "foot":
        data[0]["contact_reference_foot"] = "left"
    elif kind == "mode":
        data[7]["entry_reference_mode"] = "matched_velocity"
    else:
        data[0]["score"] = dict(qualified=True, cost=float("nan") if kind == "nan" else True)
    with pytest.raises(ValueError):
        choose_bilateral_candidate(data)


def test_observed_foot_can_disagree_with_intended_and_simultaneous():
    names = [s + t for s in ("left", "right") for t in ("_ankle_roll_link", "_ankle_pitch_link")]
    model = SimpleNamespace(nbody=5, body=lambda name: SimpleNamespace(id=names.index(name) + 1))
    f = np.zeros((280, 10, 5))
    assert (
        observed_contact_foot(model, dict(ball_contact_force_by_body_n=f), global_start_frame=20)[
            "first_contact_foot_sides"
        ]
        == []
    )
    f[30, 2, 3] = 2
    info = observed_contact_foot(model, dict(ball_contact_force_by_body_n=f), global_start_frame=20)
    assert info["first_contact_foot_sides"] == ["right"]
    assert info["first_contact_global_frame"] == 50 and info["first_contact_native_offset"] == 302
    f[30, 2, 1] = 2
    assert observed_contact_foot(
        model, dict(ball_contact_force_by_body_n=f), global_start_frame=20
    )["first_contact_foot_sides"] == ["left", "right"]


def test_builder_keeps_all_anchors_and_runs_all_left_laws(monkeypatch, tmp_path):
    from scripts import rsi_response_cem_builder as shared
    from scripts.rsi_bilateral_entry_builder import build_candidates

    calls = []

    def anchors(*args):
        calls.append("anchors")

    monkeypatch.setattr(shared, "certified_anchors", anchors)

    def predict(law, offset, proposal, **kw):
        calls.append((law, offset, kw))

    result = build_candidates(predict, [], {}, None, None, tmp_path)
    assert calls[0] == "anchors"
    assert [c[0] for c in calls[1:]] == [0, 3, 4]
    assert all(c[1:] == ((0.0, 0.0, 0.0), dict(contact_reference_foot="left")) for c in calls[1:])
    assert len(result) == 3
