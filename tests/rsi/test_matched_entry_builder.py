"""Synthetic candidate orchestration, not physical skill evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rsi_matched_entry_builder import choose_matched_candidate, make_builder


def anchors():
    return [
        dict(law_index=law, offset_root_m=list(offset), score=dict(qualified=False, cost=None))
        for law in (0, 3, 4)
        for offset in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025))
    ]


def pipeline(tmp_path, monkeypatch, kind):
    from rosclaw_soccer.providers.g1 import sonic_matched_entry as planner
    from rosclaw_soccer.providers.g1 import sonic_wrist_quarantine as quarantine
    from scripts import rsi_response_cem_builder as shared

    rows = []
    calls = []
    captured = []
    q = np.zeros((280, 43))
    q[:, 2] = 0.793
    q[:, 3] = 1
    q[:, 7] = 0.123
    v = np.zeros((280, 41))
    v[:, 6] = 0.2
    arrays = dict(canonical_qpos=q, canonical_qvel=v)
    futures = [
        SimpleNamespace(
            arrays=arrays, report=dict(handoff_frame=None if kind == "no_handoff" else 50)
        )
        for _ in range(6)
    ]

    def fake_anchors(predict, buffer, *args):
        buffer.extend(anchors())
        return futures

    monkeypatch.setattr(shared, "certified_anchors", fake_anchors)
    monkeypatch.setattr(quarantine, "quarantine_wrist_reference", lambda source, q: (source, None))

    class Pair:
        entry_intervals = 10
        position_initial_velocity_gap_rad_s = 0.5
        velocity_initial_velocity_gap_rad_s = 0.0

        def report(self, mode):
            return dict(mode=mode, entry_intervals=10, measured_velocity_hash="fixture")

    def plan(source, qpos, qvel, limits, **kwargs):
        captured.append((qpos.copy(), qvel.copy()))
        if kind == "infeasible":
            raise planner.InfeasibleMatchedBridge((dict(entry_intervals=10, reason="fixture"),))
        return Pair()

    monkeypatch.setattr(planner, "prepare_matched_entry_pair", plan)

    def predict(law, offset, proposal=None, *, entry_reference_mode):
        calls.append((law, entry_reference_mode))
        rows.append(
            dict(
                law_index=law,
                offset_root_m=list(offset),
                entry_reference_mode=entry_reference_mode,
                score=dict(qualified=False, cost=None),
            )
        )
        data = {k: a.copy() for k, a in arrays.items()}
        if kind == "changed_prefix":
            data["canonical_qpos"][0, 0] = 1
        return SimpleNamespace(
            arrays=data,
            report=dict(
                handoff_frame=50, entry_reference_adaptation=Pair().report(entry_reference_mode)
            ),
        )

    builder = make_builder(np.zeros((1000, 36)), np.tile([-3.0, 3.0], (29, 1)), "fixture")
    result = builder(predict, rows, {}, None, None, tmp_path)
    return rows, calls, result, captured


def test_all_six_new_predictions_are_same_law_matched_pairs(tmp_path, monkeypatch):
    rows, calls, availability, inputs = pipeline(tmp_path, monkeypatch, "available")
    assert len(rows) == 12 and len(calls) == 6 and len(availability) == 3
    assert calls == [
        (law, mode) for law in (0, 3, 4) for mode in ("matched_position", "matched_velocity")
    ]
    assert all(x[0][7] == 0.123 and x[1][6] == 0.2 for x in inputs)
    assert choose_matched_candidate(rows) == 0
    rows[1]["score"] = dict(qualified=True, cost=0.1)
    rows[7]["score"] = dict(qualified=True, cost=0.05)
    assert choose_matched_candidate(rows) == 7


@pytest.mark.parametrize("kind", ["no_handoff", "infeasible"])
def test_unavailable_candidates_not_fabricated_and_anchors_retained(tmp_path, monkeypatch, kind):
    rows, calls, availability, _ = pipeline(tmp_path, monkeypatch, kind)
    assert len(rows) == 6 and calls == [] and len(availability) == 3
    assert all(r["candidate_indices"] == [] for r in availability)
    if kind == "infeasible":
        assert all(len(r["feasibility_attempts"]) == 1 for r in availability)


def test_before_handoff_counterfactual_must_be_exact(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        pipeline(tmp_path, monkeypatch, "changed_prefix")


def test_incomplete_pair_is_not_a_valid_selection_domain():
    data = anchors() + [
        dict(
            law_index=0,
            offset_root_m=[0.0, 0.0, 0.0],
            entry_reference_mode="matched_position",
            score=dict(qualified=True, cost=0.0),
        )
    ]
    with pytest.raises(ValueError, match="matched pairs"):
        choose_matched_candidate(data)
