"""Thirty-candidate synthetic orchestration; never physical teacher evidence."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_response_cem_builder import build_candidates
from tests.rsi.test_contact_response_cem import score


def exercise(tmp_path, *, prefix_value=0.0, old_label=False, corrupt=False):
    archive = tmp_path / "old"
    (archive / "variant0").mkdir(parents=True)
    folder = tmp_path / "output"
    folder.mkdir()
    anchors = []

    def arrays(law, offset):
        return dict(synthetic_trace=np.array([[float(law), *offset]]))

    for law in (0, 3, 4):
        for offset in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025)):
            p = archive / f"variant0/candidate{len(anchors)}.npz"
            np.savez_compressed(p, **arrays(law, offset))
            anchors.append(
                dict(
                    artifact=p.name,
                    artifact_hash=digest(p),
                    law=dict(feedback_index=law),
                    contact_reference_offset_m=list(offset),
                    score=dict(qualified=old_label),
                )
            )
    source = (archive, dict(results=[dict(results=[{}, dict(candidates=anchors)])]))
    rows, calls = [], []

    def predict(law, offset, proposal=None):
        calls.append((law, offset))
        s = score(forward_to_task_end_m=0.3 + offset[0], lateral_to_task_end_m=offset[1])
        rows.append(dict(law_index=law, offset_root_m=list(offset), score=s))
        data = arrays(law, offset)
        if corrupt:
            data["synthetic_trace"] += 1
        return SimpleNamespace(arrays=data)

    result = build_candidates(
        predict,
        rows,
        dict(measured_prefix=np.full((20, 2), prefix_value)),
        source,
        SimpleNamespace(variant_index=0),
        folder,
    )
    return rows, calls, result


def test_exact_budget_updates_and_historical_success_not_a_generator_input(tmp_path):
    rows, calls, rounds = exercise(tmp_path / "a", old_label=False)
    _, again, _ = exercise(tmp_path / "b", old_label=True)
    assert calls == again and len(rows) == 30 and len(rounds) == 6
    assert [law for law, _ in calls[:6]] == [0, 0, 3, 3, 4, 4]
    assert [r["law_index"] for r in rounds] == [0, 0, 3, 3, 4, 4]
    for r in rounds:
        assert len(r["samples"]) == 4 and len(r["elite_indices"]) == 2
        assert r["anchors_all18_bit_exact"]
        p = tmp_path / f"a/output/cem-law{r['law_index']}-round{r['iteration']}-complete.json"
        assert json.loads(p.read_text()) == r


def test_seed_changes_with_own_measured_prefix_not_archive_path(tmp_path):
    _, a, first = exercise(tmp_path / "a", prefix_value=0)
    _, b, second = exercise(tmp_path / "b", prefix_value=1)
    assert a[:6] == b[:6] and a[6:] != b[6:]
    assert first[0]["prefix_content_hash"] != second[0]["prefix_content_hash"]


def test_wrong_same_state_anchor_stops_before_any_cem_sampling(tmp_path):
    with pytest.raises(AssertionError):
        exercise(tmp_path, corrupt=True)


def test_final_choice_retains_successful_anchors_uses_original_cost_not_merit(tmp_path):
    from rosclaw_soccer.rsi.contact_response_cem import choose_response_candidate

    rows, _, _ = exercise(tmp_path)
    assert choose_response_candidate(rows) == 0
    rows[1]["score"] = score(qualified=True, cost=0.001)
    rows[6]["score"] = score(qualified=True, cost=0.01)
    assert choose_response_candidate(rows) == 1
    rows[6]["score"]["cost"] = 0.0
    assert choose_response_candidate(rows) == 6
    rows[0]["offset_root_m"] = [0.0, 0.0, 0.001]
    with pytest.raises(ValueError, match="anchors must remain"):
        choose_response_candidate(rows)
