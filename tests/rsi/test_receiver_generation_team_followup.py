from types import SimpleNamespace

import numpy as np

from scripts import rsi_review_receiver_generation_pair as review


def test_followup_only_counts_actual_a_contacts_after_directed_b_strike(monkeypatch, tmp_path):
    f = np.zeros((4, 10, 2, 3))
    q = np.zeros((4, 10, 8))
    v = np.zeros((4, 10, 8))
    f[1, 0, 1, 0] = 11
    v[1, 0, 4] = 1.1
    f[0, 9, 0, 0] = 99  # Before strike must not count.
    f[2, 0, 0, 2] = 2
    f[2, 1, 0, 0] = 11
    path = tmp_path / "physics.npz"
    np.savez(path, ball_forces=f, qpos=q, qvel=v)
    monkeypatch.setattr(review, "bind", lambda *args: path)
    row = {"handoff": {"control": 1, "return_goal": [1, 0], "kick_origin": [0, 0]}}
    model = SimpleNamespace(nq=8, nv=8)
    layout = SimpleNamespace(bq=0, bv=4, rq=2)
    result = review.team_followup(tmp_path, row, 4, model, layout)
    assert result["b_strike_substep"] == 10
    assert result["a_first_nonfoot_substep"] == 20
    assert result["a_first_foot_substep"] == 21
    assert result["teammate_possession_proven"] is False
    limited = review.team_followup(tmp_path, row, 2, model, layout)
    assert limited["a_first_nonfoot_substep"] is None
    assert limited["a_first_foot_substep"] is None


def test_no_handoff_is_not_a_team_pass():
    assert review.team_followup(None, {"handoff": None}, 1, None, None) is None
