import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_whole_identity_comparison import compare_receiver_traces
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_review_whole_receiver_identity import handoff_equal


def traces():
    n = 3
    base = dict(
        control=np.arange(n),
        obs=np.zeros((n, 98)),
        action=np.zeros((n, 29)),
        command=np.zeros((n, 3)),
        phase=np.zeros(n),
    )
    whole = {k: v.copy() for k, v in base.items()}
    whole.update(
        context14=np.zeros((n, 14)),
        reference_action=np.zeros((n, 29)),
        realized_delta_radians=np.zeros((n, 29)),
        contact_flags_before=np.zeros((n, 2)),
    )
    return base, whole


def test_zero_identity_and_hidden_action_change():
    base, whole = traces()
    result = compare_receiver_traces(base, whole)
    assert result["common_fields_exact"] and result["extra_joint_delta_exact_zero"]
    whole["realized_delta_radians"][1, 4] = 1e-12
    assert not compare_receiver_traces(base, whole)["extra_joint_delta_exact_zero"]
    whole["obs"][1, 0] = 1
    assert not compare_receiver_traces(base, whole)["common_fields_exact"]
    del whole["reference_action"]
    with pytest.raises(ValueError):
        compare_receiver_traces(base, whole)


def test_handoff_comparison_ignores_only_confined_trial_filename(tmp_path):
    for i in (0, 1):
        folder = tmp_path / f"trial{i}"
        folder.mkdir()
        np.savez_compressed(folder / "handoff.npz", state=np.array([1.0]))
    rows = [
        dict(
            index=i,
            handoff=dict(
                control=2,
                boundary=dict(
                    path=f"trial{i}/handoff.npz",
                    hash=digest(tmp_path / f"trial{i}" / "handoff.npz"),
                ),
            ),
        )
        for i in (0, 1)
    ]
    assert handoff_equal(tmp_path, rows[0], tmp_path, rows[1])
    rows[1]["handoff"]["control"] = 3
    assert not handoff_equal(tmp_path, rows[0], tmp_path, rows[1])
