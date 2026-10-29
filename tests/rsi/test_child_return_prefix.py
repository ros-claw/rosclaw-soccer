import numpy as np
import pytest

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_review_child_return_role import assert_causal_prefix


def test_receiver_successor_uses_absolute_control_clock(tmp_path):
    rows = []
    for index in (0, 1):
        folder = tmp_path / f"trial{index}"
        folder.mkdir()
        artifacts = {}
        for name in (
            "physics",
            "passer_nn",
            "receiver_velocity",
            "receiver_shadow",
            "receiver_return_nn",
        ):
            path = folder / f"{name}.npz"
            if name == "receiver_return_nn":
                control = np.arange(8, 13)
                values = np.array([1, 1, 1, 1 + index, 1 + index])
            else:
                control = np.arange(13)
                values = np.ones(13)
            np.savez(path, control=control, value=values)
            artifacts[name] = dict(path=f"trial{index}/{name}.npz", hash=digest(path))
        rows.append(
            dict(
                index=index,
                controls=13,
                artifacts=artifacts,
                passer_velocity_recovery_boundary=None if index == 0 else dict(control=11),
            )
        )
    assert assert_causal_prefix(tmp_path, *rows) == 11
    rows[1]["passer_velocity_recovery_boundary"]["control"] = 12
    with pytest.raises(ValueError, match="before role switch"):
        assert_causal_prefix(tmp_path, *rows)
