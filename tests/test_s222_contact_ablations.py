from pathlib import Path

import pytest

from rosclaw_soccer.training.active_team_probe import run_probe


@pytest.mark.parametrize(
    "changes",
    [
        {"contact_preferred_foot": "either"},
        {"strike_ankle_lateral_m": 0.0},
        {"receive_ankle_lateral_m": float("nan")},
        {"receive_ankle_lateral_m": 0.25},
    ],
)
def test_invalid_contact_ablations_rejected_before_world_loading(tmp_path: Path, changes):
    with pytest.raises(ValueError, match="ablation"):
        run_probe(
            asset_root=tmp_path / "missing-assets",
            output=tmp_path / "output",
            active=True,
            duration=12,
            four_vs_four=True,
            **changes,
        )
    assert not (tmp_path / "output").exists()
