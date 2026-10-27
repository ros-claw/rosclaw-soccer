import json
import os
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.team_receiving_action_dataset import import_team_action_map


def test_rejects_unsealed_or_fabricated_input():
    with pytest.raises(ValueError):
        import_team_action_map({})


def test_historical_map_if_available():
    source = os.environ.get("ROSCLAW_SOCCER_TEAM_MAP")
    if not source:
        pytest.skip("historical external simulation evidence not installed")
    path = Path(source)
    result = import_team_action_map(json.loads(path.read_text()))
    assert result["features"].shape == (12, 10)
    assert result["rewards"].shape == (12, 16)
    assert result["raw_trajectories_available"] is False
