"""Source-window admission mocks; neither compact tracking nor skill evidence."""

from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.full_entry_forecast import (
    FullEntryForecaster,
    run_entry_suffix,
    validate_source_window,
)


@pytest.mark.parametrize(
    "window", [(285, 767), (466, 766), [466, 767], (466.0, 767), (True, 767), (), (466, 767, 800)]
)
def test_unqualified_window_rejected_before_body_or_model(window):
    kwargs = dict(
        model_root=Path("unused"),
        source_poses=np.empty((0, 36)),
        source_evidence_hash="unused",
        affordance=None,
        law=None,
        source_window=window,
    )
    with pytest.raises(ValueError, match="source-event compact window"):
        run_entry_suffix(None, None, **kwargs)
    with pytest.raises(ValueError, match="source-event compact window"):
        FullEntryForecaster(world_hash="unused").forecast(None, None, **kwargs)


@pytest.mark.parametrize(
    "mode,offset",
    [
        ("legacy", (0.0, -0.025, -0.025)),
        ("matched_position", (0.0, 0.0, 0.0)),
        ("matched_velocity", (0.0, 0.0, 0.0)),
    ],
)
def test_compact_cannot_inherit_old_temporal_c_bump_or_new_bridge(mode, offset):
    kwargs = dict(
        model_root=Path("unused"),
        source_poses=np.empty((0, 36)),
        source_evidence_hash="unused",
        affordance=None,
        law=None,
        source_window=(466, 767),
        entry_reference_mode=mode,
        contact_reference_offset_m=offset,
    )
    with pytest.raises(ValueError, match="original50 entry and zero C"):
        run_entry_suffix(None, None, **kwargs)
    with pytest.raises(ValueError, match="original50 entry and zero C"):
        FullEntryForecaster(world_hash="unused").forecast(None, None, **kwargs)


def test_only_original_and_single_compact_protocol_admitted():
    validate_source_window((286, 767), "legacy", (0.0, -0.025, -0.025))
    validate_source_window((286, 767), "matched_position", (0.0, 0.0, 0.0))
    validate_source_window((466, 767), "legacy", (0.0, 0.0, 0.0))
