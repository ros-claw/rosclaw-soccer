from __future__ import annotations

import inspect

import pytest

from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction
from rosclaw_soccer.training.team_pass_handshake_discovery import (
    PassReceiveTimingProbe,
    receiver_timing_corrections_sec,
    run_team_pass_handshake_discovery,
)


def test_team_handshake_timing_search_is_symmetric_and_bounded() -> None:
    values = receiver_timing_corrections_sec()
    assert values == (-0.08, -0.04, 0.0, 0.04, 0.08)
    probes = tuple(PassReceiveTimingProbe(PlaymakerPassProbeAction(), value) for value in values)
    assert len({probe.probe_hash for probe in probes}) == len(probes)


def test_team_handshake_rejects_unbounded_timing() -> None:
    with pytest.raises(ValueError, match="SIM-only envelope"):
        PassReceiveTimingProbe(PlaymakerPassProbeAction(), 0.081)


def test_team_handshake_requires_explicit_playmaker_source() -> None:
    parameters = inspect.signature(run_team_pass_handshake_discovery).parameters
    assert "source_playmaker_report_path" in parameters
    assert "rejected_repair_report_path" not in parameters
