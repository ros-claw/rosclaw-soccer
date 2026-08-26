from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.training.full_body_tracking_exam import FullBodyTrackingExamConfig


def test_full_body_tracking_exam_is_sim_only_and_hash_bound() -> None:
    config = FullBodyTrackingExamConfig()

    assert config.config_hash.startswith("sha256:")
    assert config.family_names == (
        "lefthand",
        "righthand",
        "leftjump",
        "rightjump",
        "leftstep",
        "rightstep",
    )
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)


def test_full_body_tracking_exam_rejects_partial_family_suite() -> None:
    with pytest.raises(ValueError, match="six unique"):
        FullBodyTrackingExamConfig(family_names=("leftjump",))
