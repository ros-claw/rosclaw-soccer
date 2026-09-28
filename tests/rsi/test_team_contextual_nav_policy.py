"""Frozen contextual navigation cannot infer future contact or bypass safety."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.team_contextual_nav_policy import (
    measured_entry_features,
    select_evidence_arm,
)


def test_measured_entry_is_bilateral_and_causal() -> None:
    features = measured_entry_features(
        ball=(2.0, -0.4, 0.11),
        ball_vx=-0.5,
        feet=((1.0, 0.0, 0.2), (1.2, -0.3, 0.1)),
    )
    assert np.allclose(features, (1.0, -0.4, 0.1, -0.5))
    with pytest.raises(ValueError):
        measured_entry_features(
            ball=(2.0, -0.4, 0.11),
            ball_vx=float("nan"),
            feet=((1.0, 0.0, 0.2), (1.2, -0.3, 0.1)),
        )


def test_selector_requires_all_three_neighbors_safe_and_positive() -> None:
    model = {
        "training_features": [(float(i), 0.0, 0.1, -0.5) for i in range(24)],
        "arm_names": ["safe", "unsafe"],
        "training_outcomes": {
            "safe": [{"safe": True, "useful_pass": True, "foot_contact": True} for _ in range(24)],
            "unsafe": [
                {"safe": i != 0, "useful_pass": i != 0, "foot_contact": i != 0} for i in range(24)
            ],
        },
    }
    arm, nearby = select_evidence_arm(feature=(0.1, 0.0, 0.1, -0.5), model=model)
    assert arm == "safe"
    assert nearby == [0, 1, 2]
    model["training_outcomes"]["safe"] = [
        {"safe": True, "useful_pass": False, "foot_contact": False} for _ in range(24)
    ]
    arm, _ = select_evidence_arm(feature=(0.1, 0.0, 0.1, -0.5), model=model)
    assert arm is None
