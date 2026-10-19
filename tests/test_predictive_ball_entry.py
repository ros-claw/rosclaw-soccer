from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.physics.sonic_navigation_forecast import SonicNavigationForecast
from rosclaw_soccer.rsi.ball_entry_episode import run_ball_entry_episode
from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
from rosclaw_soccer.rsi.predictive_ball_entry import (
    PredictiveBallEntry,
    entry_candidates,
    score_entry_forecast,
)


def inputs():
    names = [
        "world",
        "pelvis",
        "left_ankle_roll_link",
        "left_ankle_pitch_link",
        "right_ankle_roll_link",
        "right_ankle_pitch_link",
        "elbow",
        "ball",
    ]
    model = SimpleNamespace(
        nbody=8,
        body_rootid=np.array([0, 1, 1, 1, 1, 1, 1, 7]),
        body=lambda name: SimpleNamespace(id=names.index(name)),
    )
    q = np.zeros(43)
    q[2], q[3], q[36], q[38], q[39] = 0.793, 1, 0.5, 0.13, 1
    v = np.zeros(41)
    prior = SkillEntryAffordance((0.25, 0.0), 150, 1.0, "sha256:" + "1" * 64)
    return model, q, v, prior


def forecast(*, ready=True):
    model, initial, _, _ = inputs()
    q = np.repeat(initial[None], 40, axis=0)
    if ready:
        q[10:, 0] = 0.25
    arrays = dict(
        canonical_qpos=q,
        canonical_qvel=np.zeros((40, 41)),
        target_body_canonical_qpos_after_substep=np.repeat(q[:, None], 10, axis=1),
        target_body_canonical_qvel_after_substep=np.zeros((40, 10, 41)),
        ball_contact_force_by_body_n=np.zeros((40, 10, model.nbody)),
        pelvis_z_per_substep_m=np.full((40, 10), 0.793),
    )
    return SonicNavigationForecast(
        arrays,
        np.zeros((40, 3)),
        np.zeros((40, 10, 36)),
        np.zeros((40, 29)),
        "toy",
        "toy",
        400,
        400,
        2,
    )


def test_only_navigation_before_predicted_handoff_is_scored():
    model, _, _, prior = inputs()
    future = forecast()
    future.arrays["ball_contact_force_by_body_n"][12, :, 6] = 100
    future.arrays["ball_contact_force_by_body_n"][:, :, 0] = 200
    result = score_entry_forecast(model, future, frame=20, affordance=prior)
    assert result["first_ready_offset"] == result["evaluated_navigation_prefix_frames"] == 10
    assert result["predicted_safe"] and result["maximum_prefix_nonfoot_force_n"] == 0
    future.arrays["ball_contact_force_by_body_n"][5, :, 6] = 2
    result = score_entry_forecast(model, future, frame=20, affordance=prior)
    assert not result["predicted_safe"] and result["cost"] is None


def test_nonready_forecast_uses_whole_horizon_and_rejects_body_height_failure():
    model, _, _, prior = inputs()
    future = forecast(ready=False)
    result = score_entry_forecast(model, future, frame=20, affordance=prior)
    assert (
        result["first_ready_offset"] is None and result["evaluated_navigation_prefix_frames"] == 40
    )
    assert result["cost"] > 1
    future.arrays["pelvis_z_per_substep_m"][-1, -1] = 0.4
    assert not score_entry_forecast(model, future, frame=20, affordance=prior)["predicted_safe"]


def test_all_six_candidates_are_bounded_and_depend_on_current_velocity():
    _, q, v, prior = inputs()
    v[:2] = (2, -2)
    bank = entry_candidates(q, v, frame=20, affordance=prior)
    assert len(bank) == 6 and bank[2] == ("zero", (0, 0, 0))
    assert all(np.linalg.norm(cmd[:2]) <= 0.60000001 for _, cmd in bank)
    assert bank[-1][1][0] < 0 < bank[-1][1][1]


@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize("start", [20, 80])
def test_selection_is_deterministic_or_explicit_unsafe_fallback(unsafe, start):
    model, q, v, prior = inputs()
    body = SimpleNamespace(next_frame=start, model=model, canonical=lambda: (q, v))
    planner = PredictiveBallEntry(world_hash="sha256:" + "1" * 64)

    class FakeForecaster:
        integrated_native_steps = 0

        def forecast(self, nav, source, command):
            result = forecast()
            if unsafe:
                result.arrays["ball_contact_force_by_body_n"][0, :, 6] = 2
            self.integrated_native_steps += 400
            # The held command is for the first20 ticks; continuation is causal.
            assert len(command(q, v, 20)) == 3 and len(command(q, v, 40)) == 3
            assert command(q, v, 91) == (0, 0, 0)
            return result

    planner.forecaster = FakeForecaster()
    command, receipt = planner(None, body, prior)
    assert receipt["new_forecast_native_steps"] == 2400
    assert receipt["all_candidates_predicted_unsafe"] is unsafe
    assert receipt["selection_predicted_safe"] is not unsafe
    assert receipt["selected_index"] == (2 if unsafe else 0)
    if unsafe:
        assert command == (0, 0, 0)
    with pytest.raises(ValueError, match="one bounded planning"):
        planner(None, body, prior)


def test_fixed_control_rejects_planning_hook_before_any_allocations():
    _, _, _, prior = inputs()
    with pytest.raises(ValueError, match="fixed-reference"):
        run_ball_entry_episode(
            Path("unused"),
            Path("unused"),
            world_hash="unused",
            source_poses=np.empty((0, 36)),
            source_evidence_hash="unused",
            source_interval=(286, 767),
            affordance=prior,
            course=(0.35, -0.1, 0),
            arm="fixed_reference",
            entry_planner=lambda *args: None,
        )
