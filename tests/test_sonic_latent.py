from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_latent import SonicLatentSchedule
from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig


def test_zero_is_bitwise_identity_including_negative_zero():
    token = np.full((1, 64), -0.0, dtype=np.float32)
    schedule = SonicLatentSchedule(((0.0,) * 64,))
    assert token.tobytes() == schedule.transform(token, 20).tobytes()
    np.testing.assert_array_equal(schedule.transform(np.ones_like(token), 0), np.ones_like(token))


@pytest.mark.parametrize("magnitude", [0.0, 0.01, 1.0, 100.0])
def test_relative_and_absolute_norm_envelopes_with_ramp(magnitude):
    token = np.full((1, 64), magnitude, dtype=np.float32)
    schedule = SonicLatentSchedule(((1.0,) * 64, (-1.0,) * 64))
    for frame in range(80):
        result = schedule.transform(token, frame)
        assert result.dtype == np.float32 and result.shape == (1, 64)
        limit = min(0.5, 0.1 * np.linalg.norm(token)) * min(1.0, (frame + 1) / 10)
        assert np.linalg.norm(result - token) <= limit + 1e-5
        np.testing.assert_array_equal(result, schedule.transform(token, frame))


def test_limits_are_not_caller_expandable_and_variant_is_explicit():
    schedule = SonicLatentSchedule(((0.0,) * 64,))
    for kwargs in ({"relative_l2_limit": 0.2}, {"absolute_l2_limit": 1.0}, {"ramp_frames": 0}):
        with pytest.raises(ValueError):
            replace(schedule, **kwargs)
    with pytest.raises(ValueError, match="low-latency"):
        SonicNavigationConfig(latent_schedule=schedule)
    SonicNavigationConfig(model_variant="low_latency", latent_schedule=schedule)


@pytest.mark.parametrize("knots", [(), ((0.0,) * 63,), ((float("nan"),) * 64,), ((1.1,) * 64,)])
def test_bad_knots(knots):
    with pytest.raises(ValueError):
        SonicLatentSchedule(knots)


def test_bad_token_or_clock():
    schedule = SonicLatentSchedule(((0.0,) * 64,))
    for token, frame in (
        (np.zeros((1, 64)), 0),
        (np.zeros((64,), dtype=np.float32), 0),
        (np.zeros((1, 64), dtype=np.float32), True),
    ):
        with pytest.raises(ValueError):
            schedule.transform(token, frame)


@pytest.mark.integration
def test_real_encoder_decoder_zero_identity_and_nonzero_physical_effect(monkeypatch):
    import os
    from pathlib import Path

    required = ("ROSCLAW_G1_ASSET_ROOT", "ROSCLAW_SONIC_MODEL_ROOT", "ROSCLAW_RECEIVING_REFERENCE")
    if any(not os.environ.get(key) for key in required):
        pytest.skip("external G1 assets, SONIC models and frozen receiving checkpoint required")
    import onnxruntime as ort

    original = ort.InferenceSession

    def single_thread(*args, **kwargs):
        options = kwargs.get("sess_options") or ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        kwargs["sess_options"] = options
        return original(*args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", single_thread)
    from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
    from rosclaw_soccer.providers.g1.receiving_sonic import ReceivingSonicOption
    from rosclaw_soccer.skills.team.independent_team_world import (
        IndependentTeamWorldScenario,
        simulate_independent_team_world,
    )
    from rosclaw_soccer.training.contact_teacher_ablation import ContactTeacherSuppression
    from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
    from rosclaw_soccer.training.receiving_classroom import (
        coached_receiving_cells,
        r0_receiving_configuration,
    )
    from rosclaw_soccer.training.role_receiving_courses import (
        ReceivingCourse,
        receiving_ball_launch,
    )

    assets = Path(os.environ[required[0]])
    models = Path(os.environ[required[1]])
    policy = NearBallResidualPolicy.load(Path(os.environ[required[2]]))
    traces, contracts = [], []
    for schedule in (
        None,
        SonicLatentSchedule(((0.0,) * 64,)),
        SonicLatentSchedule(((0.5,) * 64,)),
    ):
        fixture = collection_fixture(assets, keeper_preview=True)
        focal = "blue.playmaker"
        player = next(p for p in fixture.players if p.agent_id == focal)
        position, velocity = receiving_ball_launch(
            ReceivingCourse(focal, 921000, 0.9, 0.08),
            origin=player.origin_m,
            radius_m=fixture.goal.ball_radius_m,
        )
        motor = ReceivingSonicOption(models, focal, start_frame=5, latent_schedule=schedule)
        config, teacher = r0_receiving_configuration()
        result, trace = simulate_independent_team_world(
            asset_root=assets,
            roster=fixture.roster,
            cells=coached_receiving_cells(fixture.cells, focal_agent_id=focal),
            players=fixture.players,
            scenario=IndependentTeamWorldScenario(
                "s199.latent-qualification", position, velocity, 921000
            ),
            goal=fixture.goal,
            config=replace(config, simulation_duration_sec=5.0, motor_idle_residual_fallback=True),
            near_ball_policy=policy,
            near_ball_seed=921000,
            contact_teacher_config=teacher,
            motor_options={focal: motor},
            contact_teacher_suppression=ContactTeacherSuppression(focal, 5),
        )
        assert result.safe and not trace["full_body_motor_fault"].any()
        traces.append(trace)
        contracts.append(motor.contract_hash)
        if schedule is not None:
            records = motor.navigation.backend.latent_records
            assert len(records) == 245
            assert [r[0] for r in records] == list(range(245))
            for _, encoded, transformed in records:
                assert (
                    np.linalg.norm(transformed - encoded)
                    <= min(0.5, 0.1 * np.linalg.norm(encoded)) + 1e-5
                )
    assert len(set(contracts)) == 3
    assert traces[0].keys() == traces[1].keys() == traces[2].keys()
    for key in traces[0]:
        np.testing.assert_array_equal(traces[0][key], traces[1][key])
    physical = [key for key in traces[0] if key.endswith("_joint_position")]
    assert physical and any(not np.array_equal(traces[0][key], traces[2][key]) for key in physical)
