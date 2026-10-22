import numpy as np
import pytest

from rosclaw_soccer.rsi.own_state_motor_assistance import assistance_share, blend_actual_motor


def test_exact_learner_prefix_and_smooth_standing_handoff():
    assert all(assistance_share(i, 289, 0.5) == 0 for i in range(80))
    assert 0 < assistance_share(80, 289, 0.5) < assistance_share(81, 289, 0.5) < 0.5
    assert assistance_share(100, 289, 0.5) == 0.5
    assert assistance_share(288, 289, 0.5) == 0
    assert assistance_share(287, 289, 0.5) < assistance_share(286, 289, 0.5)


def test_zero_share_never_roundtrips_and_changes_source_action():
    rng = np.random.default_rng(19)
    nominal = rng.normal(size=29).astype(np.float32)
    action = nominal + np.float32(0.01)
    out, info = blend_actual_motor(action, nominal, nominal, np.ones(29, np.float32), share=0.0)
    assert np.array_equal(out, action)
    assert info["learned_proposal_executed_without_assistance"]
    assert not info["PPO_actor_update_allowed"]


def test_bounded_oracle_blend_not_full_student_success():
    z = np.zeros(29, np.float32)
    out, info = blend_actual_motor(
        z, np.ones(29, np.float32), z, np.ones(29, np.float32), share=0.5
    )
    assert np.allclose(out, 0.1)
    assert info["oracle_authority_clipped"]
    assert not info["PPO_actor_update_allowed"]


def test_outside_authority_student_rejected_not_clipped():
    z = np.zeros(29, np.float32)
    with pytest.raises(ValueError):
        blend_actual_motor(np.ones(29, np.float32), z, z, np.ones(29, np.float32), share=0.5)
